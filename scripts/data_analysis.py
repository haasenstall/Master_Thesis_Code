import numpy as np
import pandas as pd

import os
import re
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import logging.config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import AACT_DB_CONFIG, MY_DB_CONFIG, PROJECT_SETTINGS, DATA_PATHS, LOGGING_CONFIG, PLATTFORM_IDS
from db.sql_descriptive import SQL_DatabaseManager
from db.sqlalchemy_connector import SQLAlchemyConnector, DatabaseManager
from utils.plotting import MasterPlotter

# Regression packages
from sklearn.model_selection import train_test_split
from itertools import combinations
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import sparse

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

# Bayesian stack
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import pytensor.sparse as pts

# Set random seed for reproducibility
np.random.seed(42)

# optimze runtime
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_run,floatX=float32,exception_verbosity=high'
os.environ['OMP_NUM_THREADS'] = str(PROJECT_SETTINGS.get('omp_num_threads', 4))

def setup_logging(mode='basic'):
    """Setup logging configuration"""
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Get base config and modify it based on mode
    config = LOGGING_CONFIG.copy()
    
    # Ensure no console handlers anywhere
    config['disable_existing_loggers'] = True
    
    if mode == 'basic':
        config['root']['handlers'] = ['file_handler']
    elif mode == 'testing':
        config['root']['handlers'] = ['testing_handler']
        config['root']['level'] = 'DEBUG'
    elif mode == 'descriptive':
        config['root']['handlers'] = ['descriptive_handler']    
        config['root']['level'] = 'ERROR'
    else:
        raise Exception(f"Unknown logging mode: {mode}")
    
    # Apply the configuration for ALL modes
    logging.config.dictConfig(config)
    
    # Explicitly remove any console handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

def save_results(summary, filepath):
    """Save regression results to a text file."""
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        f.write(str(summary))

def req_multi_vs_single_trial(
        logger = None,
        db: SQL_DatabaseManager = None, plotter: MasterPlotter = None):
    """Compare number of trials within a request - barplot with lines for median, mean and KDE."""
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    
    logger.info("Starting req_multi_vs_single_trial analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Fetch request data
    df_requests = db.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )
    logger.info("Fetched request data")

    df_requests = df_requests.groupby('request_id').agg({'nct_id': 'count'}).reset_index()
    df_requests = df_requests.rename(columns={'nct_id': 'num_trials'})

    if df_requests is None or df_requests.empty:
        logger.error("Could not retrieve request data")
        return
    
    # Plotting
    plotter.histogram(
        df_requests,
        column='num_trials', 
        xlabel="Number of Trials Requested (per Request)",
        ylabel="Number of Requests",
        bins=30,
        kde=True,
        save_name="req_multi_vs_single_trial_histogram.png",
        path=DATA_PATHS['img']['regression']
        )
    logger.info("Completed req_multi_vs_single_trial analysis")
    return

def co_occurrence(
        logger = None, db: SQL_DatabaseManager = None, plotter: MasterPlotter = None):
    """
    Testing the co-occurrence of trials in requests. 
    Showing the connections between trials, via network map, and showing the most common pairs.
    This should show the connections between trials and that trials are not independent to each other.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    logger.info("Starting co_occurrence analysis")
    
    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Fetch request-trial link data
    request_trial_links = db.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )

    n_requests = request_trial_links['request_id'].nunique()
    n_trials = request_trial_links['nct_id'].nunique()

    # Create a co-occurrence matrix
    co_occurrence_dict = {}
    for request_id, group in request_trial_links.groupby('request_id'):
        nct_ids = group['nct_id'].tolist()
        for combo in combinations(nct_ids, 2):
            pair = tuple(sorted(combo))
            if pair in co_occurrence_dict:
                co_occurrence_dict[pair] += 1
            else:
                co_occurrence_dict[pair] = 1
    co_occurrence_df = pd.DataFrame(list(co_occurrence_dict.items()), columns=['nct_pair', 'count'])
    co_occurrence_df[['nct_id_1', 'nct_id_2']] = pd.DataFrame(co_occurrence_df['nct_pair'].tolist(), index=co_occurrence_df.index)
    co_occurrence_df = co_occurrence_df.drop(columns=['nct_pair'])
    
    # create a network graph
    G = nx.Graph()
    for _, row in co_occurrence_df.iterrows():
        G.add_edge(row['nct_id_1'], row['nct_id_2'], weight=row['count'])

    # Analyze the network graph
    logger.info("Created co-occurrence network graph")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=False, node_size=50, font_size=8, width=[w * 0.1 for w in weights], edge_color='gray', alpha=0.7, node_color= "#2D3E50")
    plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'co_occurrence_network.png'))
    plt.close()
    logger.info("Saved co-occurrence network graph")

    # Get request-trial links
    request_trials = db.get_table_data(
        columns=['request_id', 'nct_id'], 
        table_name='request_trial_links'
    )
    
    if request_trials is None or request_trials.empty:
        logger.error("Could not retrieve request-trial links data")
        return
    
    # Group trials by request to see which trials appear together
    trials_per_request = request_trials.groupby('request_id')['nct_id'].apply(list).reset_index()
    
    # Get most common trials for co-occurrence analysis
    all_trials = request_trials['nct_id'].value_counts()
    top_trials = all_trials.head(20).index.tolist()  # Top 20 most common trials
    
    logger.info(f"Analyzing co-occurrence for {len(top_trials)} most common trials")
    
    # Create trials vs trials co-occurrence matrix
    co_occurrence_matrix = pd.DataFrame(0, index=top_trials, columns=top_trials)
    
    # Count how often each pair of trials appears together in requests
    for _, row in trials_per_request.iterrows():
        trials_in_request = [trial for trial in row['nct_id'] if trial in top_trials]
        
        # Count co-occurrences for each pair of trials in this request
        for i, trial1 in enumerate(trials_in_request):
            for trial2 in trials_in_request:
                co_occurrence_matrix.loc[trial1, trial2] += 1
    
    logger.info(f"Created {co_occurrence_matrix.shape[0]}x{co_occurrence_matrix.shape[1]} co-occurrence matrix")
    
    # FIXED: Create trials vs trials heatmap
    plotter.heatmap(
        data=co_occurrence_matrix,
        xlabel='NCT IDs',
        ylabel='NCT IDs',
        annot=True,
        fmt='d',  
        save_name='trials_co_occurrence_heatmap.png',
        path=DATA_PATHS['img']['regression']
    )
    
    logger.info("Trial co-occurrence analysis completed")
    
    # Optional: Create a subset heatmap for better readability
    top_10_trials = all_trials.head(10).index.tolist()
    subset_matrix = co_occurrence_matrix.loc[top_10_trials, top_10_trials]
    
    plotter.heatmap(
        data=subset_matrix,
        xlabel='NCT IDs',
        ylabel='NCT IDs',
        annot=True,
        fmt='d',
        save_name='top10_trials_co_occurrence_heatmap.png',
        path=DATA_PATHS['img']['regression']
    )
    
    logger.info("Co-occurrence heatmaps saved")

def extract_tree_head(tree_number, level=2):
    if pd.isna(tree_number) or tree_number == '' or tree_number is None:
        return None

    # Convert to string and strip whitespace
    tree_str = str(tree_number).strip()
    if tree_str == '' or tree_str.lower() in ['nan', 'none', 'null']:
        return None

    parts = tree_str.split('.')
    if len(parts) >= level:
        return '.'.join(parts[:level])
    else:
        return '.'.join(parts) if parts else None

def co_occur_tree_heads(
        logger = None, db: SQL_DatabaseManager = None, plotter: MasterPlotter = None):
    """Testing the co-occurrence of tree heads in trials"""
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    
    logger.info("Starting co_occur_tree_heads analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Fetch mesh terms with tree numbers
    mesh_terms = db.get_table_data(
        columns=['downcase_mesh_term', 'tree_number'],
        table_name='mesh_terms'
    )
    
    # Fetch conditions and interventions with mesh terms
    conditions = db.get_table_data(
        columns=['nct_id', 'condition as downcase_mesh_term'],
        table_name='conditions'
    )
    interventions = db.get_table_data(
        columns=['nct_id', 'intervention as downcase_mesh_term'],
        table_name='interventions_mesh_terms'
    )
    
    # Combine conditions and interventions
    df = pd.concat([conditions, interventions], ignore_index=True)
    
    # Merge with mesh terms to get tree numbers
    df_mesh_terms = df.merge(mesh_terms, on='downcase_mesh_term', how='inner')
    
    # Extract tree heads and FILTER OUT None values
    df_mesh_terms['tree_head'] = df_mesh_terms['tree_number'].apply(lambda x: extract_tree_head(x, level=2))
    
    # FIXED: Remove rows where tree_head is None
    df_mesh_terms = df_mesh_terms.dropna(subset=['tree_head'])
    logger.info(f"After filtering None tree heads: {len(df_mesh_terms)} records")
    
    if df_mesh_terms.empty:
        logger.warning("No valid tree heads found after filtering")
        return
    
    # Create a co-occurrence matrix for tree heads
    co_occurrence_dict = {}
    for nct_id, group in df_mesh_terms.groupby('nct_id'):
        tree_heads = group['tree_head'].unique().tolist()
        
        # FIXED: Additional safety check - remove any None values that might slip through
        tree_heads = [th for th in tree_heads if th is not None]
        
        if len(tree_heads) < 2:  # Need at least 2 tree heads for co-occurrence
            continue
            
        for combo in combinations(tree_heads, 2):
            # FIXED: Double-check that neither element is None before sorting
            if combo[0] is not None and combo[1] is not None:
                pair = tuple(sorted(combo))
                if pair in co_occurrence_dict:
                    co_occurrence_dict[pair] += 1
                else:
                    co_occurrence_dict[pair] = 1

    if not co_occurrence_dict:
        logger.warning("No co-occurrences found")
        return

    # Get all unique tree heads for matrix creation
    all_tree_heads = df_mesh_terms['tree_head'].unique()
    all_tree_heads = [th for th in all_tree_heads if th is not None]  # Final safety check
    
    if not all_tree_heads:
        logger.warning("No valid tree heads found")
        return

    co_occurrence_matrix = pd.DataFrame(0, index=all_tree_heads, columns=all_tree_heads)
    
    for (head1, head2), count in co_occurrence_dict.items():
        co_occurrence_matrix.loc[head1, head2] = count
        co_occurrence_matrix.loc[head2, head1] = count

    # Plot the co-occurrence matrix head as heatmap
    top_heads = co_occurrence_matrix.sum(axis=1).sort_values(ascending=False).head(20).index.tolist()
    
    if not top_heads:
        logger.warning("No top heads found for plotting")
        return
        
    top_matrix = co_occurrence_matrix.loc[top_heads, top_heads]

    plotter.heatmap(
        data=top_matrix,
        xlabel='Tree Heads',
        ylabel='Tree Heads',
        annot=True,
        fmt='d',
        save_name='tree_heads_co_occurrence_heatmap.png',
        path=DATA_PATHS['img']['regression']
    )

    logger.info("Co-occurrence heatmap saved")

    # Create a network graph for tree heads
    network_graph = nx.Graph()
    for (head1, head2), count in co_occurrence_dict.items():
        network_graph.add_edge(head1, head2, weight=count)

    logger.info("Co-occurrence network graph created")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(network_graph, k=0.1)
    weights = [network_graph[u][v]['weight'] for u, v in network_graph.edges()]
    nx.draw(network_graph, pos, with_labels=False, node_size=300, font_size=10, width=[w * 0.1 for w in weights], edge_color='gray', alpha=0.7, node_color= "#2D3E50")
    plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'tree_heads_co_occurrence_network.png'))
    plt.close()
    logger.info("Co-occurrence network graph saved")

def requested_ncts(
        logger = None, db: SQL_DatabaseManager = None) -> List[str]:
    """Fetch all requested NCT IDs from the database."""
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    
    logger.info("Fetching requested NCT IDs")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")

    df_links = db.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )
    df_trials = db.get_table_data(
        columns=['nct_id'],
        table_name='clinical_trials'
    )

    for trial in df_trials:
        if trial['nct_id'] in df_links['nct_id'].values:
            trial['requested'] = 1
        else:
            trial['requested'] = 0

    logger.info("Fetched and marked requested NCT IDs")
    return df_trials

def data_collection(logger = None, db: SQL_DatabaseManager = None, plotter: MasterPlotter = None):
    """Compare number of qualifiers per trial - barplot with lines for median, mean and KDE."""
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    
    logger.info("Starting multi_qualifier analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Fetch qualifier data
    df_conditions = db.get_table_data(
        columns=['nct_id', 'condition as downcase_mesh_term'],
        table_name='conditions'
    )
    logger.info("Fetched condition data")

    df_interventions = db.get_table_data(
        columns=['nct_id', 'intervention as downcase_mesh_term'],
        table_name='interventions_mesh_terms'
    )
    logger.info("Fetched intervention data")

    mesh_terms = db.get_table_data(
        columns=['qualifier', 'downcase_mesh_term', 'tree_number'],
        table_name='mesh_terms'
    )

    request_links = db.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )

    # qualifier and tree_number counts per trial
    df = pd.concat([df_conditions, df_interventions], ignore_index=True)
    df_mesh_terms = df.merge(mesh_terms, on='downcase_mesh_term', how='inner', )

    qualifier_counts = df_mesh_terms.groupby('nct_id').agg({'qualifier': 'nunique', 'tree_number': 'nunique'}).reset_index()
    logger.info("Fetched qualifier counts")

    # Plotting
    plotter.histogram(
        data=qualifier_counts,
        column='nct_id',  
        xlabel="Number of Qualifiers (per Trial)",
        ylabel="Number of Trials",
        bins=30,
        kde=True,
        save_name="multi_qualifier_histogram.png",
        path=DATA_PATHS['img']['regression']
    )
    logger.info("Completed multi_qualifier analysis")
    plotter.histogram(
        data=qualifier_counts,
        column='nct_id', 
        xlabel="Number of Tree Numbers (per Trial)",
        ylabel="Number of Trials",
        bins=30,
        kde=True,
        save_name="multi_tree_number_histogram.png",
        path=DATA_PATHS['img']['regression']
    )
    logger.info("Completed multi_tree_number analysis")

    # Completeness
    df_completeness = pd.read_csv(os.path.join(DATA_PATHS['descriptive'], 'trial_completeness_batch.csv'))
    logger.info("Calculated completeness data")

    # document counts per trial
    document_counts = db.get_table_data(
        columns=['nct_id', 'document_type'],
        table_name='documents'
    )
    if document_counts is None or document_counts.empty:
        logger.error("Could not retrieve document counts")
        return
    
    document_counts = document_counts.groupby('nct_id').agg({'document_type': 'nunique'}).reset_index()
    document_counts = document_counts.rename(columns={'document_type': 'document_count'})
    
    logger.info("Fetched document counts")
    # Merge document counts into completeness data
    df_completeness = pd.merge(df_completeness, document_counts, on='nct_id', how='left')

    # Fill NaN values with 0
    df_completeness['document_count'] = df_completeness['document_count'].fillna(0).astype(int)
    logger.info("Merged document counts into completeness data")

    # enrollment
    enrollment_data = db.get_table_data(
        columns=['nct_id', 'enrollment'],
        table_name='clinical_trials'
    )

    if enrollment_data is None or enrollment_data.empty:
        logger.error("Could not retrieve enrollment data")
        return
    
    enrollment_data['enrollment'] = pd.to_numeric(enrollment_data['enrollment'], errors='coerce')
    logger.info("Fetched enrollment data")

    # Merge completeness and enrollment data
    merged_data = pd.merge(df_completeness, enrollment_data, on='nct_id', how='inner')

    # merge with qualifier counts
    merged_data = pd.merge(merged_data, qualifier_counts, on='nct_id', how='inner')

    return merged_data, request_links

def baseline_analysis(logger = None, db: SQL_DatabaseManager = None, plotter: MasterPlotter = None):
    """Perform baseline logistic regression analysis with comprehensive visualizations."""
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    
    logger.info("Starting baseline logistic regression analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Get merged data
    merged_data, request_links = data_collection(logger, db, plotter)
    if merged_data is None or merged_data.empty:
        logger.error("Merged data is None or empty")
        return

    # Mark requested trials
    merged_data['requested'] = merged_data['nct_id'].isin(request_links['nct_id']).astype(int)

    # Prepare data for regression
    regression_data = merged_data.dropna(subset=['enrollment', 'completeness_percentage', 'qualifier', 'tree_number', 'requested'])
    regression_data = regression_data[regression_data['enrollment'] > 0]
    
    # Log-transform enrollment for better distribution
    regression_data['log_enrollment'] = np.log1p(regression_data['enrollment'])

    logger.info(f"Analysis dataset: {len(regression_data)} trials, {regression_data['requested'].sum()} requested")

    # 1. EXPLORATORY PLOTS - Feature comparison between groups
    features_to_analyze = ['completeness_percentage', 'qualifier', 'tree_number', 'log_enrollment']
    plotter.binary_feature_comparison(
        data=regression_data,
        features=features_to_analyze,
        target='requested',
        save_name='baseline_feature_comparison.png',
        path=DATA_PATHS['img']['regression']
    )

    # 2. INDIVIDUAL LOGISTIC REGRESSION PLOTS
    # Plot for completeness
    plotter.logistic_regression_plot(
        data=regression_data,
        x='completeness_percentage',
        y='requested',
        xlabel='Data Completeness (%)',
        ylabel='Probability of Being Requested',
        save_name='logistic_completeness.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for document counts
    plotter.logistic_regression_plot(
        data=regression_data,
        x='document_count',
        y='requested',
        xlabel='Number of Documents',
        ylabel='Probability of Being Requested',
        save_name='logistic_documents.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for qualifiers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='qualifier',
        y='requested',
        xlabel='Number of Qualifiers',
        ylabel='Probability of Being Requested',
        save_name='logistic_qualifiers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for tree numbers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='tree_number',
        y='requested',
        xlabel='Number of Tree Numbers',
        ylabel='Probability of Being Requested',
        save_name='logistic_tree_numbers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for enrollment
    plotter.logistic_regression_plot(
        data=regression_data,
        x='log_enrollment',
        y='requested',
        xlabel='Log(Enrollment + 1)',
        ylabel='Probability of Being Requested',
        save_name='logistic_enrollment.png',
        path=DATA_PATHS['img']['regression']
    )

    # 3. FIT MULTIPLE LOGISTIC REGRESSION MODEL
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_curve, auc, confusion_matrix, classification_report,
                                roc_auc_score, brier_score_loss)

    # Prepare features
    X = regression_data[['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']]
    y = regression_data['requested']

    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
    log_reg.fit(X, y)

    # Make predictions
    y_pred = log_reg.predict(X)
    y_pred_proba = log_reg.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y, y_pred)
    brier_score = brier_score_loss(y, y_pred_proba)

    # 4. COMPREHENSIVE CLASSIFICATION SUMMARY PLOT
    model_results = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'coefficients': {
            'completeness_percentage': log_reg.coef_[0][0],
            'qualifier': log_reg.coef_[0][1], 
            'log_enrollment': log_reg.coef_[0][2]
        },
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': y_pred_proba.tolist(),
        'actual_labels': y.tolist(),
        'log_likelihood': -brier_score * len(y),  # Approximate
        'aic': 2 * 4 - 2 * (-brier_score * len(y)),  # Approximate AIC
        'n_samples': len(y),
        'n_positive': y.sum(),
        'n_negative': len(y) - y.sum(),
        'brier_score': brier_score
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='baseline_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    # 5. PROBABILITY CALIBRATION PLOT
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='baseline_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    # 6. SAVE STATISTICAL RESULTS
    # Use statsmodels for detailed statistical output
    X_sm = sm.add_constant(X)  # Add intercept
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=0)

    # Save detailed results
    results_path = os.path.join(DATA_PATHS['results'], 'baseline_logistic_regression_summary.txt')
    save_results(result.summary(), results_path)
    logger.info(f"Saved detailed regression summary to {results_path}")

    # Save model performance summary
    performance_summary = f"""
        BASELINE LOGISTIC REGRESSION ANALYSIS RESULTS
        =============================================

        Dataset Information:
        - Total trials analyzed: {len(y)}
        - Requested trials: {y.sum()} ({y.mean()*100:.1f}%)
        - Non-requested trials: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)

        Model Performance:
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}
        - AUC-ROC: {auc_score:.3f}
        - Brier Score: {brier_score:.3f}

        Feature Coefficients:
        - Data Completeness: {log_reg.coef_[0][0]:.3f}
        - Number of Qualifiers: {log_reg.coef_[0][1]:.3f}
        - Log Enrollment: {log_reg.coef_[0][2]:.3f}
        - Intercept: {log_reg.intercept_[0]:.3f}

        Interpretation:
        - Higher completeness {"increases" if log_reg.coef_[0][0] > 0 else "decreases"} request probability
        - More qualifiers {"increase" if log_reg.coef_[0][1] > 0 else "decrease"} request probability  
        - Larger enrollment {"increases" if log_reg.coef_[0][2] > 0 else "decreases"} request probability

        Confusion Matrix:
        {conf_matrix}

        Classification Report:
        {classification_report(y, y_pred)}
    """

    performance_path = os.path.join(DATA_PATHS['results'], 'baseline_performance_summary.txt')
    save_results(performance_summary, performance_path)
    logger.info(f"Saved performance summary to {performance_path}")

    logger.info("Baseline logistic regression analysis completed with visualizations")
    
    return result, model_results

def build_weight_matrix(reg_data: pd.DataFrame, request_links: pd.DataFrame, logger=None) -> Tuple[sparse.csr_matrix, Dict[Any, int], Dict[str, Any]]:
    """
    Build a weighted matrix to get the relations between trials and requests.
    (N x R) matrix where N is number of trials and R is number of requests.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Building weight matrix for trials and requests")

    min_trials_per_request = 1

    # align trial order
    trial_ids = reg_data['nct_id'].tolist()
    trial_index = {tid: i for i, tid in enumerate(trial_ids)}

    # pool requests with max(min_trials_per_request) trials
    request_counts = request_links['request_id'].value_counts()
    valid_requests = request_counts[request_counts >= min_trials_per_request].index.tolist()
    req_links = request_links.copy()
    
    # Count how many requests get pooled
    n_pooled = len(request_counts) - len(valid_requests)
    req_links.loc[~req_links['request_id'].isin(valid_requests), 'request_id'] = '_OTHER_'

    # list requests
    req_ids = sorted(req_links['request_id'].unique())
    req_index = {rid: j for j, rid in enumerate(req_ids)}
    R, N = len(req_ids), len(trial_ids)

    logger.info(f"Matrix dimensions: {N} trials x {R} requests")

    # trial with request list
    trial_with_req_list = req_links.groupby('nct_id')['request_id'].apply(list).to_dict()

    # build matrix
    rows, cols, data = [], [], []
    with_membership = 0
    for trial in trial_ids:
        lst = trial_with_req_list.get(trial, [])
        if lst:
            with_membership += 1
            w = 1.0 / len(lst)
            i = trial_ids.index(trial)
            for req in lst:
                j = req_index[req]
                rows.append(i)
                cols.append(j)
                data.append(w)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, R), dtype=np.float32)
    logger.info(f"Weight matrix built with {W.nnz} non-zero entries")
    
    info = {
        'n_trials': N,
        'n_requests': R,
        'n_with_membership': with_membership,
        'density': W.nnz / (N * R),
        'pooled': n_pooled,  # Added missing key
        'rows_with_membership': with_membership  # Added missing key (alias)
    }
    return W, trial_index, info

def multilevel_logistic_regression(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None, random_state=42, test_size=0.3):
    """
    Multilevel-Membership hierarchical logistic Regression via fast MAP (ridge) approximation.
    - Row = trial
    - y = requested (0/1)
    - X = features (completeness, document_count, qualifiers, tree_number, enrollment)
    - W = weight matrix (trials x requests)
    - Design = [X | W], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Get merged data
    merged_data, request_links = data_collection(logger, db, plotter)
    if merged_data is None or merged_data.empty:
        logger.error("Merged data is None or empty")
        return
    
    # Mark requested trials
    merged_data['requested'] = merged_data['nct_id'].isin(request_links['nct_id']).astype(int)
    regression_data = merged_data.dropna(subset=['enrollment', 'completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'requested'])
    regression_data = regression_data[regression_data['enrollment'] > 0]
    regression_data['log_enrollment'] = np.log1p(regression_data['enrollment'])

    logger.info(f"Analysis dataset: {len(regression_data)} trials, {regression_data['requested'].sum()} requested")

    # 1. EXPLORATORY PLOTS - Feature comparison between groups
    features_to_analyze = ['completeness_percentage', 'qualifier', 'tree_number', 'log_enrollment']
    plotter.binary_feature_comparison(
        data=regression_data,
        features=features_to_analyze,
        target='requested',
        save_name='baseline_feature_comparison.png',
        path=DATA_PATHS['img']['regression']
    )

    # 2. INDIVIDUAL LOGISTIC REGRESSION PLOTS
    # Plot for completeness
    plotter.logistic_regression_plot(
        data=regression_data,
        x='completeness_percentage',
        y='requested',
        xlabel='Data Completeness (%)',
        ylabel='Probability of Being Requested',
        save_name='logistic_completeness.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for document counts
    plotter.logistic_regression_plot(
        data=regression_data,
        x='document_count',
        y='requested',
        xlabel='Number of Documents',
        ylabel='Probability of Being Requested',
        save_name='logistic_documents.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for qualifiers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='qualifier',
        y='requested',
        xlabel='Number of Qualifiers',
        ylabel='Probability of Being Requested',
        save_name='logistic_qualifiers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for tree numbers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='tree_number',
        y='requested',
        xlabel='Number of Tree Numbers',
        ylabel='Probability of Being Requested',
        save_name='logistic_tree_numbers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for enrollment
    plotter.logistic_regression_plot(
        data=regression_data,
        x='log_enrollment',
        y='requested',
        xlabel='Log(Enrollment + 1)',
        ylabel='Probability of Being Requested',
        save_name='logistic_enrollment.png',
        path=DATA_PATHS['img']['regression']
    )

    # 3. FIT MULTIPLE LOGISTIC REGRESSION MODEL
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_curve, auc, confusion_matrix, classification_report,
                                roc_auc_score, brier_score_loss)

    # Prepare features
    X = regression_data[['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']]
    y = regression_data['requested']

    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
    log_reg.fit(X, y)

    # Make predictions
    y_pred = log_reg.predict(X)
    y_pred_proba = log_reg.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y, y_pred)
    brier_score = brier_score_loss(y, y_pred_proba)

    # 4. COMPREHENSIVE CLASSIFICATION SUMMARY PLOT
    model_results = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'coefficients': {
            'completeness_percentage': log_reg.coef_[0][0],
            'qualifier': log_reg.coef_[0][1], 
            'log_enrollment': log_reg.coef_[0][2]
        },
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': y_pred_proba.tolist(),
        'actual_labels': y.tolist(),
        'log_likelihood': -brier_score * len(y),  # Approximate
        'aic': 2 * 4 - 2 * (-brier_score * len(y)),  # Approximate AIC
        'n_samples': len(y),
        'n_positive': y.sum(),
        'n_negative': len(y) - y.sum(),
        'brier_score': brier_score
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='baseline_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    # 5. PROBABILITY CALIBRATION PLOT
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='baseline_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    # 6. SAVE STATISTICAL RESULTS
    # Use statsmodels for detailed statistical output
    X_sm = sm.add_constant(X)  # Add intercept
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=0)

    # Save detailed results
    results_path = os.path.join(DATA_PATHS['results'], 'baseline_logistic_regression_summary.txt')
    save_results(result.summary(), results_path)
    logger.info(f"Saved detailed regression summary to {results_path}")

    # Save model performance summary
    performance_summary = f"""
        BASELINE LOGISTIC REGRESSION ANALYSIS RESULTS
        =============================================

        Dataset Information:
        - Total trials analyzed: {len(y)}
        - Requested trials: {y.sum()} ({y.mean()*100:.1f}%)
        - Non-requested trials: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)

        Model Performance:
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}
        - AUC-ROC: {auc_score:.3f}
        - Brier Score: {brier_score:.3f}

        Feature Coefficients:
        - Data Completeness: {log_reg.coef_[0][0]:.3f}
        - Number of Qualifiers: {log_reg.coef_[0][1]:.3f}
        - Log Enrollment: {log_reg.coef_[0][2]:.3f}
        - Intercept: {log_reg.intercept_[0]:.3f}

        Interpretation:
        - Higher completeness {"increases" if log_reg.coef_[0][0] > 0 else "decreases"} request probability
        - More qualifiers {"increase" if log_reg.coef_[0][1] > 0 else "decrease"} request probability  
        - Larger enrollment {"increases" if log_reg.coef_[0][2] > 0 else "decreases"} request probability

        Confusion Matrix:
        {conf_matrix}

        Classification Report:
        {classification_report(y, y_pred)}
    """

    performance_path = os.path.join(DATA_PATHS['results'], 'baseline_performance_summary.txt')
    save_results(performance_summary, performance_path)
    logger.info(f"Saved performance summary to {performance_path}")

    logger.info("Baseline logistic regression analysis completed with visualizations")
    
    return result, model_results

def build_weight_matrix(reg_data: pd.DataFrame, request_links: pd.DataFrame, logger=None) -> Tuple[sparse.csr_matrix, Dict[Any, int], Dict[str, Any]]:
    """
    Build a weighted matrix to get the relations between trials and requests.
    (N x R) matrix where N is number of trials and R is number of requests.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Building weight matrix for trials and requests")

    min_trials_per_request = 1

    # align trial order
    trial_ids = reg_data['nct_id'].tolist()
    trial_index = {tid: i for i, tid in enumerate(trial_ids)}

    # pool requests with max(min_trials_per_request) trials
    request_counts = request_links['request_id'].value_counts()
    valid_requests = request_counts[request_counts >= min_trials_per_request].index.tolist()
    req_links = request_links.copy()
    
    # Count how many requests get pooled
    n_pooled = len(request_counts) - len(valid_requests)
    req_links.loc[~req_links['request_id'].isin(valid_requests), 'request_id'] = '_OTHER_'

    # list requests
    req_ids = sorted(req_links['request_id'].unique())
    req_index = {rid: j for j, rid in enumerate(req_ids)}
    R, N = len(req_ids), len(trial_ids)

    logger.info(f"Matrix dimensions: {N} trials x {R} requests")

    # trial with request list
    trial_with_req_list = req_links.groupby('nct_id')['request_id'].apply(list).to_dict()

    # build matrix
    rows, cols, data = [], [], []
    with_membership = 0
    for trial in trial_ids:
        lst = trial_with_req_list.get(trial, [])
        if lst:
            with_membership += 1
            w = 1.0 / len(lst)
            i = trial_ids.index(trial)
            for req in lst:
                j = req_index[req]
                rows.append(i)
                cols.append(j)
                data.append(w)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, R), dtype=np.float32)
    logger.info(f"Weight matrix built with {W.nnz} non-zero entries")
    
    info = {
        'n_trials': N,
        'n_requests': R,
        'n_with_membership': with_membership,
        'density': W.nnz / (N * R),
        'pooled': n_pooled,  # Added missing key
        'rows_with_membership': with_membership  # Added missing key (alias)
    }
    return W, trial_index, info

def multilevel_logistic_regression(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None, random_state=42, test_size=0.3):
    """
    Multilevel-Membership hierarchical logistic Regression via fast MAP (ridge) approximation.
    - Row = trial
    - y = requested (0/1)
    - X = features (completeness, document_count, qualifiers, tree_number, enrollment)
    - W = weight matrix (trials x requests)
    - Design = [X | W], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Get merged data
    merged_data, request_links = data_collection(logger, db, plotter)
    if merged_data is None or merged_data.empty:
        logger.error("Merged data is None or empty")
        return

    # Mark requested trials
    merged_data['requested'] = merged_data['nct_id'].isin(request_links['nct_id']).astype(int)

    # Prepare data for regression
    regression_data = merged_data.dropna(subset=['enrollment', 'completeness_percentage', 'qualifier', 'tree_number', 'requested'])
    regression_data = regression_data[regression_data['enrollment'] > 0]
    
    # Log-transform enrollment for better distribution
    regression_data['log_enrollment'] = np.log1p(regression_data['enrollment'])

    logger.info(f"Analysis dataset: {len(regression_data)} trials, {regression_data['requested'].sum()} requested")

    # 1. EXPLORATORY PLOTS - Feature comparison between groups
    features_to_analyze = ['completeness_percentage', 'qualifier', 'tree_number', 'log_enrollment']
    plotter.binary_feature_comparison(
        data=regression_data,
        features=features_to_analyze,
        target='requested',
        save_name='baseline_feature_comparison.png',
        path=DATA_PATHS['img']['regression']
    )

    # 2. INDIVIDUAL LOGISTIC REGRESSION PLOTS
    # Plot for completeness
    plotter.logistic_regression_plot(
        data=regression_data,
        x='completeness_percentage',
        y='requested',
        xlabel='Data Completeness (%)',
        ylabel='Probability of Being Requested',
        save_name='logistic_completeness.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for document counts
    plotter.logistic_regression_plot(
        data=regression_data,
        x='document_count',
        y='requested',
        xlabel='Number of Documents',
        ylabel='Probability of Being Requested',
        save_name='logistic_documents.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for qualifiers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='qualifier',
        y='requested',
        xlabel='Number of Qualifiers',
        ylabel='Probability of Being Requested',
        save_name='logistic_qualifiers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for tree numbers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='tree_number',
        y='requested',
        xlabel='Number of Tree Numbers',
        ylabel='Probability of Being Requested',
        save_name='logistic_tree_numbers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for enrollment
    plotter.logistic_regression_plot(
        data=regression_data,
        x='log_enrollment',
        y='requested',
        xlabel='Log(Enrollment + 1)',
        ylabel='Probability of Being Requested',
        save_name='logistic_enrollment.png',
        path=DATA_PATHS['img']['regression']
    )

    # 3. FIT MULTIPLE LOGISTIC REGRESSION MODEL
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_curve, auc, confusion_matrix, classification_report,
                                roc_auc_score, brier_score_loss)

    # Prepare features
    X = regression_data[['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']]
    y = regression_data['requested']

    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
    log_reg.fit(X, y)

    # Make predictions
    y_pred = log_reg.predict(X)
    y_pred_proba = log_reg.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y, y_pred)
    brier_score = brier_score_loss(y, y_pred_proba)

    # 4. COMPREHENSIVE CLASSIFICATION SUMMARY PLOT
    model_results = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'coefficients': {
            'completeness_percentage': log_reg.coef_[0][0],
            'qualifier': log_reg.coef_[0][1], 
            'log_enrollment': log_reg.coef_[0][2]
        },
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': y_pred_proba.tolist(),
        'actual_labels': y.tolist(),
        'log_likelihood': -brier_score * len(y),  # Approximate
        'aic': 2 * 4 - 2 * (-brier_score * len(y)),  # Approximate AIC
        'n_samples': len(y),
        'n_positive': y.sum(),
        'n_negative': len(y) - y.sum(),
        'brier_score': brier_score
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='baseline_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    # 5. PROBABILITY CALIBRATION PLOT
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='baseline_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    # 6. SAVE STATISTICAL RESULTS
    # Use statsmodels for detailed statistical output
    X_sm = sm.add_constant(X)  # Add intercept
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=0)

    # Save detailed results
    results_path = os.path.join(DATA_PATHS['results'], 'baseline_logistic_regression_summary.txt')
    save_results(result.summary(), results_path)
    logger.info(f"Saved detailed regression summary to {results_path}")

    # Save model performance summary
    performance_summary = f"""
        BASELINE LOGISTIC REGRESSION ANALYSIS RESULTS
        =============================================

        Dataset Information:
        - Total trials analyzed: {len(y)}
        - Requested trials: {y.sum()} ({y.mean()*100:.1f}%)
        - Non-requested trials: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)

        Model Performance:
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}
        - AUC-ROC: {auc_score:.3f}
        - Brier Score: {brier_score:.3f}

        Feature Coefficients:
        - Data Completeness: {log_reg.coef_[0][0]:.3f}
        - Number of Qualifiers: {log_reg.coef_[0][1]:.3f}
        - Log Enrollment: {log_reg.coef_[0][2]:.3f}
        - Intercept: {log_reg.intercept_[0]:.3f}

        Interpretation:
        - Higher completeness {"increases" if log_reg.coef_[0][0] > 0 else "decreases"} request probability
        - More qualifiers {"increase" if log_reg.coef_[0][1] > 0 else "decrease"} request probability  
        - Larger enrollment {"increases" if log_reg.coef_[0][2] > 0 else "decreases"} request probability

        Confusion Matrix:
        {conf_matrix}

        Classification Report:
        {classification_report(y, y_pred)}
    """

    performance_path = os.path.join(DATA_PATHS['results'], 'baseline_performance_summary.txt')
    save_results(performance_summary, performance_path)
    logger.info(f"Saved performance summary to {performance_path}")

    logger.info("Baseline logistic regression analysis completed with visualizations")
    
    return result, model_results

def build_weight_matrix(reg_data: pd.DataFrame, request_links: pd.DataFrame, logger=None) -> Tuple[sparse.csr_matrix, Dict[Any, int], Dict[str, Any]]:
    """
    Build a weighted matrix to get the relations between trials and requests.
    (N x R) matrix where N is number of trials and R is number of requests.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Building weight matrix for trials and requests")

    min_trials_per_request = 1

    # align trial order
    trial_ids = reg_data['nct_id'].tolist()
    trial_index = {tid: i for i, tid in enumerate(trial_ids)}

    # pool requests with max(min_trials_per_request) trials
    request_counts = request_links['request_id'].value_counts()
    valid_requests = request_counts[request_counts >= min_trials_per_request].index.tolist()
    req_links = request_links.copy()
    
    # Count how many requests get pooled
    n_pooled = len(request_counts) - len(valid_requests)
    req_links.loc[~req_links['request_id'].isin(valid_requests), 'request_id'] = '_OTHER_'

    # list requests
    req_ids = sorted(req_links['request_id'].unique())
    req_index = {rid: j for j, rid in enumerate(req_ids)}
    R, N = len(req_ids), len(trial_ids)

    logger.info(f"Matrix dimensions: {N} trials x {R} requests")

    # trial with request list
    trial_with_req_list = req_links.groupby('nct_id')['request_id'].apply(list).to_dict()

    # build matrix
    rows, cols, data = [], [], []
    with_membership = 0
    for trial in trial_ids:
        lst = trial_with_req_list.get(trial, [])
        if lst:
            with_membership += 1
            w = 1.0 / len(lst)
            i = trial_ids.index(trial)
            for req in lst:
                j = req_index[req]
                rows.append(i)
                cols.append(j)
                data.append(w)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, R), dtype=np.float32)
    logger.info(f"Weight matrix built with {W.nnz} non-zero entries")
    
    info = {
        'n_trials': N,
        'n_requests': R,
        'n_with_membership': with_membership,
        'density': W.nnz / (N * R),
        'pooled': n_pooled,  # Added missing key
        'rows_with_membership': with_membership  # Added missing key (alias)
    }
    return W, trial_index, info

def multilevel_logistic_regression(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None, random_state=42, test_size=0.3):
    """
    Multilevel-Membership hierarchical logistic Regression via fast MAP (ridge) approximation.
    - Row = trial
    - y = requested (0/1)
    - X = features (completeness, document_count, qualifiers, tree_number, enrollment)
    - W = weight matrix (trials x requests)
    - Design = [X | W], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Get merged data
    merged_data, request_links = data_collection(logger, db, plotter)
    if merged_data is None or merged_data.empty:
        logger.error("Merged data is None or empty")
        return

    # Mark requested trials
    merged_data['requested'] = merged_data['nct_id'].isin(request_links['nct_id']).astype(int)

    # Prepare data for regression
    regression_data = merged_data.dropna(subset=['enrollment', 'completeness_percentage', 'qualifier', 'tree_number', 'requested'])
    regression_data = regression_data[regression_data['enrollment'] > 0]
    
    # Log-transform enrollment for better distribution
    regression_data['log_enrollment'] = np.log1p(regression_data['enrollment'])

    logger.info(f"Analysis dataset: {len(regression_data)} trials, {regression_data['requested'].sum()} requested")

    # 1. EXPLORATORY PLOTS - Feature comparison between groups
    features_to_analyze = ['completeness_percentage', 'qualifier', 'tree_number', 'log_enrollment']
    plotter.binary_feature_comparison(
        data=regression_data,
        features=features_to_analyze,
        target='requested',
        save_name='baseline_feature_comparison.png',
        path=DATA_PATHS['img']['regression']
    )

    # 2. INDIVIDUAL LOGISTIC REGRESSION PLOTS
    # Plot for completeness
    plotter.logistic_regression_plot(
        data=regression_data,
        x='completeness_percentage',
        y='requested',
        xlabel='Data Completeness (%)',
        ylabel='Probability of Being Requested',
        save_name='logistic_completeness.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for document counts
    plotter.logistic_regression_plot(
        data=regression_data,
        x='document_count',
        y='requested',
        xlabel='Number of Documents',
        ylabel='Probability of Being Requested',
        save_name='logistic_documents.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for qualifiers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='qualifier',
        y='requested',
        xlabel='Number of Qualifiers',
        ylabel='Probability of Being Requested',
        save_name='logistic_qualifiers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for tree numbers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='tree_number',
        y='requested',
        xlabel='Number of Tree Numbers',
        ylabel='Probability of Being Requested',
        save_name='logistic_tree_numbers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for enrollment
    plotter.logistic_regression_plot(
        data=regression_data,
        x='log_enrollment',
        y='requested',
        xlabel='Log(Enrollment + 1)',
        ylabel='Probability of Being Requested',
        save_name='logistic_enrollment.png',
        path=DATA_PATHS['img']['regression']
    )

    # 3. FIT MULTIPLE LOGISTIC REGRESSION MODEL
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_curve, auc, confusion_matrix, classification_report,
                                roc_auc_score, brier_score_loss)

    # Prepare features
    X = regression_data[['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']]
    y = regression_data['requested']

    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
    log_reg.fit(X, y)

    # Make predictions
    y_pred = log_reg.predict(X)
    y_pred_proba = log_reg.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y, y_pred)
    brier_score = brier_score_loss(y, y_pred_proba)

    # 4. COMPREHENSIVE CLASSIFICATION SUMMARY PLOT
    model_results = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'coefficients': {
            'completeness_percentage': log_reg.coef_[0][0],
            'qualifier': log_reg.coef_[0][1], 
            'log_enrollment': log_reg.coef_[0][2]
        },
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': y_pred_proba.tolist(),
        'actual_labels': y.tolist(),
        'log_likelihood': -brier_score * len(y),  # Approximate
        'aic': 2 * 4 - 2 * (-brier_score * len(y)),  # Approximate AIC
        'n_samples': len(y),
        'n_positive': y.sum(),
        'n_negative': len(y) - y.sum(),
        'brier_score': brier_score
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='baseline_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    # 5. PROBABILITY CALIBRATION PLOT
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='baseline_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    # 6. SAVE STATISTICAL RESULTS
    # Use statsmodels for detailed statistical output
    X_sm = sm.add_constant(X)  # Add intercept
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=0)

    # Save detailed results
    results_path = os.path.join(DATA_PATHS['results'], 'baseline_logistic_regression_summary.txt')
    save_results(result.summary(), results_path)
    logger.info(f"Saved detailed regression summary to {results_path}")

    # Save model performance summary
    performance_summary = f"""
        BASELINE LOGISTIC REGRESSION ANALYSIS RESULTS
        =============================================

        Dataset Information:
        - Total trials analyzed: {len(y)}
        - Requested trials: {y.sum()} ({y.mean()*100:.1f}%)
        - Non-requested trials: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)

        Model Performance:
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}
        - AUC-ROC: {auc_score:.3f}
        - Brier Score: {brier_score:.3f}

        Feature Coefficients:
        - Data Completeness: {log_reg.coef_[0][0]:.3f}
        - Number of Qualifiers: {log_reg.coef_[0][1]:.3f}
        - Log Enrollment: {log_reg.coef_[0][2]:.3f}
        - Intercept: {log_reg.intercept_[0]:.3f}

        Interpretation:
        - Higher completeness {"increases" if log_reg.coef_[0][0] > 0 else "decreases"} request probability
        - More qualifiers {"increase" if log_reg.coef_[0][1] > 0 else "decrease"} request probability  
        - Larger enrollment {"increases" if log_reg.coef_[0][2] > 0 else "decreases"} request probability

        Confusion Matrix:
        {conf_matrix}

        Classification Report:
        {classification_report(y, y_pred)}
    """

    performance_path = os.path.join(DATA_PATHS['results'], 'baseline_performance_summary.txt')
    save_results(performance_summary, performance_path)
    logger.info(f"Saved performance summary to {performance_path}")

    logger.info("Baseline logistic regression analysis completed with visualizations")
    
    return result, model_results

def build_weight_matrix(reg_data: pd.DataFrame, request_links: pd.DataFrame, logger=None) -> Tuple[sparse.csr_matrix, Dict[Any, int], Dict[str, Any]]:
    """
    Build a weighted matrix to get the relations between trials and requests.
    (N x R) matrix where N is number of trials and R is number of requests.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Building weight matrix for trials and requests")

    min_trials_per_request = 1

    # align trial order
    trial_ids = reg_data['nct_id'].tolist()
    trial_index = {tid: i for i, tid in enumerate(trial_ids)}

    # pool requests with max(min_trials_per_request) trials
    request_counts = request_links['request_id'].value_counts()
    valid_requests = request_counts[request_counts >= min_trials_per_request].index.tolist()
    req_links = request_links.copy()
    
    # Count how many requests get pooled
    n_pooled = len(request_counts) - len(valid_requests)
    req_links.loc[~req_links['request_id'].isin(valid_requests), 'request_id'] = '_OTHER_'

    # list requests
    req_ids = sorted(req_links['request_id'].unique())
    req_index = {rid: j for j, rid in enumerate(req_ids)}
    R, N = len(req_ids), len(trial_ids)

    logger.info(f"Matrix dimensions: {N} trials x {R} requests")

    # trial with request list
    trial_with_req_list = req_links.groupby('nct_id')['request_id'].apply(list).to_dict()

    # build matrix
    rows, cols, data = [], [], []
    with_membership = 0
    for trial in trial_ids:
        lst = trial_with_req_list.get(trial, [])
        if lst:
            with_membership += 1
            w = 1.0 / len(lst)
            i = trial_ids.index(trial)
            for req in lst:
                j = req_index[req]
                rows.append(i)
                cols.append(j)
                data.append(w)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, R), dtype=np.float32)
    logger.info(f"Weight matrix built with {W.nnz} non-zero entries")
    
    info = {
        'n_trials': N,
        'n_requests': R,
        'n_with_membership': with_membership,
        'density': W.nnz / (N * R),
        'pooled': n_pooled,  # Added missing key
        'rows_with_membership': with_membership  # Added missing key (alias)
    }
    return W, trial_index, info

def multilevel_logistic_regression(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None, random_state=42, test_size=0.3):
    """
    Multilevel-Membership hierarchical logistic Regression via fast MAP (ridge) approximation.
    - Row = trial
    - y = requested (0/1)
    - X = features (completeness, document_count, qualifiers, tree_number, enrollment)
    - W = weight matrix (trials x requests)
    - Design = [X | W], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Get merged data
    merged_data, request_links = data_collection(logger, db, plotter)
    if merged_data is None or merged_data.empty:
        logger.error("Merged data is None or empty")
        return

    # Mark requested trials
    merged_data['requested'] = merged_data['nct_id'].isin(request_links['nct_id']).astype(int)

    # Prepare data for regression
    regression_data = merged_data.dropna(subset=['enrollment', 'completeness_percentage', 'qualifier', 'tree_number', 'requested'])
    regression_data = regression_data[regression_data['enrollment'] > 0]
    
    # Log-transform enrollment for better distribution
    regression_data['log_enrollment'] = np.log1p(regression_data['enrollment'])

    logger.info(f"Analysis dataset: {len(regression_data)} trials, {regression_data['requested'].sum()} requested")

    # 1. EXPLORATORY PLOTS - Feature comparison between groups
    features_to_analyze = ['completeness_percentage', 'qualifier', 'tree_number', 'log_enrollment']
    plotter.binary_feature_comparison(
        data=regression_data,
        features=features_to_analyze,
        target='requested',
        save_name='baseline_feature_comparison.png',
        path=DATA_PATHS['img']['regression']
    )

    # 2. INDIVIDUAL LOGISTIC REGRESSION PLOTS
    # Plot for completeness
    plotter.logistic_regression_plot(
        data=regression_data,
        x='completeness_percentage',
        y='requested',
        xlabel='Data Completeness (%)',
        ylabel='Probability of Being Requested',
        save_name='logistic_completeness.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for document counts
    plotter.logistic_regression_plot(
        data=regression_data,
        x='document_count',
        y='requested',
        xlabel='Number of Documents',
        ylabel='Probability of Being Requested',
        save_name='logistic_documents.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for qualifiers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='qualifier',
        y='requested',
        xlabel='Number of Qualifiers',
        ylabel='Probability of Being Requested',
        save_name='logistic_qualifiers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for tree numbers
    plotter.logistic_regression_plot(
        data=regression_data,
        x='tree_number',
        y='requested',
        xlabel='Number of Tree Numbers',
        ylabel='Probability of Being Requested',
        save_name='logistic_tree_numbers.png',
        path=DATA_PATHS['img']['regression']
    )

    # Plot for enrollment
    plotter.logistic_regression_plot(
        data=regression_data,
        x='log_enrollment',
        y='requested',
        xlabel='Log(Enrollment + 1)',
        ylabel='Probability of Being Requested',
        save_name='logistic_enrollment.png',
        path=DATA_PATHS['img']['regression']
    )

    # 3. FIT MULTIPLE LOGISTIC REGRESSION MODEL
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_curve, auc, confusion_matrix, classification_report,
                                roc_auc_score, brier_score_loss)

    # Prepare features
    X = regression_data[['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']]
    y = regression_data['requested']

    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=200)
    log_reg.fit(X, y)

    # Make predictions
    y_pred = log_reg.predict(X)
    y_pred_proba = log_reg.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y, y_pred)
   
    brier_score = brier_score_loss(y, y_pred_proba)

    # 4. COMPREHENSIVE CLASSIFICATION SUMMARY PLOT
    model_results = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'coefficients': {
            'completeness_percentage': log_reg.coef_[0][0],
            'qualifier': log_reg.coef_[0][1], 
            'log_enrollment': log_reg.coef_[0][2]
        },
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': y_pred_proba.tolist(),
        'actual_labels': y.tolist(),
        'log_likelihood': -brier_score * len(y),  # Approximate
        'aic': 2 * 4 - 2 * (-brier_score * len(y)),  # Approximate AIC
        'n_samples': len(y),
        'n_positive': y.sum(),
        'n_negative': len(y) - y.sum(),
        'brier_score': brier_score
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='baseline_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    # 5. PROBABILITY CALIBRATION PLOT
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='baseline_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    # 6. SAVE STATISTICAL RESULTS
    # Use statsmodels for detailed statistical output
    X_sm = sm.add_constant(X)  # Add intercept
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=0)

    # Save detailed results
    results_path = os.path.join(DATA_PATHS['results'], 'baseline_logistic_regression_summary.txt')
    save_results(result.summary(), results_path)
    logger.info(f"Saved detailed regression summary to {results_path}")

    # Save model performance summary
    performance_summary = f"""
        BASELINE LOGISTIC REGRESSION ANALYSIS RESULTS
        =============================================

        Dataset Information:
        - Total trials analyzed: {len(y)}
        - Requested trials: {y.sum()} ({y.mean()*100:.1f}%)
        - Non-requested trials: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)

        Model Performance:
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}
        - AUC-ROC: {auc_score:.3f}
        - Brier Score: {brier_score:.3f}

        Feature Coefficients:
        - Data Completeness: {log_reg.coef_[0][0]:.3f}
        - Number of Qualifiers: {log_reg.coef_[0][1]:.3f}
        - Log Enrollment: {log_reg.coef_[0][2]:.3f}
        - Intercept: {log_reg.intercept_[0]:.3f}

        Interpretation:
        - Higher completeness {"increases" if log_reg.coef_[0][0] > 0 else "decreases"} request probability
        - More qualifiers {"increase" if log_reg.coef_[0][1] > 0 else "decrease"} request probability  
        - Larger enrollment {"increases" if log_reg.coef_[0][2] > 0 else "decreases"} request probability

        Confusion Matrix:
        {conf_matrix}

        Classification Report:
        {classification_report(y, y_pred)}
    """

    performance_path = os.path.join(DATA_PATHS['results'], 'baseline_performance_summary.txt')
    save_results(performance_summary, performance_path)
    logger.info(f"Saved performance summary to {performance_path}")

    logger.info("Baseline logistic regression analysis completed with visualizations")
    
    return result, model_results

def sequential_mm_logistic_models(
        logger=None,
        db: SQL_DatabaseManager=None,
        plotter: MasterPlotter=None,
        random_state=42,
        test_size=0.3):
    """
    Fit three nested multilevel (multiple-membership) logistic regression models:
      M1: qualifier
      M2: qualifier + tree_number
      M3: qualifier + tree_number + document_count + completeness_percentage
    Each includes request-level random intercept approximation via membership matrix W.
    Outputs:
      - Betas for fixed effects
      - N train / N test / positives in test
      - Log-likelihood (test set)
      - AIC / BIC (using test set logLik)
      - DataFrame summary saved to results/sequential_mm_models_summary.csv
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)
    logger.info("Starting sequential multilevel logistic regression models")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
    if plotter is None:
        plotter = MasterPlotter()

    # Collect data
    merged_data, request_links = data_collection(logger, db, plotter)
    if merged_data is None or merged_data.empty:
        logger.error("No merged data for sequential models")
        return

    # Target
    merged_data['requested'] = merged_data['nct_id'].isin(request_links['nct_id']).astype(int)

    base_cols = ['nct_id', 'requested', 'qualifier', 'tree_number',
                 'document_count', 'completeness_percentage', 'enrollment']
    df = merged_data[base_cols].dropna(subset=['qualifier', 'tree_number',
                                               'requested', 'enrollment',
                                               'document_count',
                                               'completeness_percentage'])
    df = df[df['enrollment'] > 0]
    df['log_enrollment'] = np.log1p(df['enrollment'])  # retained if needed later

    # Build request membership matrix (random effects proxy)
    W, req_index, info_req = build_weight_matrix(df, request_links, logger)

    # Model specifications
    model_specs = [
        ("M1_qualifier",              ['qualifier']),
        ("M2_qualifier_tree",         ['qualifier', 'tree_number']),
        ("M3_full",                   ['qualifier', 'tree_number',
                                       'document_count', 'completeness_percentage'])
    ]

    rows = []
    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    for model_name, feature_list in model_specs:
        logger.info(f"Fitting {model_name} with features: {feature_list}")

        X = df[feature_list].to_numpy(dtype=np.float32)
        y = df['requested'].to_numpy(dtype=np.int32)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X).astype(np.float32)

        # Split
        X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
            Xs, y, W, test_size=test_size, stratify=y, random_state=random_state
        )

        # Design matrices
        X_train_sp = sparse.csr_matrix(X_train)
        X_test_sp = sparse.csr_matrix(X_test)
        X_design_train = sparse.hstack([X_train_sp, W_train], format='csr')
        X_design_test = sparse.hstack([X_test_sp, W_test], format='csr')

        # Fit penalized logistic (multiple-membership)
        Cs = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        clf = LogisticRegressionCV(
            Cs=Cs, cv=cv, solver="saga", penalty="l2", scoring="neg_log_loss",
            max_iter=5000, n_jobs=-1, fit_intercept=True, class_weight="balanced", refit=True
        )
        clf.fit(X_design_train, y_train)
        C_star = float(clf.C_[0])

        # Predictions & metrics (test)
        p_test = clf.predict_proba(X_design_test)[:, 1]
        ll_test = _log_likelihood_binary(y_test, p_test)
        k_params = clf.coef_.shape[1] + 1  # coefficients + intercept
        N_test = len(y_test)
        aic = 2 * k_params - 2 * ll_test
        bic = k_params * np.log(N_test) - 2 * ll_test
        auc = roc_auc_score(y_test, p_test)
        brier = brier_score_loss(y_test, p_test)

        # Extract fixed effect betas
        p_fixed = len(feature_list)
        coef_all = clf.coef_.ravel()
        beta_fixed = coef_all[:p_fixed]

        # Store row
        row = {
            'model': model_name,
            'features': '|'.join(feature_list),
            'n_train': len(y_train),
            'n_test': N_test,
            'test_requested': int(y_test.sum()),
            'test_requested_pct': y_test.mean() * 100.0,
            'log_likelihood_test': ll_test,
            'AIC_test': aic,
            'BIC_test': bic,
            'AUC_test': auc,
            'Brier_test': brier,
            'C_selected': C_star
        }
        # Add betas
        for f, b in zip(feature_list, beta_fixed):
            row[f'beta_{f}'] = b
            row[f'OR_{f}'] = np.exp(b)
        rows.append(row)

        # Save per-model coefficient table
        coef_df = pd.DataFrame({
            'feature': feature_list,
            'beta': beta_fixed,
            'OR': np.exp(beta_fixed)
        })
        coef_df.to_csv(os.path.join(
            DATA_PATHS['results'],
            f'{model_name}_fixed_effects.csv'
        ), index=False)

        # Quick text summary
        summary_txt = f"""
        {model_name} Multilevel Logistic Regression
        -------------------------------------------
        Features: {feature_list}
        Train N: {len(y_train)}
        Test N: {N_test}
        Test positives: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)
        Log-Likelihood (test): {ll_test:.3f}
        AIC (test): {aic:.3f}
        BIC (test): {bic:.3f}
        AUC (test): {auc:.3f}
        Brier (test): {brier:.3f}
        Selected C: {C_star:.3f}
        Betas (std scale):
        {chr(10).join([f'  {f}: {b:.4f} (OR={np.exp(b):.4f})' for f, b in zip(feature_list, beta_fixed)])}
        """
        save_results(summary_txt,
                     os.path.join(DATA_PATHS['results'],
                                  f'{model_name}_summary.txt'))

    # Combined summary
    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(DATA_PATHS['results'],
                               'sequential_mm_models_summary.csv'),
                  index=False)
    logger.info("Sequential multilevel models completed and summary saved.")
    return out_df

def _log_likelihood_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    """Exact log-likelihood for Bernoulli logistic model."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def runner():
    """Main runner function to execute analyses."""
    setup_logging(mode='basic')
    logger = logging.getLogger(__name__)
    logger.info("Script started")
    
    db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='regression_analysis')
    plotting = MasterPlotter()

    co_occurrence(logger, db, plotting)
    co_occur_tree_heads(logger, db, plotting)
    req_multi_vs_single_trial(logger, db, plotting)
    baseline_analysis(logger, db, plotting)
    multilevel_logistic_regression(logger, db, plotting)

    sequential_mm_logistic_models(logger, db, plotting)  # <-- added
    
    logger.info("Script finished")

if __name__== "__main__":
    runner()
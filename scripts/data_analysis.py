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

    features = ['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']
    X = regression_data[features].to_numpy(dtype=np.float32)
    y = regression_data['requested'].to_numpy(dtype=np.int32)

    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Build weight matrix
    W, req_index, info = build_weight_matrix(regression_data, request_links, logger)
    logger.info(f"Weight matrix info: {info}")

    # train test split
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
        Xs, y, W, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train = sparse.csr_matrix(X_train); X_test = sparse.csr_matrix(X_test)
    X_design_train = sparse.hstack([X_train, W_train], format='csr')
    X_design_test  = sparse.hstack([X_test,  W_test],  format='csr')

    # CV or C - log loss for probabilistic calibration
    Cs = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=Cs, cv=cv, solver="saga", penalty="l2", scoring="neg_log_loss",
        max_iter=5000, n_jobs=-1, fit_intercept=True, class_weight="balanced", refit=True
    )
    clf.fit(X_design_train, y_train)
    C_star = float(clf.C_[0])
    logger.info(f"Selected C (prior var) = {C_star}")

    p_hat = clf.predict_proba(X_design_test)[:, 1]
    auc = roc_auc_score(y_test, p_hat)  # Use y_test, not y
    ap = average_precision_score(y_test, p_hat)
    brier = brier_score_loss(y_test, p_hat)
    fpr, tpr, _ = roc_curve(y_test, p_hat)
    y_pred = (p_hat >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)  # Use y_test, not y
    cls_rep = classification_report(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Multilevel Model AUC = {auc:.3f}, AP = {ap:.3f}, Brier = {brier:.3f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{cls_rep}")

    # Extract coefficients
    coef = clf.coef_.ravel()  # length p+R
    p = len(features)
    beta_fixed = coef[:p]
    gamma_req = coef[p:]      # request effects (shrunk)

    model_results = {
        'confusion_matrix': cm.tolist(),
        'accuracy': float(accuracy),           
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': p_hat.tolist(),  # Test set predictions
        'actual_labels': y_test.tolist(),           # FIXED: Use y_test, not y
        'brier_score': float(brier),
        'n_samples': len(y_test),                   # FIXED: Use len(y_test)
        'n_positive': int(y_test.sum()),            # FIXED: Use y_test
        'n_negative': int((1 - y_test).sum()),      # FIXED: Use y_test
        'coefficients': {col: float(beta_fixed[i]) for i, col in enumerate(features)}
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='mm_map_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='mm_map_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    # 1) fixed effects table (standardized scale)
    fixed_tbl = pd.DataFrame({
        'feature': features,
        'coef_std': beta_fixed,
        'OR_per_1SD': np.exp(beta_fixed)
    })
    fixed_tbl.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_fixed_effects.csv'), index=False)

    # 2) request effects (top/bottom 30 for readability)
    inv_req_index = {j: rid for rid, j in req_index.items()}
    req_df = pd.DataFrame({
        'request_id': [inv_req_index[j] for j in range(len(gamma_req))],
        'effect_logit': gamma_req,
        'effect_OR': np.exp(gamma_req)
    }).sort_values('effect_logit')
    req_df.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_request_effects_full.csv'), index=False)

    head = req_df.tail(30); tail = req_df.head(30)
    head.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_request_effects_top30.csv'), index=False)
    tail.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_request_effects_bottom30.csv'), index=False)

    # 3) quick caterpillar plot
    try:
        plt.figure(figsize=(8, 5))
        ordered = req_df.reset_index(drop=True)
        idx = np.arange(len(ordered))
        plt.plot(idx, ordered['effect_logit'].values, '.', ms=3)
        plt.axhline(0, ls='--', c='k')
        plt.xlabel('Requests (sorted)'); plt.ylabel('Effect on log-odds')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'mm_map_request_caterpillar.png'), dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Caterpillar plot failed: {e}")

    # 4) text summary
    txt = f"""
        MAP (Ridge) Multiple-Membership Logistic (Trial-level)

        Data:
        - Total trials: {len(y)}
        - Train trials: {len(y_train)} 
        - Test trials: {len(y_test)}
        - Test requested: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)
        - Requests (columns in W): {info['n_requests']}
        - Trials with membership: {info['n_with_membership']}
        - Matrix density: {info['density']:.4f}

        Model:
        - Design = [X_std | W] (sparse CSR)
        - Penalty: L2 (Gaussian prior), solver='saga'
        - Selected C (prior variance ~ tau^2): {C_star:.3f}

        Performance (test set):
        - AUC: {auc:.3f}
        - Average Precision: {ap:.3f}
        - Brier Score: {brier:.3f}
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}

        Fixed effects (std. scale):
        {chr(10).join([f"  {feat}: {coef:.3f} (OR={np.exp(coef):.3f})" for feat, coef in zip(features, beta_fixed)])}

        Matrix Information:
        - Total trials: {info['n_trials']}
        - Total requests: {info['n_requests']}  
        - Trials with request membership: {info['n_with_membership']}
        - Matrix sparsity: {(1-info['density'])*100:.1f}% zeros

        Files saved:
        - Fixed effects: mm_map_fixed_effects.csv
        - Request effects: mm_map_request_effects_full.csv (with top/bottom 30 CSVs)
        """
    save_results(txt, os.path.join(DATA_PATHS['results'], 'mm_map_summary.txt'))

    logger.info("MAP (ridge) multilevel logistic regression finished")

    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        clf, X_design_train, y_train,
        cv=5, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label="train AUC")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="cv AUC")
    plt.legend()
    plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'mm_map_learning_curve.png'), dpi=150)
    plt.close()

    return clf, model_results, {'scaler': scaler, 'req_index': req_index, 'info': info}

def multilevel_logistic_regression_best_features(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None, random_state=42, test_size=0.3):
    """
    Multilevel logistic regression with automatic feature selection to find the best model.
    Uses forward selection based on AUC to identify optimal feature subset.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression with feature selection")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")
    if plotter is None:
        plotter = MasterPlotter()
        logger.info("Initialized MasterPlotter")

    # Get merged data (same as regular multilevel)
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

    # All available features
    all_features = ['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']
    
    # Build weight matrix (same as regular multilevel)
    W, req_index, info = build_weight_matrix(regression_data, request_links, logger)
    logger.info(f"Weight matrix info: {info}")

    y = regression_data['requested'].to_numpy(dtype=np.int32)
    
    # Feature selection using forward selection
    logger.info("Starting forward feature selection...")
    
    best_features = []
    best_auc = 0
    feature_scores = {}
    
    # Try each feature individually first
    for feature in all_features:
        X_single = regression_data[[feature]].to_numpy(dtype=np.float32)
        scaler = StandardScaler()
        Xs_single = scaler.fit_transform(X_single).astype(np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
            Xs_single, y, W, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Build design matrix
        X_train_sparse = sparse.csr_matrix(X_train)
        X_test_sparse = sparse.csr_matrix(X_test)
        X_design_train = sparse.hstack([X_train_sparse, W_train], format='csr')
        X_design_test = sparse.hstack([X_test_sparse, W_test], format='csr')
        
        # Fit model
        clf = LogisticRegressionCV(
            Cs=np.array([0.01, 0.03, 0.1, 0.3, 1.0]), cv=3, solver="saga", penalty="l2",
            scoring="neg_log_loss", max_iter=3000, n_jobs=-1, fit_intercept=True,
            class_weight="balanced", refit=True
        )
        clf.fit(X_design_train, y_train)
        
        # Evaluate
        p_hat = clf.predict_proba(X_design_test)[:, 1]
        auc = roc_auc_score(y_test, p_hat)
        feature_scores[feature] = auc
        
        logger.info(f"Single feature '{feature}': AUC = {auc:.3f}")
    
    # Start with best single feature
    best_single_feature = max(feature_scores, key=feature_scores.get)
    best_features = [best_single_feature]
    best_auc = feature_scores[best_single_feature]
    
    logger.info(f"Best single feature: '{best_single_feature}' with AUC = {best_auc:.3f}")
    
    # Forward selection
    remaining_features = [f for f in all_features if f != best_single_feature]
    
    while remaining_features:
        best_addition = None
        best_addition_auc = best_auc
        
        for feature in remaining_features:
            candidate_features = best_features + [feature]
            
            # Test this combination
            X_candidate = regression_data[candidate_features].to_numpy(dtype=np.float32)
            scaler = StandardScaler()
            Xs_candidate = scaler.fit_transform(X_candidate).astype(np.float32)
            
            # Split data
            X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
                Xs_candidate, y, W, test_size=test_size, stratify=y, random_state=random_state
            )
            
            # Build design matrix
            X_train_sparse = sparse.csr_matrix(X_train)
            X_test_sparse = sparse.csr_matrix(X_test)
            X_design_train = sparse.hstack([X_train_sparse, W_train], format='csr')
            X_design_test = sparse.hstack([X_test_sparse, W_test], format='csr')
            
            # Fit model
            clf = LogisticRegressionCV(
                Cs=np.array([0.01, 0.03, 0.1, 0.3, 1.0]), cv=3, solver="saga", penalty="l2",
                scoring="neg_log_loss", max_iter=3000, n_jobs=-1, fit_intercept=True,
                class_weight="balanced", refit=True
            )
            clf.fit(X_design_train, y_train)
            
            # Evaluate
            p_hat = clf.predict_proba(X_design_test)[:, 1]
            auc = roc_auc_score(y_test, p_hat)
            
            logger.info(f"Features {candidate_features}: AUC = {auc:.3f}")
            
            if auc > best_addition_auc:
                best_addition = feature
                best_addition_auc = auc
        
        # Add best feature if it improves performance
        if best_addition and best_addition_auc > best_auc + 0.001:  # Small threshold for improvement
            best_features.append(best_addition)
            remaining_features.remove(best_addition)
            best_auc = best_addition_auc
            logger.info(f"Added feature '{best_addition}'. New best AUC = {best_auc:.3f}")
        else:
            logger.info("No feature addition improves performance. Stopping selection.")
            break
    
    logger.info(f"Final selected features: {best_features}")
    logger.info(f"Best AUC achieved: {best_auc:.3f}")
    
    # Fit final model with best features
    X_best = regression_data[best_features].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs_best = scaler.fit_transform(X_best).astype(np.float32)
    
    # Final train-test split
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
        Xs_best, y, W, test_size=test_size, stratify=y, random_state=random_state
    )
    
    X_train_sparse = sparse.csr_matrix(X_train)
    X_test_sparse = sparse.csr_matrix(X_test)
    X_design_train = sparse.hstack([X_train_sparse, W_train], format='csr')
    X_design_test = sparse.hstack([X_test_sparse, W_test], format='csr')
    
    # Final model fit
    clf_final = LogisticRegressionCV(
        Cs=np.array([0.01, 0.03, 0.1, 0.3, 1.0]), cv=5, solver="saga", penalty="l2",
        scoring="neg_log_loss", max_iter=5000, n_jobs=-1, fit_intercept=True,
        class_weight="balanced", refit=True
    )
    clf_final.fit(X_design_train, y_train)
    C_star = float(clf_final.C_[0])
    logger.info(f"Final model - Selected C (prior var) = {C_star:.3f}")
    
    # Final evaluation
    p_hat_final = clf_final.predict_proba(X_design_test)[:, 1]
    auc_final = roc_auc_score(y_test, p_hat_final)
    ap_final = average_precision_score(y_test, p_hat_final)
    brier_final = brier_score_loss(y_test, p_hat_final)
    fpr, tpr, _ = roc_curve(y_test, p_hat_final)
    y_pred_final = (p_hat_final >= 0.5).astype(int)
    cm_final = confusion_matrix(y_test, y_pred_final)
    
    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final, zero_division=0)
    recall = recall_score(y_test, y_pred_final, zero_division=0)
    f1 = f1_score(y_test, y_pred_final, zero_division=0)
    
    logger.info(f"Final Model Performance - AUC = {auc_final:.3f}, AP = {ap_final:.3f}, Brier = {brier_final:.3f}")
    logger.info(f"Accuracy = {accuracy:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")
    
    # Extract coefficients
    coef_final = clf_final.coef_.ravel()
    p_best = len(best_features)
    beta_fixed_final = coef_final[:p_best]
    gamma_req_final = coef_final[p_best:]
    
    # Create model results
    model_results = {
        'confusion_matrix': cm_final.tolist(),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc_final),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': p_hat_final.tolist(),
        'actual_labels': y_test.tolist(),
        'brier_score': float(brier_final),
        'n_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_negative': int((1 - y_test).sum()),
        'coefficients': {col: float(beta_fixed_final[i]) for i, col in enumerate(best_features)},
        'selected_features': best_features,
        'feature_selection_scores': feature_scores
    }
    
    # Generate plots
    plotter.classification_summary(
        model_results=model_results,
        save_name='mm_best_features_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )
    
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='mm_best_features_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )
    
    # Save results
    os.makedirs(DATA_PATHS['results'], exist_ok=True)
    
    # Feature selection results
    selection_df = pd.DataFrame([
        {'feature': feat, 'single_feature_auc': feature_scores[feat], 'selected': feat in best_features}
        for feat in all_features
    ]).sort_values('single_feature_auc', ascending=False)
    selection_df.to_csv(os.path.join(DATA_PATHS['results'], 'mm_best_features_selection.csv'), index=False)
    
    # Fixed effects for best features
    fixed_tbl = pd.DataFrame({
        'feature': best_features,
        'coef_std': beta_fixed_final,
        'OR_per_1SD': np.exp(beta_fixed_final)
    })
    fixed_tbl.to_csv(os.path.join(DATA_PATHS['results'], 'mm_best_features_fixed_effects.csv'), index=False)
    
    # Request effects
    inv_req_index = {j: rid for rid, j in req_index.items()}
    req_df = pd.DataFrame({
        'request_id': [inv_req_index[j] for j in range(len(gamma_req_final))],
        'effect_logit': gamma_req_final,
        'effect_OR': np.exp(gamma_req_final)
    }).sort_values('effect_logit')
    req_df.to_csv(os.path.join(DATA_PATHS['results'], 'mm_best_features_request_effects.csv'), index=False)
    
    # Summary text
    txt = f"""
    Best Features Multilevel Logistic Regression Results
    ==================================================
    
    Feature Selection Process:
    - Started with {len(all_features)} candidate features: {all_features}
    - Used forward selection with AUC as criterion
    - Selected {len(best_features)} features: {best_features}
    
    Single Feature Performance:
    {chr(10).join([f"  {feat}: AUC = {score:.3f}" for feat, score in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)])}
    
    Final Model Performance (test set):
    - Selected Features: {best_features}
    - AUC: {auc_final:.3f}
    - Average Precision: {ap_final:.3f}
    - Brier Score: {brier_final:.3f}
    - Accuracy: {accuracy:.3f}
    - Precision: {precision:.3f}
    - Recall: {recall:.3f}
    - F1-Score: {f1:.3f}
    
    Model Configuration:
    - Design = [X_best | W] (sparse CSR)
    - Penalty: L2 (Gaussian prior), solver='saga'
    - Selected C (prior variance): {C_star:.3f}
    
    Fixed Effects (standardized scale):
    {chr(10).join([f"  {feat}: {coef:.3f} (OR={np.exp(coef):.3f})" for feat, coef in zip(best_features, beta_fixed_final)])}
    
    Data Information:
    - Total trials: {len(y)}
    - Train trials: {len(y_train)}
    - Test trials: {len(y_test)}
    - Test requested: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)
    - Requests: {info['n_requests']}
    - Trials with membership: {info['n_with_membership']}
    """
    
    save_results(txt, os.path.join(DATA_PATHS['results'], 'mm_best_features_summary.txt'))
    
    logger.info("Best features multilevel logistic regression completed")
    
    return clf_final, model_results, {
        'scaler': scaler, 
        'req_index': req_index, 
        'info': info, 
        'best_features': best_features,
        'feature_scores': feature_scores
    }   

def mm_log_reg_only_enrollment_tree_number(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None, random_state=42, test_size=0.3):
    """
    Multilevel-Membership logistic regression using only enrollment and tree_number as features.
    - Row = trial
    - y = requested (0/1)
    - X = features (log_enrollment, tree_number)
    - W = weight matrix (trials x requests)
    - Design = [X | W], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression with enrollment and tree_number only")

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
    regression_data = merged_data.dropna(subset=['enrollment', 'tree_number', 'requested'])
    regression_data = regression_data[regression_data['enrollment'] > 0]
    regression_data['log_enrollment'] = np.log1p(regression_data['enrollment'])

    logger.info(f"Analysis dataset: {len(regression_data)} trials, {regression_data['requested'].sum()} requested")

    features = ['log_enrollment', 'tree_number']
    X = regression_data[features].to_numpy(dtype=np.float32)
    y = regression_data['requested'].to_numpy(dtype=np.int32)

    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Build weight matrix
    W, req_index, info = build_weight_matrix(regression_data, request_links, logger)
    logger.info(f"Weight matrix info: {info}")

    # train test split
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(
        Xs, y, W, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train = sparse.csr_matrix(X_train); X_test = sparse.csr_matrix(X_test)
    X_design_train = sparse.hstack([X_train, W_train], format='csr')
    X_design_test  = sparse.hstack([X_test,  W_test],  format='csr')

    # CV or C - log loss for probabilistic calibration
    Cs = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=Cs, cv=cv, solver="saga", penalty="l2", scoring="neg_log_loss",
        max_iter=5000, n_jobs=-1, fit_intercept=True, class_weight="balanced", refit=True
    )
    clf.fit(X_design_train, y_train)
    C_star = float(clf.C_[0])
    logger.info(f"Selected C (prior var) = {C_star}")

    p_hat = clf.predict_proba(X_design_test)[:, 1]
    auc = roc_auc_score(y_test, p_hat)  # Use y_test, not y
    ap = average_precision_score(y_test, p_hat)
    brier = brier_score_loss(y_test, p_hat)
    fpr, tpr, _ = roc_curve(y_test, p_hat)
    y_pred = (p_hat >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)  # Use y_test, not y
    cls_rep = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(f"Multilevel Model AUC = {auc:.3f}, AP = {ap:.3f}, Brier = {brier:.3f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{cls_rep}")
    # Extract coefficients
    coef = clf.coef_.ravel()  # length p+R
    p = len(features)
    beta_fixed = coef[:p]
    gamma_req = coef[p:]      # request effects (shrunk)
    model_results = {
        'confusion_matrix': cm.tolist(),
        'accuracy': float(accuracy),           
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': p_hat.tolist(),  # Test set predictions
        'actual_labels': y_test.tolist(),           # FIXED: Use y_test, not y
        'brier_score': float(brier),
        'n_samples': len(y_test),                   # FIXED: Use len(y_test)
        'n_positive': int(y_test.sum()),            # FIXED: Use y_test
        'n_negative': int((1 - y_test).sum()),      # FIXED: Use y_test
        'coefficients': {col: float(beta_fixed[i]) for i, col in enumerate(features)}
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='mm_enrollment_tree_number_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )
    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='mm_enrollment_tree_number_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )
    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    # 1) fixed effects table (standardized scale)
    fixed_tbl = pd.DataFrame({
        'feature': features,
        'coef_std': beta_fixed,
        'OR_per_1SD': np.exp(beta_fixed)
    })
    fixed_tbl.to_csv(os.path.join(DATA_PATHS['results'], 'mm_enrollment_tree_number_fixed_effects.csv'), index=False)

    # 2) request effects (top/bottom 30 for readability)
    inv_req_index = {j: rid for rid, j in req_index.items()}
    req_df = pd.DataFrame({
        'request_id': [inv_req_index[j] for j in range(len(gamma_req))],
        'effect_logit': gamma_req,
        'effect_OR': np.exp(gamma_req)
    }).sort_values('effect_logit')
    req_df.to_csv(os.path.join(DATA_PATHS['results'], 'mm_enrollment_tree_number_request_effects_full.csv'), index=False)
    head = req_df.tail(30); tail = req_df.head(30)
    head.to_csv(os.path.join(DATA_PATHS['results'], 'mm_enrollment_tree_number_request_effects_top30.csv'), index=False)
    tail.to_csv(os.path.join(DATA_PATHS['results'], 'mm_enrollment_tree_number_request_effects_bottom30.csv'), index=False)
    # 3) quick caterpillar plot
    try:
        plt.figure(figsize=(8, 5))
        ordered = req_df.reset_index(drop=True)
        idx = np.arange(len(ordered))
        plt.plot(idx, ordered['effect_logit'].values, '.', ms=3)
        plt.axhline(0, ls='--', c='k')
        plt.xlabel('Requests (sorted)'); plt.ylabel('Effect on log-odds')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'mm_enrollment_tree_number_request_caterpillar.png'), dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Caterpillar plot failed: {e}")
    # 4) text summary
    txt = f"""
        MM Logistic Regression (enrollment + tree_number)

        Data:
        - Total trials: {len(y)}
        - Train trials: {len(y_train)} 
        - Test trials: {len(y_test)}
        - Test requested: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)
        - Requests (columns in W): {info['n_requests']}
        - Trials with membership: {info['n_with_membership']}
        - Matrix density: {info['density']:.4f}

        Model:
        - Design = [X | W] (sparse CSR)
        - Penalty: L2 (Gaussian prior), solver='saga'
        - Selected C (prior variance ~ tau^2): {C_star:.3f}

        Performance (test set):
        - AUC: {auc:.3f}
        - Average Precision: {ap:.3f}
        - Brier Score: {brier:.3f}
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}

        Fixed effects (std. scale):
        {chr(10).join([f"  {feat}: {coef:.3f} (OR={np.exp(coef):.3f})" for feat, coef in zip(features, beta_fixed)])}

        Matrix Information:
        - Total trials: {info['n_trials']}
        - Total requests: {info['n_requests']}  
        - Trials with request membership: {info['n_with_membership']}
        - Matrix sparsity: {(1-info['density'])*100:.1f}% zeros

        Files saved:
        - Fixed effects: mm_enrollment_tree_number_fixed_effects.csv
        - Request effects: mm_enrollment_tree_number_request_effects_full.csv (with top/bottom 30 CSVs)
        """
    save_results(txt, os.path.join(DATA_PATHS['results'], 'mm_enrollment_tree_number_summary.txt'))
    logger.info("MM logistic regression with enrollment and tree_number finished")
    return clf, model_results, {'scaler': scaler, 'req_index': req_index, 'info': info}

def tree_head_weight_matrix(regression_data: pd.DataFrame, logger=None, db: SQL_DatabaseManager=None) -> Tuple[sparse.csr_matrix, Dict[Any, int], Dict[str, Any]]:
    """
    Function for weighted matrix based on tree heads. showing realtionship between the trials and tree heads.
    (N x T) matrix where N is number of trials and T is number of tree heads.
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting tree head weight matrix analysis")

    if db is None:
        db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='my_database')
        logger.info("Connected to My Database")

    trial_ids = regression_data['nct_id'].tolist()
    trial_index = {tid: i for i, tid in enumerate(trial_ids)}
    N = len(trial_ids)

    # Fetch condition and intervention data
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

    # get mesh terms with tree
    mesh_terms = db.get_table_data(
        columns=['downcase_mesh_term', 'tree_number'],
        table_name='mesh_terms'
    )
    logger.info("Fetched mesh terms data")

    # Merge mesh terms with conditions and interventions
    df_trials_downcase_mesh_term = pd.concat([df_conditions, df_interventions], ignore_index=True)
    df_trials_mesh = pd.merge(df_trials_downcase_mesh_term, mesh_terms, on='downcase_mesh_term', how='inner')
    logger.info("Merged trials with mesh terms")

    df_trials_mesh = df_trials_mesh[df_trials_mesh['nct_id'].isin(trial_ids)]
    logger.info(f"Filtered mesh data to {len(df_trials_mesh)} records for {N} trials")

    # Extract tree heads
    df_trials_mesh['tree_head'] = df_trials_mesh['tree_number'].apply(lambda x: extract_tree_head(x, level=2))
    df_trials_mesh = df_trials_mesh.dropna(subset=['tree_head'])

    # Get unique tree heads and create index
    unique_tree_heads = df_trials_mesh['tree_head'].unique()
    unique_tree_heads = [th for th in unique_tree_heads if th is not None]
    tree_head_index = {th: j for j, th in enumerate(unique_tree_heads)}
    T = len(unique_tree_heads)

    logger.info(f"Matrix dimensions: {N} trials x {T} tree heads")

    # Build weight matrix
    rows, cols, data = [], [], []
    trial_with_tree_heads = df_trials_mesh.groupby('nct_id')['tree_head'].apply(list).to_dict()
    
    with_membership = 0
    for trial_id in trial_ids:  # Use trial_ids from regression_data to maintain order
        tree_heads = trial_with_tree_heads.get(trial_id, [])
        tree_heads = [th for th in tree_heads if th is not None and th in tree_head_index]
        
        if tree_heads:
            with_membership += 1
            weight = 1.0 / len(tree_heads)
            i = trial_index[trial_id]
            for tree_head in tree_heads:
                j = tree_head_index[tree_head]
                rows.append(i)
                cols.append(j)
                data.append(weight)

    # Create sparse matrix with exact dimensions matching regression_data
    W_tree = sparse.csr_matrix((data, (rows, cols)), shape=(N, T), dtype=np.float32)
    logger.info(f"Built aligned tree head weight matrix: {W_tree.shape} with {len(rows)} non-zero entries")

    info = {
        'n_trials': N,
        'n_tree_heads': T,
        'density': len(rows) / (N * T) if N * T > 0 else 0,
        'n_with_membership': with_membership
    }

    return W_tree, tree_head_index, info

def multilevel_logistic_regression_with_tree_id(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None):
    """
    Multilevel-Membership hierarchical logistic Regression via fast MAP (ridge) approximation.
    - Row = trial
    - y = requested (0/1)
    - X = features (completeness, qualifiers, tree_number, enrollment)
    - W = weight matrix (trials x requests)
    - T = weight matrix (trials x tree heads)
    - Design = [X | T], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression with tree_id analysis")

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

    features = ['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']
    X = regression_data[features].to_numpy(dtype=np.float32)
    y = regression_data['requested'].to_numpy(dtype=np.int32)
    
    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Build weight matrices
    W_tree, tree_index, info_tree = tree_head_weight_matrix(regression_data, logger, db)
    logger.info(f"Tree head weight matrix info: {info_tree}")

    # Verify matrix dimensions match
    logger.info(f"Matrix shapes - X: {Xs.shape}, y: {y.shape}, W_tree: {W_tree.shape}")
    if W_tree.shape[0] != len(y):
        logger.error(f"Matrix dimension mismatch: W_tree({W_tree.shape}), y({len(y)})")
        return
    # train test split
    X_train, X_test, y_train, y_test, W_tree_train, W_tree_test = train_test_split(
        Xs, y, W_tree, test_size=0.3, stratify=y, random_state=42
    )

    # Convert to sparse matrices
    X_train = sparse.csr_matrix(X_train); X_test = sparse.csr_matrix(X_test)
    X_design_train = sparse.hstack([X_train, W_tree_train], format='csr')
    X_design_test  = sparse.hstack([X_test,  W_tree_test],  format='csr')

    # CV or C - log loss for probabilistic calibration
    clf = LogisticRegressionCV(
        Cs=np.array([0.01, 0.03, 0.1, 0.3, 1.0]), cv=5, solver="saga", penalty="l2",
        scoring="neg_log_loss", max_iter=5000, n_jobs=-1, fit_intercept=True,
        class_weight="balanced", refit=True
    )
    clf.fit(X_design_train, y_train)
    C_star = float(clf.C_[0])
    logger.info(f"Selected C (prior var) = {C_star}")
    
    # Model evaluation
    p_hat = clf.predict_proba(X_design_test)[:, 1]
    auc = roc_auc_score(y_test, p_hat)
    ap = average_precision_score(y_test, p_hat)
    brier = brier_score_loss(y_test, p_hat)
    fpr, tpr, _ = roc_curve(y_test, p_hat)
    y_pred = (p_hat >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cls_rep = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(f"Multilevel Model AUC = {auc:.3f}, AP = {ap:.3f}, Brier = {brier:.3f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{cls_rep}")
    logger.info(f"Accuracy = {accuracy:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")

    # Extract coefficients
    coef = clf.coef_.ravel()  # length p+T
    p = len(features)
    beta_fixed = coef[:p]
    gamma_tree = coef[p:]     # tree head effects (shrunk)
    model_results = {
        'confusion_matrix': cm.tolist(),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'beta_fixed': beta_fixed.tolist(),
        'gamma_tree': gamma_tree.tolist(),
        'auc': float(auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': p_hat.tolist(),
        'actual_labels': y_test.tolist(),
        'brier_score': float(brier),
        'n_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_negative': int((1 - y_test).sum()),
        'coefficients': {col: float(beta_fixed[i]) for i, col in enumerate(features)}
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='mm_map_tree_id_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='mm_map_tree_id_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    os.makedirs(DATA_PATHS['results'], exist_ok=True)
    # 1) fixed effects table (standardized scale)
    fixed_tbl = pd.DataFrame({
        'feature': features,
        'coef_std': beta_fixed,
        'OR_per_1SD': np.exp(beta_fixed)
    })
    fixed_tbl.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_tree_id_fixed_effects.csv'), index=False)
    # 2) tree head effects (top/bottom 30 for readability)
    inv_tree_index = {j: th for th, j in tree_index.items()}
    tree_df = pd.DataFrame({
        'tree_head': [inv_tree_index[j] for j in range(len(gamma_tree))],
        'effect_logit': gamma_tree,
    })
    tree_df.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_tree_id_tree_head_effects.csv'), index=False)

    head = tree_df.tail(30); tail = tree_df.head(30)
    head.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_tree_id_tree_head_effects_top30.csv'), index=False)
    tail.to_csv(os.path.join(DATA_PATHS['results'], 'mm_map_tree_id_tree_head_effects_bottom30.csv'), index=False)

    # 3) quick caterpillar plot
    try:
        plt.figure(figsize=(8, 5))
        ordered = tree_df.sort_values('effect_logit').reset_index(drop=True)
        idx = np.arange(len(ordered))
        plt.plot(idx, ordered['effect_logit'].values, '.', ms=3)
        plt.axhline(0, ls='--', c='k')
        plt.xlabel('Tree heads (sorted)'); plt.ylabel('Effect on log-odds')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'mm_map_tree_id_caterpillar.png'), dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Caterpillar plot failed: {e}")
    # 4) text summary
    txt = f"""
        MAP (Ridge) Multiple-Membership Logistic (Trial-level with Tree ID)
        =============================================
        Data:
        - Total trials: {len(y)}
        - Train trials: {len(y_train)}
        - Test trials: {len(y_test)}
        - Test requested: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)
        - Tree heads (columns in W): {info_tree['n_tree_heads']}
        - Trials with tree head membership: {info_tree['n_with_membership']}
        - Matrix density: {info_tree['density']:.4f}

        Model:
        - Design = [X_std | W_tree] (sparse CSR)
        - Penalty: L2 (Gaussian prior), solver='saga'
        - Selected C (prior variance ~ tau^2): {C_star:.3f}

        Performance (test set):
        - AUC: {auc:.3f}
        - Average Precision: {ap:.3f}
        - Brier Score: {brier:.3f}
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}

        Fixed effects (std. scale):
        {chr(10).join([f"  {feat}: {coef:.3f} (OR={np.exp(coef):.3f})" for feat, coef in zip(features, beta_fixed)])}

        Files saved:
        - Fixed effects: mm_map_tree_id_fixed_effects.csv
        - Tree head effects: mm_map_tree_id_tree_head_effects.csv (with top/bottom 30 CSVs)
        """
    save_results(txt, os.path.join(DATA_PATHS['results'], 'mm_map_tree_id_summary.txt'))
    logger.info("MAP (ridge) multilevel logistic regression with tree ID finished")

    train_sizes, train_scores, test_scores = learning_curve(
        clf, X_design_train, y_train,
        cv=5, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label="train AUC")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="cv AUC")
    plt.legend()
    plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'mm_map_tree_id_learning_curve.png'), dpi=150)
    plt.close()

def multilevel_logistic_regression_with_tree_id_weight_and_request_weights(logger=None, db: SQL_DatabaseManager=None, plotter: MasterPlotter=None):
    """
    Multilevel-Membership hierarchical logistic Regression via fast MAP (ridge) approximation.
    - Row = trial
    - y = requested (0/1)
    - X = features (completeness, qualifiers, tree_number, enrollment)
    - W = weight matrix (trials x requests)
    - T = weight matrix (trials x tree heads)
    - Design = [X | W | T], fit with L2 (Gaussian) prior on coefficients, solver = ''
    """
    if logger is None:
        setup_logging(mode='basic')
        logger = logging.getLogger(__name__)

    logger.info("Starting multilevel logistic regression with tree_id analysis")

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

    features = ['completeness_percentage', 'document_count', 'qualifier', 'tree_number', 'log_enrollment']
    X = regression_data[features].to_numpy(dtype=np.float32)
    y = regression_data['requested'].to_numpy(dtype=np.int32)
    
    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Build weight matrices - FIXED: Use the same regression_data for both matrices
    W, req_index, info_req = build_weight_matrix(regression_data, request_links, logger)
    W_tree, tree_index, info_tree = tree_head_weight_matrix(regression_data, logger, db)
    
    logger.info(f"Weight matrix info (requests): {info_req}")
    logger.info(f"Weight matrix info (tree heads): {info_tree}")
    
    # FIXED: Verify matrix dimensions match
    logger.info(f"Matrix shapes - X: {Xs.shape}, y: {y.shape}, W: {W.shape}, W_tree: {W_tree.shape}")
    
    if W.shape[0] != W_tree.shape[0] or W.shape[0] != len(y):
        logger.error(f"Matrix dimension mismatch: W({W.shape}), W_tree({W_tree.shape}), y({len(y)})")
        return

    # train test split
    X_train, X_test, y_train, y_test, W_train, W_test, W_tree_train, W_tree_test = train_test_split(
        Xs, y, W, W_tree, test_size=0.3, stratify=y, random_state=42
    )

    # Convert to sparse matrices
    X_train = sparse.csr_matrix(X_train); X_test = sparse.csr_matrix(X_test)
    X_design_train = sparse.hstack([X_train, W_train, W_tree_train], format='csr')
    X_design_test  = sparse.hstack([X_test,  W_test,  W_tree_test],  format='csr')

    # CV or C - log loss for probabilistic calibration
    clf = LogisticRegressionCV(
        Cs=np.array([0.01, 0.03, 0.1, 0.3, 1.0]), cv=5, solver="saga", penalty="l2",
        scoring="neg_log_loss", max_iter=5000, n_jobs=-1, fit_intercept=True,
        class_weight="balanced", refit=True
    )
    clf.fit(X_design_train, y_train)
    C_star = float(clf.C_[0])
    logger.info(f"Selected C (prior var) = {C_star}")

    # Model evaluation
    p_hat = clf.predict_proba(X_design_test)[:, 1]
    auc = roc_auc_score(y_test, p_hat)
    ap = average_precision_score(y_test, p_hat)
    brier = brier_score_loss(y_test, p_hat)
    fpr, tpr, _ = roc_curve(y_test, p_hat)
    y_pred = (p_hat >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cls_rep = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(f"Multilevel Model AUC = {auc:.3f}, AP = {ap:.3f}, Brier = {brier:.3f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Extract coefficients
    coef = clf.coef_.ravel()  # length p+R+T
    p = len(features)
    beta_fixed = coef[:p]
    gamma_req = coef[p:p+W.shape[1]]      # request effects (shrunk)
    delta_tree = coef[p+W.shape[1]:]       # tree head effects (shrunk)

    model_results = {
        'confusion_matrix': cm.tolist(),
        'accuracy': float(accuracy),           
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'predicted_probabilities': p_hat.tolist(),  # Test set predictions
        'actual_labels': y_test.tolist(),           # FIXED: Use y_test, not y
        'brier_score': float(brier),
        'n_samples': len(y_test),                   # FIXED: Use len(y_test)
        'n_positive': int(y_test.sum()),            # FIXED: Use y_test
        'n_negative': int((1 - y_test).sum()),      # FIXED: Use y_test
        'coefficients': {col: float(beta_fixed[i]) for i, col in enumerate(features)}
    }

    plotter.classification_summary(
        model_results=model_results,
        save_name='combined_model_classification_summary.png',
        path=DATA_PATHS['img']['regression']
    )

    plotter.probability_calibration_plot(
        model_results=model_results,
        save_name='combined_model_probability_calibration.png',
        path=DATA_PATHS['img']['regression']
    )

    os.makedirs(DATA_PATHS['results'], exist_ok=True)

    # 1) fixed effects table (standardized scale)
    fixed_tbl = pd.DataFrame({
        'feature': features,
        'coef_std': beta_fixed,
        'OR_per_1SD': np.exp(beta_fixed)
    })

    fixed_tbl.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_fixed_effects.csv'), index=False)

    # 2) request effects (top/bottom 30 for readability)
    inv_req_index = {j: rid for rid, j in req_index.items()}
    req_df = pd.DataFrame({
        'request_id': [inv_req_index[j] for j in range(len(gamma_req))],
        'effect_logit': gamma_req,
        'effect_OR': np.exp(gamma_req)
    }).sort_values('effect_logit')

    req_df.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_request_effects_full.csv'), index=False)
    head = req_df.tail(30); tail = req_df.head(30)
    head.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_request_effects_head.csv'), index=False)
    tail.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_request_effects_tail.csv'), index=False)

    # 3) tree head effects (top/bottom 30 for readability)
    inv_tree_index = {j: th for th, j in tree_index.items()}
    tree_df = pd.DataFrame({
        'tree_head': [inv_tree_index[j] for j in range(len(delta_tree))],
        'effect_logit': delta_tree,
        'effect_OR': np.exp(delta_tree)
    }).sort_values('effect_logit')
    tree_df.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_effects_full.csv'), index=False)
    head = tree_df.tail(30); tail = tree_df.head(30)
    head.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_effects_head.csv'), index=False)
    tail.to_csv(os.path.join(DATA_PATHS['results'], 'combined_model_effects_tail.csv'), index=False)

    # 4) quick caterpillar plot for requests
    try:
        plt.figure(figsize=(8, 5))
        ordered = req_df.reset_index(drop=True)
        idx = np.arange(len(ordered))
        plt.plot(idx, ordered['effect_logit'].values, '.', ms=3)
        plt.axhline(0, ls='--', c='k')
        plt.xlabel('Requests (sorted)'); plt.ylabel('Effect on log-odds')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'combined_model_request_caterpillar.png'), dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Caterpillar plot for requests failed: {e}")

    # 5) quick caterpillar plot for tree heads
    try:
        plt.figure(figsize=(8, 5))
        ordered = tree_df.reset_index(drop=True)
        idx = np.arange(len(ordered))
        plt.plot(idx, ordered['effect_logit'].values, '.', ms=3)
        plt.axhline(0, ls='--', c='k')
        plt.xlabel('Tree Heads (sorted)'); plt.ylabel('Effect on log-odds')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'combined_model_tree_head_caterpillar.png'), dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Caterpillar plot for tree heads failed: {e}")

    # 6) text summary
    txt = f"""
        MAP (Ridge) Multiple-Membership Logistic (Trial-level) with Tree ID

        Data:
        - Total trials: {len(y)}
        - Train trials: {len(y_train)} 
        - Test trials: {len(y_test)}
        - Test requested: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)
        - Requests (columns in W): {info_req['n_requests']}
        - Trials with request membership: {info_req['n_with_membership']}
        - Tree Heads (columns in W_tree): {info_tree['n_tree_heads']}
        - Trials with tree head membership: {info_tree['n_with_membership']}
        - Request Matrix density: {info_req['density']:.4f}
        - Tree Head Matrix density: {info_tree['density']:.4f}

        Model:
        - Design = [X_std | W | W_tree] (sparse CSR)
        - Penalty: L2 (Gaussian prior), solver='saga'
        - Selected C (prior variance ~ tau^2): {C_star:.3f}

        Performance (test set):
        - AUC: {auc:.3f}
        - Average Precision: {ap:.3f}
        - Brier Score: {brier:.3f}
        - Accuracy: {accuracy:.3f}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1:.3f}

        Fixed effects (std. scale):
        {chr(10).join([f"  {feat}: {coef:.3f} (OR={np.exp(coef):.3f})" for feat, coef in zip(features, beta_fixed)])}

        Files saved:
        - Fixed effects: mm_map_tree_id_fixed_effects.csv
        - Request effects: mm_map_tree_id_request_effects_full.csv (with top/bottom 30 CSVs)
        - Tree head effects: mm_map_tree_id_effects_full.csv (with top/bottom 30 CSVs)
        """
    save_results(txt, os.path.join(DATA_PATHS['results'], 'combined_model_summary.txt'))

    logger.info("MAP (ridge) multilevel logistic regression with tree_id finished")

    train_sizes, train_scores, test_scores = learning_curve(
        clf, X_design_train, y_train,
        cv=5, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label="train AUC")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="cv AUC")
    plt.savefig(os.path.join(DATA_PATHS['img']['regression'], 'combined_model_learning_curve.png'), dpi=150)
    plt.close()

    return clf, model_results, {'scaler': scaler, 'req_index': req_index, 'tree_index': tree_index, 'info_req': info_req, 'info_tree': info_tree}
    
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
    multilevel_logistic_regression_best_features(logger, db, plotting)  # Add this line
    mm_log_reg_only_enrollment_tree_number(logger, db, plotting)
    multilevel_logistic_regression_with_tree_id(logger, db, plotting)
    multilevel_logistic_regression_with_tree_id_weight_and_request_weights(logger, db, plotting)
    
    logger.info("Script finished")

if __name__== "__main__":
    runner()
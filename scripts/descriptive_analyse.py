import numpy as np
import pandas as pd

import os
import re
import sys
import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import AACT_DB_CONFIG, MY_DB_CONFIG, PROJECT_SETTINGS, DATA_PATHS, LOGGING_CONFIG, PLATTFORM_IDS
from db.sql_descriptive import SQL_DatabaseManager
from db.sqlalchemy_connector import SQLAlchemyConnector, DatabaseManager
from utils.plotting import MasterPlotter

import matplotlib
matplotlib.use('Agg')  # use non-interactive backend to avoid Tkinter issues in threads / headless env
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(mode='basic'):
    """Setup logging configuration"""
    import logging.config
    
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

def investigator_analysis(logger=None): # done
    """Perform investigator analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    # TRIALs
    # Plotting Investigators with highest number of trials presented in all data platforms
    df = db_manager.get_table_data(
        columns=['nct_id', 'name as investigator'],
        table_name='investigator'
    )
    if df is None or df.empty:
        logger.error("No investigator data found.")
        return
    
    investigator_plotting_data = df.groupby('investigator').size().reset_index(name='count').sort_values(by='count', ascending=False)

    plotting.bar_plot(
        data=investigator_plotting_data.head(20),
        x='investigator',
        y='count',
        xlabel='Investigator',
        ylabel='Number of Trials',
        save_name= 'top20_investigators_NCT.png',
        path = DATA_PATHS['img']['investigator']
    )

    # REQUEST
    # Plotting Investigators with highest number of requests
    df = db_manager.get_table_data(
        columns=['investigator', 'request_id'], 
        table_name='requests'
        )

    if df is None or df.empty:
        logger.error("No request data found.")
        return
    
    investigator_request_data = df.groupby('investigator').size().reset_index(name='request_count').sort_values(by='request_count', ascending=False)

    plotting.bar_plot(
        data=investigator_request_data.head(20),
        x='investigator',
        y='request_count',
        xlabel='Investigator',
        ylabel='Number of Requests',
        save_name= 'top20_investigators_requests.png',
        path = DATA_PATHS['img']['investigator']
    )

    logger.info("Investigator analysis completed and plot saved.")

def institution_analysis(logger=None): # done
    """Perform institution analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()
    # TRIALs
    # Plotting Institutions with highest number of trials presented in all data platforms
    df = db_manager.get_table_data(
        columns=['nct_id', 'name as institution'],
        table_name='institution'
    )
    if df is None or df.empty:
        logger.error("No institution data found.")
        return
    institution_plotting_data = df.groupby('institution').size().reset_index(name='count').sort_values(by='count', ascending=False)
    plotting.bar_plot(
        data=institution_plotting_data.head(20),
        x='institution',
        y='count',
        xlabel='Institution',
        ylabel='Number of Trials',
        save_name= 'top20_institutions_NCT.png',
        path = DATA_PATHS['img']['institution']
    )

    # REQUEST
    # Plotting Institutions with highest number of requests
    df = db_manager.get_table_data(
        columns=['institution', 'request_id'], 
        table_name='requests'
        )
    if df is None or df.empty:
        logger.error("No request data found.")
        return
    institution_request_data = df.groupby('institution').size().reset_index(name='request_count').sort_values(by='request_count', ascending=False)
    plotting.bar_plot(
        data=institution_request_data.head(20),
        x='institution',
        y='request_count',
        xlabel='Institution',
        ylabel='Number of Requests',
        save_name= 'top20_institutions_requests.png',
        path = DATA_PATHS['img']['institution']
    )
    logger.info("Institution analysis completed and plot saved.")

def enrollment_analysis(logger=None): # done
    """Perform enrollment analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()
    
    df = db_manager.get_table_data(
        columns=['nct_id', 'enrollment'],
        table_name='clinical_trials'
    )
    if df is None or df.empty:
        logger.error("No enrollment data found.")
        return
    
    # Log raw data information
    logger.info(f"Raw enrollment data - Total records: {len(df)}")
    logger.debug(f"Enrollment data type: {df['enrollment'].dtype}")
    logger.debug(f"Unique enrollment values: {df['enrollment'].nunique()}")
    
    # Convert to numeric and handle errors
    df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce')
    df = df.dropna(subset=['enrollment'])
    df = df[df['enrollment'] > 0]  # Remove zero and negative values
    
    # Log data statistics
    logger.info(f"After cleaning - Valid records: {len(df)}")
    logger.info(f"Enrollment range: {df['enrollment'].min():.0f} - {df['enrollment'].max():.0f}")
    logger.info(f"Mean enrollment: {df['enrollment'].mean():.2f}")
    logger.info(f"Median enrollment: {df['enrollment'].median():.2f}")
    logger.debug(f"95th percentile: {df['enrollment'].quantile(0.95):.2f}")
    logger.debug(f"99th percentile: {df['enrollment'].quantile(0.99):.2f}")
    
    # Remove extreme outliers that skew the visualization
    upper_limit = df['enrollment'].quantile(0.95)
    df_filtered = df[df['enrollment'] <= upper_limit].copy()
    
    logger.info(f"Outlier removal - Upper limit: {upper_limit:.0f}")
    logger.info(f"Remaining trials: {len(df_filtered)}, Removed: {len(df) - len(df_filtered)}")
    logger.info(f"New enrollment range: {df_filtered['enrollment'].min():.0f} - {df_filtered['enrollment'].max():.0f}")

    # Create histogram with proper bins - REMOVED 'title' parameter
    plotting.histogram(
        data=df,
        column='enrollment',
        xlabel='Enrollment (Number of Participants)',
        ylabel='Number of Trials',
        save_name='enrollment_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=50
    )

    # outlier removed histogram
    plotting.histogram(
        data=df_filtered,
        column='enrollment',
        xlabel='Enrollment (Number of Participants)',
        ylabel='Number of Trials',
        save_name='enrollment_distribution_without_outliers.png',
        path=DATA_PATHS['img']['trials'],
        bins=50
    )

    # save info to text file
    stats_path = os.path.join(DATA_PATHS['descriptive'], 'enrollment_statistics.txt')

    with open(stats_path, 'w') as f:
        f.write("Enrollment Statistics\n")
        f.write("=====================\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Valid records: {len(df_filtered)}\n")
        f.write(f"Enrollment range: {df_filtered['enrollment'].min():.0f} - {df_filtered['enrollment'].max():.0f}\n")
        f.write(f"Mean enrollment: {df_filtered['enrollment'].mean():.2f}\n")
        f.write(f"Median enrollment: {df_filtered['enrollment'].median():.2f}\n")
        f.write(f"Std deviation: {df_filtered['enrollment'].std():.2f}\n")
        f.write(f"95th percentile: {df_filtered['enrollment'].quantile(0.95):.2f}\n")
        f.write(f"99th percentile: {df_filtered['enrollment'].quantile(0.99):.2f}\n")

    logger.info("Enrollment analysis completed and plots saved.")

def nct_per_request_analysis(logger=None): # done
    """Analyze number of NCTs per data request and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )
    if df is None or df.empty:
        logger.error("No request-trial link data found.")
        return
    
    nct_per_request = df.groupby('request_id').size().reset_index(name='nct_count')
    plotting.histogram(
        data=nct_per_request,
        column='nct_count',
        xlabel='Number of NCTs',
        ylabel='Number of Requests',
        save_name='nct_per_request_distribution.png',
        path=DATA_PATHS['img']['requests'],
        bins=range(1, nct_per_request['nct_count'].max() + 1)
    )
    logger.info("NCT per request analysis completed and plot saved.")

    # Save statistics to text file
    stats_path = os.path.join(DATA_PATHS['descriptive'], 'nct_per_request_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("NCTs per Request Statistics\n")
        f.write("===========================\n")
        f.write(f"Total requests: {len(nct_per_request)}\n")
        f.write(f"Total unique NCTs: {nct_per_request['nct_count'].sum()}\n")
        f.write(f"Mean NCTs per request: {nct_per_request['nct_count'].mean():.2f}\n")
        f.write(f"Median NCTs per request: {nct_per_request['nct_count'].median():.2f}\n")
        f.write(f"Max NCTs in a request: {nct_per_request['nct_count'].max()}\n")
        f.write(f"Min NCTs in a request: {nct_per_request['nct_count'].min()}\n")
        f.write(f"Std deviation: {nct_per_request['nct_count'].std():.2f}\n")
        f.write(f"95th percentile: {nct_per_request['nct_count'].quantile(0.95):.2f}\n")
        f.write(f"99th percentile: {nct_per_request['nct_count'].quantile(0.99):.2f}\n")
    logger.info("NCT per request statistics saved.")

def phase_analysis(logger=None): # done
    """Perform phase analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['phase', 'nct_id'],
        table_name='clinical_trials'
    )
    if df is None or df.empty:
        logger.error("No phase data found.")
        return
    
    phase_data = df.groupby('phase').size().reset_index(name='nct_count')
    plotting.bar_plot(
        data=phase_data,
        x='phase',
        y='nct_count',
        xlabel='Phase',
        ylabel='Number of Trials',
        save_name='trials_by_phase.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Phase analysis completed and plot saved.")

    # requests per phase
    requests_trials = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'  # Note: should be 'request_trial_links' not 'request_trial_link'
    )

    if requests_trials is None or requests_trials.empty:
        logger.error("No request-trial link data found.")
        return
    
    merged_data = pd.merge(requests_trials, df, on='nct_id', how='inner')
    requests_phase_summary = merged_data.groupby('phase').size().reset_index(name='request_count')
    
    plotting.bar_plot(
        data=requests_phase_summary,
        x='phase',
        y='request_count',
        xlabel='Phase',
        ylabel='Number of Requests',
        save_name='requests_by_phase.png',
        path=DATA_PATHS['img']['requests']
    )
    logger.info("Requests by phase analysis completed and plot saved.")

def study_type_analysis(logger=None): # done
    """Perform study type analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['study_type', 'nct_id'],
        table_name='clinical_trials'
    )
    if df is None or df.empty:
        logger.error("No study type data found.")
        return
    
    study_type_data = df.groupby('study_type').size().reset_index(name='nct_count')
    plotting.bar_plot(
        data=study_type_data,
        x='study_type',
        y='nct_count',
        xlabel='Study Type',
        ylabel='Number of Trials',
        save_name='trials_by_study_type.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Study type analysis completed and plot saved.")

    # requests per study type
    requests_trials = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )
    if requests_trials is None or requests_trials.empty:
        logger.error("No request-trial link data found.")
        return
    

    merged_data = pd.merge(requests_trials, df, on='nct_id', how='inner')
    requests_study_type_summary = merged_data.groupby('study_type').size().reset_index(name='request_count')
    
    plotting.bar_plot(
        data=requests_study_type_summary,
        x='study_type',
        y='request_count',
        xlabel='Study Type',
        ylabel='Number of Requests',
        save_name='requests_by_study_type.png',
        path=DATA_PATHS['img']['requests']
    )

    logger.info("Requests by study type analysis completed and plot saved.")

def year_analysis(logger=None): # done
    """Perform year analysis for trials or publications and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['date_published', 'nct_id'],
        table_name='clinical_trials'
    )

    if df is None or df.empty:
        logger.error("No date published data found.")
        return
    
    df['year'] = pd.to_datetime(df['date_published'], errors='coerce').dt.year
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # per platform
    platform_data = db_manager.get_table_data(
        columns=['platform_name', 'trial_nct_id AS nct_id'],
        table_name='data_access'
    )
    if platform_data is None or platform_data.empty:
        logger.error("No platform data found.")
        return
    

    merged_df = pd.merge(df, platform_data, on='nct_id', how='inner')
    year_platform_summary = merged_df.groupby(['year', 'platform_name']).size().reset_index(name='nct_count')
    plotting.line_plot(
        data=year_platform_summary,
        x='year',
        y='nct_count',
        hue='platform_name',
        xlabel='Year',
        ylabel='Number of Trials',
        save_name=f'trials_by_year_platform.png',
        path=DATA_PATHS['img']['trials']
    )

    logger.info("Year and platform analysis completed and plot saved.")

    requests_data = db_manager.get_table_data(
        columns=['request_id', 'date_of_request', 'platform_id'],
        table_name='requests'
    )
    if requests_data is None or requests_data.empty:
        logger.error("No request data found.")
        return
    
    platforms = db_manager.get_table_data(
        columns=['id', 'name'],
        table_name='platforms'
    )
    
    # Merge with platforms to get names
    requests_with_platforms = pd.merge(requests_data, platforms, left_on='platform_id', right_on='id', how='left')
    requests_with_platforms['platform_name'] = requests_with_platforms['name'].fillna('Unknown')
    
    requests_with_platforms['year'] = pd.to_datetime(requests_with_platforms['date_of_request'], errors='coerce').dt.year
    requests_with_platforms = requests_with_platforms.dropna(subset=['year'])

    requests_year_summary = requests_with_platforms.groupby(['year', 'platform_name']).size().reset_index(name='request_count')
    plotting.line_plot(
        data=requests_year_summary,
        x='year',
        y='request_count',
        hue='platform_name',
        xlabel='Year',
        ylabel='Number of Requests',
        save_name='requests_by_year_platform.png',
        path=DATA_PATHS['img']['requests']
    )

    logger.info("Requests by year analysis completed and plot saved.")

def partial_year(logger=None): # done
    """Perform year analysis for trials or publications and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['date_published', 'nct_id'],
        table_name='clinical_trials'
    )
    
    data_access = db_manager.get_table_data(
        columns=['platform_name', 'trial_nct_id AS nct_id'],
        table_name='data_access'
    )

    if data_access is None or data_access.empty:
        logger.error("No data access data found.")
        return

    if df is None or df.empty:
        logger.error("No date published data found.")
        return
    
    df = pd.merge(df, data_access, on='nct_id', how='inner')
    
    # date published to year
    df['year'] = pd.to_datetime(df['date_published'], errors='coerce').dt.year

    # Fix: Use .copy() to avoid SettingWithCopyWarning
    vivli_df = df[df['platform_name'] == 'vivli'].copy()
    csdr_df = df[df['platform_name'] == 'csdr'].copy()
    yoda_df = df[df['platform_name'] == 'yoda'].copy()

    # Check if dataframes have data
    if vivli_df.empty and csdr_df.empty and yoda_df.empty:
        logger.error("No data found for any platform")
        return

    min_year_vivli = vivli_df['year'].min() if not vivli_df.empty else None
    min_year_csdr = csdr_df['year'].min() if not csdr_df.empty else None
    min_year_yoda = yoda_df['year'].min() if not yoda_df.empty else None

    # Only process platforms that have data
    df_list = []
    if not vivli_df.empty and pd.notna(min_year_vivli):
        vivli_df['partial_year'] = vivli_df['year'] - min_year_vivli
        df_list.append(vivli_df)
        
    if not csdr_df.empty and pd.notna(min_year_csdr):
        csdr_df['partial_year'] = csdr_df['year'] - min_year_csdr
        df_list.append(csdr_df)
        
    if not yoda_df.empty and pd.notna(min_year_yoda):
        yoda_df['partial_year'] = yoda_df['year'] - min_year_yoda
        df_list.append(yoda_df)

    if not df_list:
        logger.error("No valid data for partial year analysis")
        return

    df_summary = pd.concat(df_list, ignore_index=True)
    df_summary = df_summary.groupby(['partial_year', 'platform_name']).size().reset_index(name='nct_count')

    # Add debugging info
    logger.info(f"Partial year summary data shape: {df_summary.shape}")
    logger.info(f"Partial year data preview:\n{df_summary.head()}")

    plotting.line_plot(
        data=df_summary,
        x='partial_year',
        y='nct_count',
        hue='platform_name',
        xlabel='Years since first publication in platform',
        ylabel='Number of Trials',
        save_name='trials_by_partial_year_platform.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Partial year and platform analysis completed and plot saved.")

    # Similar fix for requests part
    requests_data = db_manager.get_table_data(
        columns=['request_id', 'date_of_request'],
        table_name='requests'
    )
    if requests_data is None or requests_data.empty:
        logger.error("No request data found.")
        return
    
    request_trial_link = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )

    if request_trial_link is None or request_trial_link.empty:
        logger.error("No request-trial link data found.")
        return
    
    requests_data = pd.merge(requests_data, request_trial_link, on='request_id', how='inner')
    requests_data = pd.merge(requests_data, data_access, on='nct_id', how='inner')

    requests_data['req_year'] = pd.to_datetime(requests_data['date_of_request'], errors='coerce').dt.year

    # Fix: Use .copy() to avoid warnings
    vivli_req = requests_data[requests_data['platform_name'] == 'vivli'].copy()
    csdr_req = requests_data[requests_data['platform_name'] == 'csdr'].copy()
    yoda_req = requests_data[requests_data['platform_name'] == 'yoda'].copy()

    req_list = []
    if not vivli_req.empty:
        min_vivli = vivli_req['req_year'].min()
        if pd.notna(min_vivli):
            vivli_req['partial_year'] = vivli_req['req_year'] - min_vivli
            req_list.append(vivli_req)
            
    if not csdr_req.empty:
        min_csdr = csdr_req['req_year'].min()
        if pd.notna(min_csdr):
            csdr_req['partial_year'] = csdr_req['req_year'] - min_csdr
            req_list.append(csdr_req)
            
    if not yoda_req.empty:
        min_yoda = yoda_req['req_year'].min()
        if pd.notna(min_yoda):
            yoda_req['partial_year'] = yoda_req['req_year'] - min_yoda
            req_list.append(yoda_req)

    if not req_list:
        logger.error("No valid data for requests partial year analysis")
        return

    req_data = pd.concat(req_list, ignore_index=True)
    req_summary = req_data.groupby(['partial_year', 'platform_name']).size().reset_index(name='request_count')

    # Add debugging info
    logger.info(f"Request partial year summary data shape: {req_summary.shape}")
    logger.info(f"Request partial year data preview:\n{req_summary.head()}")

    plotting.line_plot(
        data=req_summary,
        x='partial_year',
        y='request_count',
        hue='platform_name',
        xlabel='Years since first request in platform',
        ylabel='Number of Requests',
        save_name='requests_by_partial_year_platform.png',
        path=DATA_PATHS['img']['requests']
    )
    logger.info("Requests by partial year analysis completed and plot saved.")

def condition_analysis(logger=None): # done
    """Perform condition analysis and generate plots."""
    # =========================================
    # conditions per nct_id
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['nct_id', 'condition'],
        table_name='conditions'
    )
    if df is None or df.empty:
        logger.error("No condition data found.")
        return
    
    # group by ncts and sort them by number of conditions
    condition_data = df.groupby('nct_id').size().reset_index(name='condition_count').sort_values(by='condition_count', ascending=False)
    plotting.histogram(
        data=condition_data,
        column='condition_count',
        xlabel='Number of Conditions',
        ylabel='Number of Trials',
        save_name='conditions_per_trial_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=range(1, condition_data['condition_count'].max() + 1)
    )
    logger.info("Condition analysis completed and plot saved.")

    # =========================================
    # most common conditions
    common_conditions = df['condition'].value_counts().reset_index()
    common_conditions.columns = ['condition', 'count']
    plotting.bar_plot(
        data=common_conditions.head(20),  # ADDED: Limit to top 20 for readability
        x='condition',
        y='count',
        xlabel='Condition',
        ylabel='Count',
        save_name='most_common_conditions.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Most common conditions analysis completed and plot saved.")

    # =========================================
    # conditions mesh terms

    df_conditions = db_manager.get_table_data(
        columns=['nct_id', 'condition'],
        table_name='conditions'
    )

    df_mesh_terms = db_manager.get_table_data(
        columns=['qualifier', 'tree_number', 'downcase_mesh_term'],
        table_name='mesh_terms'
    )

    df_requests = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )

    if any(data is None or data.empty for data in [df_conditions, df_mesh_terms, df_requests]):
        logger.error("Missing data for mesh terms analysis.")
        return

    # FIXED: Use pandas merge operations instead of manual loops
    # Merge conditions with mesh terms
    conditions_with_mesh = pd.merge(
        df_conditions, 
        df_mesh_terms, 
        left_on='condition', 
        right_on='downcase_mesh_term', 
        how='left'
    )
    
    # Merge with requests
    conditions_with_requests = pd.merge(
        conditions_with_mesh, 
        df_requests, 
        on='nct_id', 
        how='left'
    )
    # ---------- common conditions by request ------------
    conditions_names_with_requests = conditions_with_requests['condition'].value_counts().reset_index()
    conditions_names_with_requests.columns = ['condition', 'request_count']
    plotting.bar_plot(
        data=conditions_names_with_requests.head(20),  # Top 20 for readability
        x='condition',
        y='request_count',
        xlabel='Condition',
        ylabel='Number of Requests',
        save_name='most_common_conditions_by_requests.png',
        path=DATA_PATHS['img']['requests']
    )
    logger.info("Most common conditions by requests analysis completed and plot saved.")


    # Filter out rows without mesh term matches for qualifier analysis
    conditions_with_qualifier = conditions_with_mesh.dropna(subset=['qualifier'])
    
    if not conditions_with_qualifier.empty:
        # ---------- qualifiers ------------
        qualifier_summary = conditions_with_qualifier.groupby('qualifier').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
        plotting.bar_plot(
            data=qualifier_summary.head(20),  # Top 20 qualifiers
            x='qualifier',
            y='nct_count',
            xlabel='Mesh Term Qualifier',
            ylabel='Number of Trials',
            save_name='conditions_trials_by_mesh_qualifier.png',
            path=DATA_PATHS['img']['trials']
        )
        logger.info("Mesh term qualifier analysis completed and plot saved.")

        # Requests by qualifier
        request_conditions_with_qualifier = conditions_with_requests.dropna(subset=['qualifier', 'request_id'])
        if not request_conditions_with_qualifier.empty:
            req_qualifier_summary = request_conditions_with_qualifier.groupby('qualifier').size().reset_index(name='request_count').sort_values(by='request_count', ascending=False)
            plotting.bar_plot(
                data=req_qualifier_summary.head(20),
                x='qualifier',
                y='request_count',
                xlabel='Mesh Term Qualifier',
                ylabel='Number of Requests',
                save_name='requests_by_conditions_mesh_qualifier.png',
                path=DATA_PATHS['img']['requests']
            )
            logger.info("Mesh Terms per requests plotted successfully")
    else:
        logger.warning("No mesh term qualifiers found for conditions.")

    # --------- Tree Numbers ------------
    conditions_with_tree = conditions_with_mesh.dropna(subset=['tree_number'])
    if not conditions_with_tree.empty:
        tree_number_summary = conditions_with_tree.groupby('tree_number').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
        plotting.bar_plot(
            data=tree_number_summary.head(20),
            x='tree_number',
            y='nct_count',
            xlabel='Mesh Term Tree Number',
            ylabel='Number of Trials',
            save_name='conditions_trials_by_mesh_tree_number.png',
            path=DATA_PATHS['img']['trials']
        )
        logger.info("Mesh term tree number analysis completed and plot saved.")
    else:
        logger.warning("No mesh term tree numbers found for conditions.")

def intervention_analysis(logger=None): # done
    """
    Perform intervention analysis and generate plots.
    For intervention types, names, and other relevant statistics.
    """
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()
    
    # =========================================
    # interventions per nct_id
    df = db_manager.get_table_data(
        columns=['nct_id', 'name as intervention'],
        table_name='interventions'
    )
    if df is None or df.empty:
        logger.error("No intervention data found.")
        return
    
    # interventions for grouping all to UPPER case
    df['intervention'] = df['intervention'].str.upper()
    
    intervention_data = df.groupby('nct_id').size().reset_index(name='intervention_count').sort_values(by='intervention_count', ascending=False)
    plotting.histogram(
        data=intervention_data,
        column='intervention_count',
        xlabel='Number of Interventions',
        ylabel='Number of Trials',
        save_name='interventions_per_trial_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=range(1, intervention_data['intervention_count'].max() + 1)
    )
    logger.info("Intervention analysis completed and plot saved.")

    # =========================================
    # most common interventions
    common_interventions = df['intervention'].value_counts().reset_index()
    common_interventions.columns = ['intervention', 'count']
    plotting.bar_plot(
        data=common_interventions.head(20),  # Top 20 for readability
        x='intervention',
        y='count',
        xlabel='Intervention',
        ylabel='Count',
        save_name='most_common_interventions.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Most common interventions analysis completed and plot saved.")

    # =========================================
    # intervention types
    intervention_types = db_manager.get_table_data(
        columns=['intervention_type', 'nct_id'],
        table_name='interventions'
    )
    if intervention_types is None or intervention_types.empty:
        logger.error("No intervention type data found.")
        return
    
    intervention_type_summary = intervention_types.groupby('intervention_type').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=intervention_type_summary,
        x='intervention_type',
        y='nct_count',
        xlabel='Intervention Type',
        ylabel='Number of Trials',
        save_name='trials_by_intervention_type.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Intervention type analysis completed and plot saved.")
    
    # =========================================
    # Mesh Terms
    df_interventions = db_manager.get_table_data(
        columns=['nct_id', 'intervention'],
        table_name='interventions_mesh_terms'
    )
    
    df_mesh_terms = db_manager.get_table_data(
        columns=['qualifier', 'tree_number', 'downcase_mesh_term'],
        table_name='mesh_terms'
    )
    
    df_requests = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )

    if any(data is None or data.empty for data in [df_interventions, df_mesh_terms, df_requests]):
        logger.error("Missing data for intervention mesh terms analysis.")
        return

    interventions_with_mesh = pd.merge(
        df_interventions, 
        df_mesh_terms, 
        left_on='intervention', 
        right_on='downcase_mesh_term', 
        how='left'
    )
    
    interventions_with_requests = pd.merge(
        interventions_with_mesh, 
        df_requests, 
        on='nct_id', 
        how='left'
    )

    # ---------- common interventions by request ------------
    interventions_names_with_requests = interventions_with_requests['intervention'].value_counts().reset_index()
    interventions_names_with_requests.columns = ['intervention', 'request_count']
    plotting.bar_plot(
        data=interventions_names_with_requests.head(20),  # Top 20 for readability
        x='intervention',
        y='request_count',
        xlabel='Intervention',
        ylabel='Number of Requests',
        save_name='most_common_interventions_by_requests.png',
        path=DATA_PATHS['img']['requests']
    )
    logger.info("Most common interventions by requests analysis completed and plot saved.")

    # ---------- qualifiers ------------
    interventions_with_qualifier = interventions_with_mesh.dropna(subset=['qualifier'])
    if not interventions_with_qualifier.empty:
        qualifier_summary = interventions_with_qualifier.groupby('qualifier').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
        plotting.bar_plot(
            data=qualifier_summary.head(20),
            x='qualifier',
            y='nct_count',
            xlabel='Mesh Term Qualifier',
            ylabel='Number of Trials',
            save_name='trials_by_intervention_mesh_qualifier.png',
            path=DATA_PATHS['img']['trials']
        )
        logger.info("Intervention mesh term qualifier analysis completed and plot saved.")

        # Requests by qualifier
        request_interventions_with_qualifier = interventions_with_requests.dropna(subset=['qualifier', 'request_id'])
        if not request_interventions_with_qualifier.empty:
            req_qualifier_summary = request_interventions_with_qualifier.groupby('qualifier').size().reset_index(name='request_count').sort_values(by='request_count', ascending=False)
            plotting.bar_plot(
                data=req_qualifier_summary.head(20),
                x='qualifier',
                y='request_count',
                xlabel='Mesh Term Qualifier',
                ylabel='Number of Requests',
                save_name='requests_by_intervention_mesh_qualifier.png',
                path=DATA_PATHS['img']['requests']
            )
            logger.info("Intervention mesh terms per requests plotted successfully")
    else:
        logger.warning("No mesh term qualifiers found for interventions.")

def mesh_terms_analysis(logger=None): # done
    """Analysing the combinations of mesh terms on trial, request and publication level."""

    db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    # trails
    # interventions per nct_id
    interventions = db.get_table_data(
        columns=['nct_id', 'intervention'],
        table_name='interventions_mesh_terms'
    )
    # conditions per nct_id
    conditions = db.get_table_data(
        columns=['nct_id', 'condition'],
        table_name='conditions'
    )

    # Requests
    requests = db.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )

    # publications
    publications = db.get_table_data(
        columns=['request_id', 'doi'],
        table_name='public_disclosures'
    )

    # mesh terms
    mesh_terms = db.get_table_data(
        columns=['downcase_mesh_term', 'qualifier', 'tree_number'],
        table_name='mesh_terms'
    )

    # get mesh term code for interventions and conditions
    interventions_mesh_terms = pd.merge(
        interventions, 
        mesh_terms, 
        left_on='intervention', 
        right_on='downcase_mesh_term', 
        how='left'
    )
    conditions_mesh_terms = pd.merge(
        conditions, 
        mesh_terms, 
        left_on='condition', 
        right_on='downcase_mesh_term', 
        how='left'
    )

    # trials with mesh terms
    df_trials = pd.concat([interventions_mesh_terms, conditions_mesh_terms], ignore_index=True)
    
    df_requests = pd.merge(df_trials, requests, on='nct_id', how='inner')

    logger.error(f"Publications data shape: {publications.shape}")
    logger.error(f"Requests data shape: {df_requests.shape}")
    logger.error(f"Trials data shape: {df_trials.shape}")

    df_publications = pd.merge(df_requests, publications, on='request_id', how='inner')

    # combinations of mesh terms
    # Trials

    qualifier_combinations = []
    tree_number_combinations = []
    qualifier_pairs = []
    tree_number_pairs = []

    df_trials = df_trials.dropna(subset=['qualifier', 'tree_number'], how='all')
    df_requests = df_requests.dropna(subset=['qualifier', 'tree_number'], how='all')
    df_publications = df_publications.dropna(subset=['qualifier', 'tree_number'], how='all')

    trials = df_trials.copy()
    requests = df_requests.copy()

    for trial_id, trial_group in df_trials.groupby('nct_id'):
        
        # Store full combinations (comma-separated)
        if len(trial_group['qualifier']) > 0:
            qualifier_combinations.append(', '.join(sorted(trial_group['qualifier'].dropna().unique())))
        if len(trial_group['tree_number']) > 0:
            tree_number_combinations.append(', '.join(sorted(trial_group['tree_number'].dropna().unique())))

        # Generate and store pairs
        for i, qual1 in enumerate(trial_group['qualifier']):
            for qual2 in trial_group['qualifier'][i+1:]:  # Avoid duplicates and self-pairs
                pair = tuple(sorted([str(qual1), str(qual2)]))  # Sort to avoid (A,B) vs (B,A)
                qualifier_pairs.append(pair)

        for i, tree1 in enumerate(trial_group['tree_number']):
            for tree2 in trial_group['tree_number'][i+1:]:  # Avoid duplicates and self-pairs
                pair = tuple(sorted([str(tree1), str(tree2)]))
                tree_number_pairs.append(pair)

    qualifier_combination_counts = pd.Series(qualifier_combinations).value_counts().reset_index()
    qualifier_combination_counts.columns = ['combination', 'count']
    
    tree_number_combination_counts = pd.Series(tree_number_combinations).value_counts().reset_index()
    tree_number_combination_counts.columns = ['combination', 'count']

    qualifier_pair_counts = pd.Series(qualifier_pairs).value_counts().reset_index()
    qualifier_pair_counts.columns = ['pair', 'count']
    qualifier_pair_counts['pair_string'] = qualifier_pair_counts['pair'].apply(lambda x: f"{x[0]} / {x[1]}")
    
    tree_number_pair_counts = pd.Series(tree_number_pairs).value_counts().reset_index()
    tree_number_pair_counts.columns = ['pair', 'count']
    tree_number_pair_counts['pair_string'] = tree_number_pair_counts['pair'].apply(lambda x: f"{x[0]} / {x[1]}")

    # Plot combinations
    plotting.bar_plot(
        data=qualifier_combination_counts.head(20),
        x='combination',
        y='count',
        xlabel='Mesh Term Qualifier Combination',
        ylabel='Count',
        save_name='top20_mesh_qualifier_combinations_trials.png',
        path=DATA_PATHS['img']['trials']
    )

    plotting.bar_plot(
        data=qualifier_pair_counts.head(20),
        x='pair_string',
        y='count',
        xlabel='Mesh Term Qualifier Pair',
        ylabel='Count',
        save_name='top20_mesh_qualifier_pairs_trials.png',
        path=DATA_PATHS['img']['trials']
    )
    
    plotting.bar_plot(
        data=tree_number_pair_counts.head(20),
        x='pair_string',
        y='count',
        xlabel='Mesh Term Tree Number Pair',
        ylabel='Count',
        save_name='top20_mesh_tree_number_pairs_trials.png',
        path=DATA_PATHS['img']['trials']
    ) 

    # requests
    for req_id, req_group in df_requests.groupby('request_id'):
        
        # Store full combinations (comma-separated)
        if len(req_group['qualifier']) > 0:
            qualifier_combinations.append(', '.join(sorted(req_group['qualifier'].dropna().unique())))
        if len(req_group['tree_number']) > 0:
            tree_number_combinations.append(', '.join(sorted(req_group['tree_number'].dropna().unique())))

        # Generate and store pairs
        for i, qual1 in enumerate(req_group['qualifier']):
            for qual2 in req_group['qualifier'][i+1:]:  # Avoid duplicates and self-pairs
                pair = tuple(sorted([str(qual1), str(qual2)]))  # Sort to avoid (A,B) vs (B,A)
                qualifier_pairs.append(pair)

        for i, tree1 in enumerate(req_group['tree_number']):
            for tree2 in req_group['tree_number'][i+1:]:  # Avoid duplicates and self-pairs
                pair = tuple(sorted([str(tree1), str(tree2)]))
                tree_number_pairs.append(pair)

    qualifier_combination_counts = pd.Series(qualifier_combinations).value_counts().reset_index()
    qualifier_combination_counts.columns = ['combination', 'count']
    
    tree_number_combination_counts = pd.Series(tree_number_combinations).value_counts().reset_index()
    tree_number_combination_counts.columns = ['combination', 'count']

    qualifier_pair_counts = pd.Series(qualifier_pairs).value_counts().reset_index()
    qualifier_pair_counts.columns = ['pair', 'count']
    qualifier_pair_counts['pair_string'] = qualifier_pair_counts['pair'].apply(lambda x: f"{x[0]} / {x[1]}")
    
    tree_number_pair_counts = pd.Series(tree_number_pairs).value_counts().reset_index()
    tree_number_pair_counts.columns = ['pair', 'count']
    tree_number_pair_counts['pair_string'] = tree_number_pair_counts['pair'].apply(lambda x: f"{x[0]} / {x[1]}")
    
    # Plot combinations
    plotting.bar_plot(
        data=qualifier_combination_counts.head(20),
        x='combination',
        y='count',
        xlabel='Mesh Term Qualifier Combination',
        ylabel='Count',
        save_name='top20_mesh_qualifier_combinations_requests.png',
        path=DATA_PATHS['img']['requests']
    )
    plotting.bar_plot(
        data=qualifier_pair_counts.head(20),
        x='pair_string',
        y='count',
        xlabel='Mesh Term Qualifier Pair',
        ylabel='Count',
        save_name='top20_mesh_qualifier_pairs_requests.png',
        path=DATA_PATHS['img']['requests']
    )
    plotting.bar_plot(
        data=tree_number_pair_counts.head(20),
        x='pair_string',
        y='count',
        xlabel='Mesh Term Tree Number Pair',
        ylabel='Count',
        save_name='top20_mesh_tree_number_pairs_requests.png',
        path=DATA_PATHS['img']['requests']
    )

    # publications
    for pub_id, pub_group in df_publications.groupby('doi'):

        # Store full combinations (comma-separated)
        if len(pub_group['qualifier']) > 0:
            qualifier_combinations.append(', '.join(sorted(pub_group['qualifier'].dropna().unique())))
        if len(pub_group['tree_number']) > 0:
            tree_number_combinations.append(', '.join(sorted(pub_group['tree_number'].dropna().unique())))

        # Generate and store pairs
        for i, qual1 in enumerate(pub_group['qualifier']):
            for qual2 in pub_group['qualifier'][i+1:]:  # Avoid duplicates and self-pairs
                pair = tuple(sorted([str(qual1), str(qual2)]))  # Sort to avoid (A,B) vs (B,A)
                qualifier_pairs.append(pair)

        for i, tree1 in enumerate(pub_group['tree_number']):
            for tree2 in pub_group['tree_number'][i+1:]:  # Avoid duplicates and self-pairs
                pair = tuple(sorted([str(tree1), str(tree2)]))
                tree_number_pairs.append(pair)
    
    qualifier_combination_counts = pd.Series(qualifier_combinations).value_counts().reset_index()
    qualifier_combination_counts.columns = ['combination', 'count']

    tree_number_combination_counts = pd.Series(tree_number_combinations).value_counts().reset_index()
    tree_number_combination_counts.columns = ['combination', 'count']

    qualifier_pair_counts = pd.Series(qualifier_pairs).value_counts().reset_index()
    qualifier_pair_counts.columns = ['pair', 'count']

    qualifier_pair_counts['pair_string'] = qualifier_pair_counts['pair'].apply(lambda x: f"{x[0]} / {x[1]}")
    tree_number_pair_counts = pd.Series(tree_number_pairs).value_counts().reset_index()
    tree_number_pair_counts.columns = ['pair', 'count']
    tree_number_pair_counts['pair_string'] = tree_number_pair_counts['pair'].apply(lambda x: f"{x[0]} / {x[1]}")

    # Plot combinations
    plotting.bar_plot(
        data=qualifier_combination_counts.head(20),
        x='combination',
        y='count',
        xlabel='Mesh Term Qualifier Combination',
        ylabel='Count',
        save_name='top20_mesh_qualifier_combinations_publications.png',
        path=DATA_PATHS['img']['publications']
    )
    plotting.bar_plot(
        data=qualifier_pair_counts.head(20),
        x='pair_string',
        y='count',
        xlabel='Mesh Term Qualifier Pair',
        ylabel='Count',
        save_name='top20_mesh_qualifier_pairs_publications.png',
        path=DATA_PATHS['img']['publications']
    )
    plotting.bar_plot(
        data=tree_number_pair_counts.head(20),
        x='pair_string',
        y='count',
        xlabel='Mesh Term Tree Number Pair',
        ylabel='Count',
        save_name='top20_mesh_tree_number_pairs_publications.png',
        path=DATA_PATHS['img']['publications']
    )
    logger.info("Mesh terms combinations analysis completed and plots saved.")

    # =========================================
    # Unique Tree number per trial
    unique_tree_numbers = trials.dropna(subset=['tree_number']).groupby('nct_id')['tree_number'].nunique().reset_index()
    plotting.histogram(
        data=unique_tree_numbers,
        column='tree_number',
        xlabel='Number of Unique Mesh Term Tree Numbers',
        ylabel='Number of Trials',
        save_name='unique_mesh_tree_numbers_per_trial_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=range(1, unique_tree_numbers['tree_number'].max() + 1)
    )

    # Save statistics to text file
    stats_path = os.path.join(DATA_PATHS['descriptive'], 'unique_tree_numbers_per_trial_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Unique Tree Numbers per Trial Statistics\n")
        f.write("=========================================\n")
        f.write(f"Total trials: {len(unique_tree_numbers)}\n")
        f.write(f"Total unique tree numbers: {unique_tree_numbers['tree_number'].sum()}\n")
        f.write(f"Mean unique tree numbers: {unique_tree_numbers['tree_number'].mean():.2f}\n")
        f.write(f"Median unique tree numbers: {unique_tree_numbers['tree_number'].median():.2f}\n")
        f.write(f"Max unique tree numbers: {unique_tree_numbers['tree_number'].max()}\n")
        f.write(f"Min unique tree numbers: {unique_tree_numbers['tree_number'].min()}\n")
        f.write(f"Std deviation: {unique_tree_numbers['tree_number'].std():.2f}\n")
        f.write(f"95th percentile: {unique_tree_numbers['tree_number'].quantile(0.95):.2f}\n")
        f.write(f"99th percentile: {unique_tree_numbers['tree_number'].quantile(0.99):.2f}\n")
    logger.info("Unique tree numbers per trial statistics saved.")

    # =========================================
    # Unique Qualifier per trial
    unique_qualifiers = trials.dropna(subset=['qualifier']).groupby('nct_id')['qualifier'].nunique().reset_index()
    plotting.histogram(
        data=unique_qualifiers,
        column='qualifier',
        xlabel='Number of Unique Mesh Term Qualifiers',
        ylabel='Number of Trials',
        save_name='unique_mesh_qualifiers_per_trial_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=range(1, unique_qualifiers['qualifier'].max() + 1)
    )
    logger.info("Unique qualifiers per trial analysis completed and plot saved.")

    # Save statistics to text file
    stats_path_qual = os.path.join(DATA_PATHS['descriptive'], 'unique_qualifiers_per_trial_statistics.txt')
    with open(stats_path_qual, 'w') as f:
        f.write("Unique Qualifiers per Trial Statistics\n")
        f.write("=========================================\n")
        f.write(f"Total trials: {len(unique_qualifiers)}\n")
        f.write(f"Total unique qualifiers: {unique_qualifiers['qualifier'].sum()}\n")
        f.write(f"Mean unique qualifiers: {unique_qualifiers['qualifier'].mean():.2f}\n")
        f.write(f"Median unique qualifiers: {unique_qualifiers['qualifier'].median():.2f}\n")
        f.write(f"Max unique qualifiers: {unique_qualifiers['qualifier'].max()}\n")
        f.write(f"Min unique qualifiers: {unique_qualifiers['qualifier'].min()}\n")
        f.write(f"Std deviation: {unique_qualifiers['qualifier'].std():.2f}\n")
        f.write(f"95th percentile: {unique_qualifiers['qualifier'].quantile(0.95):.2f}\n")
        f.write(f"99th percentile: {unique_qualifiers['qualifier'].quantile(0.99):.2f}\n")
    logger.info("Unique qualifiers per trial statistics saved.")

    # =========================================
    # Unique Tree number per request
    unique_tree_numbers_req = requests.dropna(subset=['tree_number']).groupby('request_id')['tree_number'].nunique().reset_index()
    plotting.histogram(
        data=unique_tree_numbers_req,
        column='tree_number',
        xlabel='Number of Unique Mesh Term Tree Numbers',
        ylabel='Number of Requests',
        save_name='unique_mesh_tree_numbers_per_request_distribution.png',
        path=DATA_PATHS['img']['requests'],
        bins=range(1, unique_tree_numbers_req['tree_number'].max() + 1)
    )
    logger.info("Unique tree numbers per request analysis completed and plot saved.")

    #Save statistics to text file
    stats_path_req = os.path.join(DATA_PATHS['descriptive'], 'unique_tree_numbers_per_request_statistics.txt')
    with open(stats_path_req, 'w') as f:
        f.write("Unique Tree Numbers per Request Statistics\n")
        f.write("=========================================\n")
        f.write(f"Total requests: {len(unique_tree_numbers_req)}\n")
        f.write(f"Total unique tree numbers: {unique_tree_numbers_req['tree_number'].sum()}\n")
        f.write(f"Mean unique tree numbers: {unique_tree_numbers_req['tree_number'].mean():.2f}\n")
        f.write(f"Median unique tree numbers: {unique_tree_numbers_req['tree_number'].median():.2f}\n")
        f.write(f"Max unique tree numbers: {unique_tree_numbers_req['tree_number'].max()}\n")
        f.write(f"Min unique tree numbers: {unique_tree_numbers_req['tree_number'].min()}\n")
        f.write(f"Std deviation: {unique_tree_numbers_req['tree_number'].std():.2f}\n")
        f.write(f"95th percentile: {unique_tree_numbers_req['tree_number'].quantile(0.95):.2f}\n")
        f.write(f"99th percentile: {unique_tree_numbers_req['tree_number'].quantile(0.99):.2f}\n")
    logger.info("Unique tree numbers per request statistics saved.")

    # =========================================
    # Unique Qualifiers per Request
    unique_qualifiers_req = requests.dropna(subset=['qualifier']).groupby('request_id')['qualifier'].nunique().reset_index()
    plotting.histogram(
        data=unique_qualifiers_req,
        column='qualifier',
        xlabel='Number of Unique Mesh Term Qualifiers',
        ylabel='Number of Requests',
        save_name='unique_mesh_qualifiers_per_request_distribution.png',
        path=DATA_PATHS['img']['requests'],
        bins=range(1, unique_qualifiers_req['qualifier'].max() + 1)
    )
    logger.info("Unique qualifiers per request analysis completed and plot saved.")
    # Save statistics to text file
    stats_path_qual_req = os.path.join(DATA_PATHS['descriptive'], 'unique_qualifiers_per_request_statistics.txt')
    with open(stats_path_qual_req, 'w') as f:
        f.write("Unique Qualifiers per Request Statistics\n")
        f.write("=========================================\n")
        f.write(f"Total requests: {len(unique_qualifiers_req)}\n")
        f.write(f"Total unique qualifiers: {unique_qualifiers_req['qualifier'].sum()}\n")
        f.write(f"Mean unique qualifiers: {unique_qualifiers_req['qualifier'].mean():.2f}\n")
        f.write(f"Median unique qualifiers: {unique_qualifiers_req['qualifier'].median():.2f}\n")
        f.write(f"Max unique qualifiers: {unique_qualifiers_req['qualifier'].max()}\n")
        f.write(f"Min unique qualifiers: {unique_qualifiers_req['qualifier'].min()}\n")
        f.write(f"Std deviation: {unique_qualifiers_req['qualifier'].std():.2f}\n")
        f.write(f"95th percentile: {unique_qualifiers_req['qualifier'].quantile(0.95):.2f}\n")
        f.write(f"99th percentile: {unique_qualifiers_req['qualifier'].quantile(0.99):.2f}\n")
    logger.info("Unique qualifiers per request statistics saved.")
    
def outcome_type_analysis(logger=None): # done
    """Outcome Type - secondary outcomes available or not."""
    db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db.get_table_data(
        columns=['outcome_type', 'nct_id'],
        table_name='outcomes'
    )
    if df is None or df.empty:
        logger.error("No outcome data found.")
        return
    
    outcome_type_summary = df.groupby('outcome_type').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=outcome_type_summary,
        x='outcome_type',
        y='nct_count',
        xlabel='Outcome Type',
        ylabel='Number of Trials',
        save_name='trials_by_outcome_type.png',
        path=DATA_PATHS['img']['trials']
    )   
    logger.info("Outcome type analysis completed and plot saved.")

def eligibility_criteria_analysis(logger=None): # done
    """Perform eligibility criteria analysis and generate plots.
    Important criterias like Gender, age limits, healthy volunteers, etc. will be analyzed.
    """
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    # gender - all, male, female
    df = db_manager.get_table_data(
        columns=['gender', 'nct_id'],
        table_name='eligibility_criteria'
    )
    if df is None or df.empty:
        logger.error("No eligibility criteria data found.")
        return

    gender_summary = df.groupby('gender').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=gender_summary,
        x='gender',
        y='nct_count',
        xlabel='Gender',
        ylabel='Number of Trials',
        save_name='trials_by_gender.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Gender analysis completed and plot saved.")

    # healthy volunteers - yes, no, unknown
    healthy_df = db_manager.get_table_data(
        columns=['healthy_volunteers', 'nct_id'],
        table_name='eligibility_criteria'
    )
    if healthy_df is None or healthy_df.empty:
        logger.error("No healthy volunteers data found.")
        return
    
    healthy_summary = healthy_df.groupby('healthy_volunteers').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=healthy_summary,
        x='healthy_volunteers',
        y='nct_count',
        xlabel='Healthy Volunteers',
        ylabel='Number of Trials',
        save_name='trials_by_healthy_volunteers.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Healthy volunteers analysis completed and plot saved.")

def status_analysis(logger=None):
    """Perform status analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['status', 'nct_id'],
        table_name='clinical_trials'
    )
    if df is None or df.empty:
        logger.error("No overall status data found.")
        return
    
    status_summary = df.groupby('status').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)

    plotting.bar_plot(
        data=status_summary,
        x='status',
        y='nct_count',
        xlabel='Status',
        ylabel='Number of Trials',
        save_name='trials_by_status.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Status analysis completed and plot saved.")

def sponsors_analysis(logger=None):
    """Perform sponsors analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['agency_class', 'type', 'nct_id'],
        table_name='sponsors'
    )
    if df is None or df.empty:
        logger.error("No sponsor data found.")
        return

    # Create pivot table for stacked bar chart
    sponsor_pivot = df.groupby(['agency_class', 'type']).size().unstack(fill_value=0)
    
    # Create stacked bar plot
    plotting.stacked_bar_plot(
        data=sponsor_pivot,
        xlabel='Agency Class of Sponsor',
        ylabel='Number of Trials',
        save_name='trials_by_agency_class_stacked.png',
        path=DATA_PATHS['img']['trials']
    )
    
    # Properly create the summary with correct column names
    sponsor_summary = df.groupby(['agency_class', 'type']).size().reset_index(name='nct_count')
    
    plotting.bar_plot(
        data=sponsor_summary,
        x='agency_class',
        y='nct_count',  
        color_column='type',  
        xlabel='Agency Class of Sponsor',
        ylabel='Number of Trials',
        save_name='trials_by_agency_class_colored.png',
        path=DATA_PATHS['img']['trials']
    )
    
    logger.info("Sponsors analysis completed and plot saved.")

def design_analysis(logger=None):
    """Perform design analysis and generate plots."""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    df = db_manager.get_table_data(
        columns=['allocation', 'intervention_model', 'primary_purpose', 'nct_id'],
        table_name='design'
    )
    if df is None or df.empty:
        logger.error("No design data found.")
        return
    
    # Allocation
    allocation_summary = df.groupby('allocation').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=allocation_summary,
        x='allocation',
        y='nct_count',
        xlabel='Allocation',
        ylabel='Number of Trials',
        save_name='trials_by_allocation.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Allocation analysis completed and plot saved.")

    # Masking
    masking_summary = df.groupby('intervention_model').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=masking_summary,
        x='intervention_model',
        y='nct_count',
        xlabel='Intervention Model',
        ylabel='Number of Trials',
        save_name='trials_by_intervention_model.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Masking analysis completed and plot saved.")

    # Primary Purpose
    primary_purpose_summary = df.groupby('primary_purpose').size().reset_index(name='nct_count').sort_values(by='nct_count', ascending=False)
    plotting.bar_plot(
        data=primary_purpose_summary,
        x='primary_purpose',
        y='nct_count',
        xlabel='Primary Purpose',
        ylabel='Number of Trials',
        save_name='trials_by_primary_purpose.png',
        path=DATA_PATHS['img']['trials']
    )
    logger.info("Primary purpose analysis completed and plot saved.")

def completeness_check(logger=None):
    """Calculate completeness percentage per trial by counting empty cells across all AACT tables."""
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import matplotlib
    matplotlib.use('Agg')
    
    db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    # Get available trials - SAMPLE SIZE
    trials = db.get_table_data(
        columns=['nct_id'],
        table_name='clinical_trials'
    )
    if trials is None or trials.empty:
        logger.error("No clinical trials data found for completeness check.")
        return
    
    nct_ids = trials['nct_id'].tolist()
    logger.info(f"Checking completeness for {len(nct_ids)} trials")
    
    aact_tables = [
        "baseline_counts", "baseline_measurements", "brief_summaries", "browse_conditions",
        "browse_interventions", "calculated_values", "categories", "central_contacts",
        "conditions", "countries", "design_group_interventions", "design_groups",
        "design_outcomes", "designs", "detailed_descriptions", "documents",
        "drop_withdrawals", "eligibilities", "facilities", "facility_contacts",
        "facility_investigators", "id_information", "intervention_other_names",
        "interventions", "ipd_information_types", "keywords", "links", "milestones",
        "outcome_analyses", "outcome_analysis_groups", "outcome_counts",
        "outcome_measurements", "outcomes", "overall_officials", "participant_flows",
        "pending_results", "provided_documents", "reported_event_totals",
        "reported_events", "responsible_parties", "result_agreements",
        "result_contacts", "result_groups", "retractions", "search_results",
        "search_term_results", "sponsors", "studies", "study_references"
    ]
    
    def process_batch_of_trials(nct_batch, logger=logger, AACT_SCHEMA="ctgov"):
        """
        Process a batch of NCT IDs using server-side null counting.
        Works with execute_raw_query(query: str, params: dict) -> list[dict].
        """
        aact_db = None
        try:
            # make sure your manager here is the one that exposes execute_raw_query exactly as shown
            aact_db = DatabaseManager()
            aact_db.connect_aact()
            batch_results = []

            for table_name in aact_tables:
                try:
                    col_query = """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                        AND table_name = :table
                        AND column_name <> 'nct_id';
                    """
                    cols_rows = aact_db.execute_raw_query(col_query, {"schema": AACT_SCHEMA, "table": table_name})
                    if not cols_rows:  # list is empty
                        logger.debug(f"No data columns for {AACT_SCHEMA}.{table_name}; skipping.")
                        continue
                    columns = [r["column_name"] for r in cols_rows]

                    # --- 2) build SUM(CASE ...) across all columns (fully qualified is optional here)
                    null_sums = " + ".join([f"(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in columns])

                    # --- 3) build an IN (...) with named placeholders because your wrapper re-wraps text()
                    if not nct_batch:
                        continue
                    placeholders = ",".join([f":id{i}" for i in range(len(nct_batch))])
                    id_params = {f"id{i}": v for i, v in enumerate(nct_batch)}

                    sql = f"""
                        SELECT
                            nct_id,
                            COUNT(*) * :ncols AS total_cells,
                            SUM({null_sums}) AS empty_cells
                        FROM {AACT_SCHEMA}.{table_name}
                        WHERE nct_id IN ({placeholders})
                        GROUP BY nct_id;
                    """

                    params = {"ncols": len(columns), **id_params}
                    rows = aact_db.execute_raw_query(sql, params)

                    if rows:
                        # rows is list[dict]; append directly
                        for r in rows:
                            # r keys come from the SELECT aliases
                            batch_results.append({
                                "nct_id": r["nct_id"],
                                "table_name": table_name,
                                "total_cells": int(r["total_cells"] or 0),
                                "empty_cells": int(r["empty_cells"] or 0),
                            })
                    else:
                        logger.debug(f"No rows for {AACT_SCHEMA}.{table_name} in batch of {len(nct_batch)} ids.")

                except Exception as e:
                    logger.warning(f"Table {AACT_SCHEMA}.{table_name} failed: {e}")
                    continue

            return batch_results

        except Exception as e:
            logger.error(f"Error connecting to AACT database: {e}")
            return []
        finally:
            if aact_db is not None:
                try:
                    aact_db.close_aact()
                except Exception:
                    pass


    # OPTIMIZED: Process in larger batches
    batch_size = 50
    batches = [nct_ids[i:i + batch_size] for i in range(0, len(nct_ids), batch_size)]
    if not batches:
        logger.error("No batches created for completeness check.")
        return
    
    all_results = []
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_batch = {executor.submit(process_batch_of_trials, batch, logger): batch for batch in batches}
        
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
    
    if not all_results or len(all_results) == 0:
        logger.error("No completeness data retrieved from AACT.")
        return
    
    # Aggregate per NCT ID across all tables
    from collections import defaultdict
    agg = defaultdict(lambda: {'total_cells': 0, 'empty_cells': 0})
    
    for row in all_results:  # row is a dict with keys: nct_id, table_name, total_cells, empty_cells
        # Be defensive in case something odd slipped in
        if not isinstance(row, dict):
            logger.error(f"Unexpected row type in all_results: {type(row)}; skipping")
            continue
        nct_id = row.get('nct_id')
        if nct_id is None:
            logger.error(f"Row missing nct_id: {row}; skipping")
            continue
        agg[nct_id]['total_cells'] += int(row.get('total_cells', 0))
        agg[nct_id]['empty_cells'] += int(row.get('empty_cells', 0))

        
        # Build final completeness list
        completeness_results = []
        for nct_id, vals in agg.items():
            total_cells = vals['total_cells']
            empty_cells = vals['empty_cells']
            completeness_percentage = (100.0 * (total_cells - empty_cells) / total_cells) if total_cells > 0 else 0.0
            completeness_results.append({
                'nct_id': nct_id,
                'total_cells': total_cells,
                'empty_cells': empty_cells,
                'completeness_percentage': completeness_percentage
            })

    
    # Convert to DataFrame and continue with plotting...
    df_completeness = pd.DataFrame(completeness_results)
    
    if df_completeness.empty:
        logger.error("No completeness results to process.")
        return
    # Generate histogram
    plotting.histogram(
        data=df_completeness,
        column='completeness_percentage',
        xlabel='Completeness Percentage (%)',
        ylabel='Number of Trials',
        save_name='trial_metadata_completeness_distribution_batch.png',
        path=DATA_PATHS['img']['trials'],
        bins=20,
        kde=False
    )
    
    # Save results
    csv_path = os.path.join(DATA_PATHS['descriptive'], 'trial_completeness_batch.csv')
    df_completeness.to_csv(csv_path, index=False)
    
    logger.info("Completeness check completed successfully")
    
    db.close()
    return df_completeness

def document_completeness_check(logger=None):
    """Document completeness analysis - number of document types per trial."""
    db = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()

    # Document Completeness
    document = db.get_table_data(
        columns=['nct_id', 'document_type'],
        table_name='documents'
    )

    if document is None or document.empty:
        logger.error("No document data found for completeness check.")
        return

    doc = document.copy()
    doc_summary = document.groupby('nct_id').agg({'document_type': 'nunique'}).reset_index()
    doc_summary.columns = ['nct_id', 'unique_document_types']

    if doc_summary.empty:
        logger.error("No document summary data to process.")
        return

    documents_sum = doc.groupby('nct_id').size().reset_index(name='total_documents')

    if documents_sum.empty:
        logger.error("No documents sum data to process.")
        return

    logger.error("Starting document completeness plotting...") 
    # plotting document completeness
    plotting.histogram(
        data=doc_summary,
        column='unique_document_types',
        xlabel='Number of Unique Document Types',
        ylabel='Number of Trials',
        save_name='trial_document_type_completeness_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=range(1, doc_summary['unique_document_types'].max() + 1),
        kde=False
    )

    # plotting the total number of documents per trial
    plotting.histogram(
        data=documents_sum,
        column='total_documents',
        xlabel='Total Number of Documents',
        ylabel='Number of Trials',
        save_name='com_trial_total_documents_distribution.png',
        path=DATA_PATHS['img']['trials'],
        bins=20,
        kde=False
    )

    logger.error("Document completeness plotting completed.")

    logger.info("Document completeness analysis completed and plots saved.")

def available_data(logger=None):
    """Check collected data for each platform"""
    db_manager = SQL_DatabaseManager(db_config=MY_DB_CONFIG, connection_name='descriptive_analysis')
    plotting = MasterPlotter()
    
    trials_platform = db_manager.get_table_data(
        columns=['trial_nct_id', 'platform_id', 'platform_name'],
        table_name='data_access'
    )
    
    if trials_platform is None or trials_platform.empty:
        logger.error("No clinical trials platform data found.")
        return
    
    # rename for clarity
    trials_platform.rename(columns={'trial_nct_id': 'nct_id'}, inplace=True)

    trials_requests = db_manager.get_table_data(
        columns=['request_id', 'nct_id'],
        table_name='request_trial_links'
    )
    
    if trials_requests is None or trials_requests.empty:
        logger.error("No request-trial link data found.")
        return
    
    csdr = trials_platform[trials_platform['platform_name'] == 'csdr']
    csdr_trials_requests = pd.merge(csdr, trials_requests, on='nct_id', how='inner')

    vivli = trials_platform[trials_platform['platform_name'] == 'vivli']
    vivli_trials_requests = pd.merge(vivli, trials_requests, on='nct_id', how='inner')

    yoda = trials_platform[trials_platform['platform_name'] == 'yoda']
    yoda_trials_requests = pd.merge(yoda, trials_requests, on='nct_id', how='inner')

    # export statistics into csv files
    stats = [
        {
            'platform': 'csdr',
            'total_trials': len(csdr),
            'total_requests': csdr_trials_requests['request_id'].nunique(),
            'requested_trials': csdr_trials_requests['nct_id'].count(),
            'requested_trials_unique': csdr_trials_requests['nct_id'].nunique()
        },
        {
            'platform': 'vivli',
            'total_trials': len(vivli),
            'total_requests': vivli_trials_requests['request_id'].nunique(),
            'requested_trials': vivli_trials_requests['nct_id'].count(),
            'requested_trials_unique': vivli_trials_requests['nct_id'].nunique()
        },
        {
            'platform': 'yoda',
            'total_trials': len(yoda),
            'total_requests': yoda_trials_requests['request_id'].nunique(),
            'requested_trials': yoda_trials_requests['nct_id'].count(),
            'requested_trials_unique': yoda_trials_requests['nct_id'].nunique()
        }
    ]

    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(DATA_PATHS['descriptive'], 'available_data_statistics.csv')
    stats_df.to_csv(stats_path, index=False)

    logger.info("Available data statistics saved.")

def run_descriptive_analysis():
    """Run descriptive analysis and generate plots."""
    setup_logging(mode='descriptive')
    logger = logging.getLogger('descriptive_logger')


    plt.rcParams['figure.max_open_warning'] = 0 
    try:
        investigator_analysis(logger)
        institution_analysis(logger) 
        enrollment_analysis(logger)
        nct_per_request_analysis(logger)
        phase_analysis(logger)
        study_type_analysis(logger)
        year_analysis(logger)
        partial_year(logger)
        condition_analysis(logger)
        intervention_analysis(logger)
        outcome_type_analysis(logger)
        eligibility_criteria_analysis(logger)
        mesh_terms_analysis(logger)
        status_analysis(logger)
        logger.error("Starting sponsors analysis...")  # Add this line
        sponsors_analysis(logger)
        logger.error("Sponsors analysis completed.")  # Add this line
        logger.error("Starting design analysis...")  # Add this line
        design_analysis(logger)
        logger.error("Design analysis completed.")  # Add this line
        logger.error("Starting completeness check analysis...")  # Add this line
        #completeness_check(logger)
        logger.error("Completeness check completed.")  # Add this line
        available_data(logger)
        logger.error("Starting document completeness check...")  # Add this line
        document_completeness_check(logger)
        logger.error("Document completeness check completed.")
    except Exception as e:
        logger.error(f"An error occurred during descriptive analysis: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")  # Add traceback
    finally:
        plt.close('all')
        logger.info("All plots generated and saved successfully.")


if __name__ == "__main__":
    run_descriptive_analysis()



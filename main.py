#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: Value Creation of Data Platforms through Secondary Use
Author: Felix Haas
Description: Main orchestration script for clinical trial data analysis
"""

import logging
import argparse
import sys
from pathlib import Path

# Import project modules
from config import PROJECT_SETTINGS, LOGGING_CONFIG
from db.sqlalchemy_connector import DatabaseManager
from scrapers.manager import run_all_scrapers
from scripts.aact_fetcher import AACTDataFetcher

# Configure logging
logging.basicConfig(
    level=getattr(logging, PROJECT_SETTINGS.get("log_level", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_database():
    """Initialize and setup the database"""
    logger.info("Setting up database...")
    
    db_manager = DatabaseManager()
    db_manager.connect_local()
    db_manager.setup_local_database()
    
    logger.info("Database setup completed")
    return db_manager


def collect_platform_data():
    """Run all platform scrapers to collect request data"""
    logger.info("Starting platform data collection...")
    
    try:
        run_all_scrapers()
        logger.info("Platform data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during platform data collection: {e}")
        return False


def enrich_with_aact_data():
    """Enrich collected data with AACT clinical trial information"""
    logger.info("Starting AACT data enrichment...")
    
    try:
        db_manager = DatabaseManager()
        db_manager.connect_all()
        
        # Get all unique NCT IDs from requests
        query = """
        SELECT DISTINCT nct_id 
        FROM request_trial_links 
        WHERE nct_id IS NOT NULL
        """
        
        nct_ids_df = db_manager.local_db.query_to_dataframe(query)
        nct_ids = nct_ids_df['nct_id'].tolist()
        
        logger.info(f"Found {len(nct_ids)} unique NCT IDs to enrich")
        
        # Fetch and update with AACT data
        aact_fetcher = AACTDataFetcher(db_manager)
        success = aact_fetcher.update_local_trials(nct_ids)
        
        db_manager.close_all()
        
        if success:
            logger.info("AACT data enrichment completed successfully")
        else:
            logger.error("AACT data enrichment failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error during AACT data enrichment: {e}")
        return False


def run_descriptive_analysis():
    """Run descriptive statistical analysis"""
    logger.info("Starting descriptive analysis...")
    
    try:
        # Import analysis module (to be created)
        from analysis.descriptive_analysis import DescriptiveAnalyzer
        
        analyzer = DescriptiveAnalyzer()
        analyzer.run_all_analyses()
        
        logger.info("Descriptive analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"Error during descriptive analysis: {e}")
        return False


def run_quantitative_analysis():
    """Run quantitative analysis"""
    logger.info("Starting quantitative analysis...")
    
    try:
        # Import analysis module (to be created)
        from analysis.quantitative_analysis import QuantitativeAnalyzer
        
        analyzer = QuantitativeAnalyzer()
        analyzer.run_all_analyses()
        
        logger.info("Quantitative analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"Error during quantitative analysis: {e}")
        return False


def run_hypothesis_testing():
    """Run hierarchical regression for hypothesis testing"""
    logger.info("Starting hypothesis testing with hierarchical regression...")
    
    try:
        # Import analysis module (to be created)
        from analysis.hypothesis_testing import HypothesisAnalyzer
        
        analyzer = HypothesisAnalyzer()
        analyzer.run_hierarchical_regression()
        
        logger.info("Hypothesis testing completed")
        return True
        
    except Exception as e:
        logger.error(f"Error during hypothesis testing: {e}")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Data Analysis Pipeline"
    )
    
    parser.add_argument(
        '--step', 
        choices=['setup', 'scrape', 'enrich', 'descriptive', 'quantitative', 'hypothesis', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Run in dry-run mode (no database changes)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Update project settings based on arguments
    if args.dry_run:
        PROJECT_SETTINGS['dry_run'] = True
        logger.info("Running in DRY-RUN mode")
    
    if args.debug:
        PROJECT_SETTINGS['debug'] = True
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    logger.info("=" * 80)
    logger.info("CLINICAL TRIAL DATA ANALYSIS PIPELINE")
    logger.info("=" * 80)
    
    success = True
    
    # Execute requested steps
    if args.step in ['setup', 'all']:
        success &= bool(setup_database())
    
    if args.step in ['scrape', 'all'] and success:
        success &= collect_platform_data()
    
    if args.step in ['enrich', 'all'] and success:
        success &= enrich_with_aact_data()
    
    if args.step in ['descriptive', 'all'] and success:
        success &= run_descriptive_analysis()
    
    if args.step in ['quantitative', 'all'] and success:
        success &= run_quantitative_analysis()
    
    if args.step in ['hypothesis', 'all'] and success:
        success &= run_hypothesis_testing()
    
    # Final status
    if success:
        logger.info("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
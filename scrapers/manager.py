# scrapers/manager.py

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scrapers.yoda import YodaScraper
from scrapers.vivli import VivliScraper
from scrapers.csdr import CSDRScraper
from scrapers.aact import AACTConnector

from config import PLATTFORM_IDS, PROJECT_SETTINGS, LOGGING_CONFIG
from db.sqlalchemy_connector import DatabaseManager


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
    elif mode == 'scraper':
        config['root']['handlers'] = ['scraper_handler']
    elif mode == 'db':
        config['root']['handlers'] = ['db_handler']
    elif mode == 'api':
        config['root']['handlers'] = ['api_handler']
    else:
        raise Exception(f"Unknown logging mode: {mode}")
    
    # Apply the configuration for ALL modes
    logging.config.dictConfig(config)
    
    # Explicitly remove any console handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

def run_all_scrapers():
    """Run all scraper in a parallel manner."""
    setup_logging(mode='scraper')
    logger = logging.getLogger('ScraperManager')
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.setup_local_database()
    
    # Create scraper instances
    scrapers = [
        YodaScraper(platform_id=PLATTFORM_IDS['yoda'], db_manager=db_manager),
        VivliScraper(platform_id=PLATTFORM_IDS['vivli'], db_manager=db_manager),
        CSDRScraper(platform_id=PLATTFORM_IDS['csdr'], db_manager=db_manager)
    ]
    
    # Use ThreadPoolExecutor to run scrapers in parallel
    with ThreadPoolExecutor(max_workers=len(scrapers)) as executor:
        futures = {}
        
        # Submit each scraper with its appropriate runner method
        for scraper in scrapers:
            if isinstance(scraper, YodaScraper):
                future = executor.submit(scraper.yoda_runner)
            elif isinstance(scraper, VivliScraper):
                future = executor.submit(scraper.vivli_runner)
            elif isinstance(scraper, CSDRScraper):
                future = executor.submit(scraper.csdr_runner)
            else:
                logger.error(f"Unknown scraper type: {type(scraper)}")
                continue
            
            futures[future] = scraper

        for future in futures:
            scraper = futures[future]
            try:
                future.result()  # Wait for the scraper to finish
                logger.info(f"{scraper.__class__.__name__} completed successfully.")
            except Exception as e:
                logger.error(f"Error running {scraper.__class__.__name__}: {e}")
    
    # Close database connections when done
    db_manager.close_all()

def run_one_scraper_testing():
    """Run a single scraper for testing purposes."""
    setup_logging(mode='testing')
    logger = logging.getLogger('ScraperManager')
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.connect_all()
    
    # Choose one scraper to test
    #scraper = YodaScraper(platform_id=PLATTFORM_IDS['yoda'], db_manager=db_manager)
    #scraper = VivliScraper(platform_id=PLATTFORM_IDS['vivli'], db_manager=db_manager)
    scraper = CSDRScraper(platform_id=PLATTFORM_IDS['csdr'], db_manager=db_manager)
    
    try:
        logger.debug(f"Starting {scraper.__class__.__name__} in testing mode.")
        scraper.csdr_runner()
        logger.info(f"{scraper.__class__.__name__} completed successfully.")
    except Exception as e:
        logger.error(f"Error running {scraper.__class__.__name__}: {e}")
        logger.debug("Exception details:", exc_info=True)
    
    # Close database connections when done
    db_manager.close_all()
    logger.debug("Database connections closed.")

def run_aact_connector():
    """Run the AACT database connector to fetch trials."""
    setup_logging(mode='db')
    logger = logging.getLogger('AACTConnector')
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.connect_local()
    db_manager.setup_aact_database()

    aact_scraper = AACTConnector(db_manager=db_manager)

    try:
        logger.info("Starting AACT data fetching process.")
        aact_scraper.aact_runner()
        logger.info("AACT data fetching process completed successfully.")
    except Exception as e:
        logger.error(f"Error during AACT data fetching: {e}")
        logger.debug("Exception details:", exc_info=True)
    
    # Close database connections when done
    db_manager.close_all()
    logger.info("Database connections closed.")

def run_public_disclosure_only():
    """Run only the public disclosure fetching for all scrapers."""
    setup_logging(mode='scraper')
    logger = logging.getLogger('PublicDisclosureManager')
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.connect_all()
    
    # Create scraper instances
    scrapers = [
        YodaScraper(platform_id=PLATTFORM_IDS['yoda'], db_manager=db_manager),
        VivliScraper(platform_id=PLATTFORM_IDS['vivli'], db_manager=db_manager),
        CSDRScraper(platform_id=PLATTFORM_IDS['csdr'], db_manager=db_manager)
    ]
    
    for scraper in scrapers:
        try:
            logger.info(f"Starting public disclosure fetch for {scraper.__class__.__name__}.")
            scraper.fetch_public_disclosure()
            logger.info(f"Public disclosure fetch for {scraper.__class__.__name__} completed successfully.")
        except Exception as e:
            logger.error(f"Error fetching public disclosures for {scraper.__class__.__name__}: {e}")
            logger.debug("Exception details:", exc_info=True)
    
    # Close database connections when done
    db_manager.close_all()
    logger.info("Database connections closed.")

def update_data():
    """Update existing data in database"""
    setup_logging(mode='testing')
    logger = logging.getLogger('testing_logger')
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.connect_all()
    
    # Create scraper instances
    scrapers = [
        YodaScraper(platform_id=PLATTFORM_IDS['yoda'], db_manager=db_manager),
        VivliScraper(platform_id=PLATTFORM_IDS['vivli'], db_manager=db_manager),
        CSDRScraper(platform_id=PLATTFORM_IDS['csdr'], db_manager=db_manager)
    ]
    
    # Use ThreadPoolExecutor to run scrapers in parallel
    with ThreadPoolExecutor(max_workers=len(scrapers)) as executor:
        futures = {}
        
        # Submit each scraper with its appropriate runner method
        for scraper in scrapers:
            if isinstance(scraper, YodaScraper):
                future = executor.submit(scraper.yoda_runner)
            elif isinstance(scraper, VivliScraper):
                future = executor.submit(scraper.vivli_runner)
            elif isinstance(scraper, CSDRScraper):
                future = executor.submit(scraper.csdr_runner)
            else:
                logger.error(f"Unknown scraper type: {type(scraper)}")
                continue
            
            futures[future] = scraper

        for future in futures:
            scraper = futures[future]
            try:
                future.result()  # Wait for the scraper to finish
                logger.info(f"{scraper.__class__.__name__} completed successfully.")
            except Exception as e:
                logger.error(f"Error running {scraper.__class__.__name__}: {e}")
    
    # Close database connections when done
    db_manager.close_all()

if __name__ == "__main__":
    #run_all_scrapers()
    run_one_scraper_testing()
    #run_public_disclosure_only()
    #run_aact_connector()
    #update_data()


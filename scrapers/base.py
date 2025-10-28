# scrapers/base.py

import logging
import requests as req
from abc import ABC, abstractmethod
from config import REQUEST_SETTINGS, PROJECT_SETTINGS
from db.sqlalchemy_connector import DatabaseManager

class BaseScraper(ABC):
    def __init__(self, platform_id: int, db_manager: DatabaseManager):
        self.platform_id = platform_id
        self.db_manager = db_manager
        self.db = db_manager.local_db
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set other common attributes
        self.session = None
        self.timeout = REQUEST_SETTINGS.get('timeout', 30)
        self.retry_attempts = PROJECT_SETTINGS.get('max_retries', 3)
        self.max_workers = PROJECT_SETTINGS.get('max_workers', 5)
        self.max_empty_pages = PROJECT_SETTINGS.get('max_empty_pages', 3)

    def scrape(self) -> int:
        """Scrape data and save to database - implement in subclasses"""
        raise NotImplementedError("Subclasses must implement scrape method")
    


import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Union
import datetime


import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session, DeclarativeMeta
from sqlalchemy.exc import SQLAlchemyError

from config import PROJECT_SETTINGS, MY_DB_CONFIG, AACT_DB_CONFIG, LOGGING_CONFIG
from db.sqlalchemy_connector import SQLAlchemyConnector, DatabaseManager

class SQL_DatabaseManager():
    def __init__(self, db_config: Dict[str, Any], connection_name: str = 'default'):
        # Logger
        self.logger = logging.getLogger(f'SQL_DatabaseManager.{self.__class__.__name__}')

        self.db_config = db_config
        self.connection_name = connection_name
        self.dry_run = PROJECT_SETTINGS.get('dry_run', False)

        # SQLAlchemy connector
        self.connector = SQLAlchemyConnector(db_config=self.db_config, connection_name=self.connection_name)
        self.connection_string = self.connector._create_connection_string(db_config)

        self.engine = create_engine(
            self.connection_string,
            echo=False,
            pool_pre_ping=True
            )
        
        self.SessionLocal = sessionmaker(bind=self.engine)

        self.connector._test_connection()

        if PROJECT_SETTINGS.get("debug", False):
            logging.info(f"Connected to database: {self.connection_name}")
   
    def get_session(self) -> Session:
        """Get a new SQLAlchemy session."""
        return self.connector.get_session()

    def get_table_data(self, table_name: str = 'clinical_trials', columns: Union[str, List[str]] = None, where_clause: str = None) -> Optional[pd.DataFrame]:
        """Get specific columns from a table"""
        if not self.connector.is_connected:
            self.logger.error("Database is not connected.")
            return None

        try:
            # Handle column selection
            if columns is None:
                column_str = "*"  # Select all columns
            elif isinstance(columns, str):
                column_str = columns  # Single column as string
            elif isinstance(columns, list):
                column_str = ", ".join(columns)  # Multiple columns as list
            else:
                self.logger.error(f"Invalid columns parameter: {columns}")
                return None

            # Build query
            query_str = f"SELECT {column_str} FROM {table_name}"
            
            # Add WHERE clause if provided
            if where_clause:
                query_str += f" WHERE {where_clause}"

            query = text(query_str)
            
            self.logger.info(f"Executing query: {query_str}")

            with self.get_session() as session:
                result = session.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
            self.logger.info(f"Fetched {len(df)} records from {table_name}")
            return df
            
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error during get_table_data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during get_table_data: {e}")
            return None
        
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute a raw SQL query and return the results as a DataFrame."""
        if not self.connector.is_connected:
            self.logger.error("Database is not connected.")
            return None

        try:
            with self.get_session() as session:
                result = session.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error during execute_query: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during execute_query: {e}")
            return None
        
    def close(self):
        """Close the database connection."""
        self.connector.close()
        self.logger.info("Database connection closed.")

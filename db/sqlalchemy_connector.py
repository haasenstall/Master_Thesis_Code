# db/sqlalchemy_connector.py

import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
import datetime

import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session, DeclarativeMeta
from sqlalchemy.exc import SQLAlchemyError

from config import PROJECT_SETTINGS, MY_DB_CONFIG, AACT_DB_CONFIG
from model.database_models import (
    Base, Request, ClinicalTrial, RequestTrialLink, Platform, DataAccess, 
    Institution, Investigator, Design, Outcome, EligibilityCriteria, 
    Sponsor, Country, Condition, Intervention_Mesh_Term, Intervention,
    PublicDisclosure, FocalReference, FutureCitation, MeshTerm, Document
    )


class SQLAlchemyConnector:
    def __init__(self, db_config: dict, connection_name: str = "unknown"):
        # Add logger to SQLAlchemyConnector too
        self.logger = logging.getLogger(f'SQLAlchemyConnector.{connection_name}')
        
        self.db_config = db_config
        self.connection_name = connection_name
        self.dry_run = PROJECT_SETTINGS.get("dry_run", False)
        
        # Create connection string
        self.connection_string = self._create_connection_string(db_config)
        
        # Initialize engine and session
        self.engine = create_engine(
            self.connection_string,
            echo=False, 
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Test connection
        self._test_connection()
        
        if PROJECT_SETTINGS.get("debug", False):
            logging.info(f"Connected to database: {self.connection_name}")
    
    def _create_connection_string(self, config: dict) -> str:
        """Create PostgreSQL connection string from config"""
        return (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['dbname']}"
        )
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logging.error(f"Connection failed for database: {self.connection_name}")
            raise e
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            if not self.dry_run:
                session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Session error on '{self.connection_name}': {e}")
            raise e
        finally:
            session.close()

    def is_connected(self) -> bool:
        """Check if the database connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logging.error(f"Connection check failed for database: {self.connection_name} - {e}")
            return False
    
    def create_tables(self):
        """Create all tables defined in the models"""
        if not self.dry_run:
            Base.metadata.create_all(self.engine)
            logging.info("Database tables created successfully")
        else:
            logging.info("DRY-RUN: Would create database tables")
    
    def drop_tables(self):
        """Drop all tables"""
        if not self.dry_run:
            Base.metadata.drop_all(self.engine)
            logging.info("Database tables dropped successfully")
        else:
            logging.info("DRY-RUN: Would drop database tables")
    
    def execute_raw_query(self, query: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query and return results"""
        with self.engine.connect() as conn:
            if self.dry_run:
                logging.info(f"DRY-RUN: {query} with params {params}")
                return []
            
            try:
                start_time = time.time()
                result = conn.execute(text(query), params or {})
                duration = time.time() - start_time
                
                if PROJECT_SETTINGS.get("debug", False):
                    logging.info(f"Query took {duration:.3f}s")
                
                # Return as list of dictionaries for SELECT queries
                if result.returns_rows:
                    return [dict(row._mapping) for row in result.fetchall()]
                return []
                
            except SQLAlchemyError as e:
                logging.error(f"SQL Error on '{self.connection_name}': {e}")
                raise e
    
    def query_to_dataframe(self, query: str, params: dict = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        if self.dry_run:
            logging.info(f"DRY-RUN: Would execute query: {query}")
            return pd.DataFrame()
        
        try:
            return pd.read_sql_query(query, self.engine, params=params)
        except Exception as e:
            logging.error(f"Error executing query to DataFrame: {e}")
            raise e
    
    def record_exists(self, model_class, **filters) -> bool:
        """Check if record exists using SQLAlchemy model"""
        with self.get_session() as session:
            query = session.query(model_class).filter_by(**filters)
            return session.query(query.exists()).scalar()
    
    def insert_request(self, request_data: dict) -> bool:
        """Insert a request into the database"""
        try:
            with self.get_session() as session:
                # Filter out fields that don't exist in the Request model
                valid_fields = {column.name for column in Request.__table__.columns}
                filtered_data = {k: v for k, v in request_data.items() if k in valid_fields}
                
                # Check if request already exists
                existing = session.query(Request).filter_by(
                    request_id=filtered_data['request_id']
                ).first()
                
                if existing:
                    # Update existing request
                    for key, value in filtered_data.items():
                        setattr(existing, key, value)
                    logging.info(f"Updated request {filtered_data['request_id']}")
                else:
                    # Create new request
                    request = Request(**filtered_data)
                    session.add(request)
                    logging.info(f"Inserted new request {filtered_data['request_id']}")
                
                return True
                
        except Exception as e:
            logging.error(f"Error inserting request: {e}")
            return False
    
    def insert_request_list(self, request_list: List[dict]) -> bool:
        """Insert multiple requests into the database"""
        if not request_list:
            return True
        
        try:
            with self.get_session() as session:
                valid_fields = {column.name for column in Request.__table__.columns}
                
                for request_data in request_list:
                    # Filter out invalid fields like 'detail_url'
                    filtered_data = {k: v for k, v in request_data.items() if k in valid_fields}
                    
                    # Check if request already exists
                    existing = session.query(Request).filter_by(
                        request_id=filtered_data['request_id']
                    ).first()
                    
                    if existing:
                        # Update existing request
                        for key, value in filtered_data.items():
                            setattr(existing, key, value)
                    else:
                        # Create new request
                        request = Request(**filtered_data)
                        session.add(request)
                
                logging.info(f"Processed {len(request_list)} requests")
                return True
                
        except Exception as e:
            logging.error(f"Error inserting request list: {e}")
            return False
    
    def insert_nct_request_ids(self, request_id: str, nct_ids: List[str]) -> bool:
        """Insert NCT IDs associated with a request"""
        if not nct_ids:
            return True
        
        try:
            with self.get_session() as session:
                for nct_id in nct_ids:
                    # Ensure clinical trial exists
                    trial = session.query(ClinicalTrial).filter_by(nct_id=nct_id).first()
                    if not trial:
                        trial = ClinicalTrial(nct_id=nct_id)
                        session.add(trial)
                    
                    # Create link if it doesn't exist
                    link = session.query(RequestTrialLink).filter_by(
                        request_id=request_id, nct_id=nct_id
                    ).first()
                    
                    if not link:
                        link = RequestTrialLink(request_id=request_id, nct_id=nct_id)
                        session.add(link)
                
                logging.info(f"Linked {len(nct_ids)} NCT IDs to request {request_id}")
                return True
                
        except Exception as e:
            logging.error(f"Error inserting NCT IDs: {e}")
            return False

    def insert_nct_ids(self, nct_ids: List[str]) -> bool:
        """Insert NCT IDs into the database"""
        if not nct_ids:
            return True

        try:
            with self.get_session() as session:
                # Handle both flat list and nested list structures
                flat_nct_ids = []
                for item in nct_ids:
                    if isinstance(item, list):
                        flat_nct_ids.extend(item)
                    else:
                        flat_nct_ids.append(item)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_nct_ids = []
                for nct_id in flat_nct_ids:
                    if nct_id not in seen:
                        seen.add(nct_id)
                        unique_nct_ids.append(nct_id)
                
                # Insert each NCT ID individually
                inserted_count = 0
                for nct_id in unique_nct_ids:
                    existing = session.query(ClinicalTrial).filter_by(nct_id=nct_id).first()
                    if not existing:
                        trial = ClinicalTrial(nct_id=nct_id)
                        session.add(trial)
                        inserted_count += 1
                    else:
                        self.logger.debug(f"NCT ID {nct_id} already exists in the database")

                self.logger.info(f"Inserted {inserted_count} new NCT IDs into the database")
                return True

        except Exception as e:
            self.logger.error(f"Error inserting NCT IDs: {e}")
            return False

    def link_nct_to_platform(self, nct_ids: List[str], platform_name: str) -> bool:
        """Link NCT IDs to platform"""
        try:
            with self.get_session() as session:
                # Handle both flat list and nested list structures
                flat_nct_ids = []
                for item in nct_ids:
                    if isinstance(item, list):
                        flat_nct_ids.extend(item)
                    else:
                        flat_nct_ids.append(item)
                
                # Remove duplicates
                unique_nct_ids = list(set(flat_nct_ids))
                
                # Query platform by name (case-insensitive)
                platform = session.query(Platform).filter(
                    Platform.name.ilike(platform_name)
                ).first()
                
                if not platform:
                    self.logger.error(f"Platform '{platform_name}' not found in database")
                    available_platforms = session.query(Platform.name).all()
                    self.logger.error(f"Available platforms: {[p.name for p in available_platforms]}")
                    return False
                
                links_added = 0
                for nct_id in unique_nct_ids:
                    # Check if NCT ID exists
                    nct_exists = session.query(ClinicalTrial).filter_by(nct_id=nct_id).first()
                    if not nct_exists:
                        self.logger.warning(f"NCT ID {nct_id} not found in database, skipping link")
                        continue
                    
                    # Check if link already exists
                    existing_link = session.query(DataAccess).filter_by(  
                        trial_nct_id=nct_id,
                        platform_id=platform.id
                    ).first()
                    
                    if not existing_link:
                        link = DataAccess(
                            trial_nct_id=nct_id, 
                            platform_id=platform.id,
                            platform_name=platform_name
                        )
                        session.add(link)
                        links_added += 1
        
            self.logger.info(f"Linked {links_added} NCT IDs to platform '{platform_name}'")
            return True
        
        except Exception as e:
            self.logger.error(f"Error linking NCT IDs to platform: {e}")
            return False

    def bulk_insert(self, table_name: str, data_list: List[Dict], nct_id: str = None):
        """Bulk insert data into specified table"""
        if not data_list:
            self.logger.debug(f"No data to insert for table {table_name}")
            return True

        try:
            with self.get_session() as session:
                # Map table names to SQLAlchemy models
                table_model_map = {
                    'design': Design,
                    'outcomes': Outcome,
                    'eligibility_criteria': EligibilityCriteria,
                    'sponsors': Sponsor,
                    'countries': Country,
                    'conditions': Condition,
                    'interventions_mesh_terms': Intervention_Mesh_Term,
                    'interventions': Intervention,
                    'institution': Institution,
                    'investigator': Investigator,
                    'mesh_terms': MeshTerm,
                    'documents': Document 
                }
                
                model_class = table_model_map.get(table_name)
                if not model_class:
                    self.logger.error(f"Unknown table name: {table_name}")
                    return False

                # Add nct_id to each record if provided
                if nct_id:
                    for data in data_list:
                        data['nct_id'] = nct_id

                # Create model instances
                instances = []
                for data in data_list:
                    try:
                        instance = model_class(**data)
                        instances.append(instance)
                    except Exception as e:
                        self.logger.warning(f"Error creating instance for {table_name}: {e}")
                        continue

                if instances:
                    # Use bulk_insert_mappings for better performance
                    session.bulk_insert_mappings(model_class, [instance.__dict__ for instance in instances])
                    session.commit()
                    self.logger.debug(f"Successfully inserted {len(instances)} records into {table_name}")
                
                return True

        except Exception as e:
            self.logger.error(f"Error bulk inserting into {table_name}: {e}")
            if 'session' in locals():
                session.rollback()
            return False
    
    def close(self):
        """Close database connections"""
        self.engine.dispose()
        if PROJECT_SETTINGS.get("debug", False):
            logging.info(f"Closed connection to: {self.connection_name}")

    def institution_insert(self, institution_name: str) -> bool:
        """Insert institution into database"""
        try:
            with self.get_session() as session:
                existing = session.query(Institution).filter_by(name=institution_name).first()
                
                if not existing:
                    institution = Institution(name=institution_name)
                    session.add(institution)
                    # Remove the extra session.commit() - it's handled by context manager
                    self.logger.debug(f"Inserted new institution: {institution_name}")
                    return True
                else:
                    self.logger.debug(f"Institution already exists: {institution_name}")
                    return True
                
        except Exception as e:
            self.logger.error(f"Error inserting institution {institution_name}: {e}")
            return False

    def investigator_insert(self, investigator: str) -> bool:
        """Insert investigator into database"""
        try:
            with self.get_session() as session:
                investigator = Investigator(
                    nct_id=investigator['nct_id'],
                    name=investigator['name'],
                    institution_id=investigator['affiliation'],
                    role=investigator['role']
                )
                session.add(investigator)
                # Remove the extra session.commit() - it's handled by context manager
                self.logger.debug(f"Inserted new investigator: {investigator['name']}")
                return True

        except Exception as e:
            self.logger.error(f"Error inserting investigator {investigator['name']}: {e}")
            return False


class DatabaseManager:
    """Manages connections to both local and AACT databases"""
    
    def __init__(self):
        self.logger = logging.getLogger('DatabaseManager')
        self.local_db: Optional[SQLAlchemyConnector] = None
        self.aact_db: Optional[SQLAlchemyConnector] = None
        self.connect_local()
    
    def connect_local(self):
        """Connect to local database"""
        try:
            self.local_db = SQLAlchemyConnector(MY_DB_CONFIG, "LOCAL_DB")
            self.logger.info("Connected to local database")
            return self.local_db
        except Exception as e:
            self.logger.error(f"Failed to connect to local database: {e}")
            raise e
        
    def connect_aact(self):
        """Connect to AACT database"""
        try:
            self.aact_db = SQLAlchemyConnector(AACT_DB_CONFIG, "AACT_DB")
            self.logger.info("Connected to AACT database")
            return self.aact_db
        except Exception as e:
            self.logger.error(f"Failed to connect to AACT database: {e}")
            raise e

    def connect_all(self):
        """Connect to both databases"""
        self.connect_local()
        self.connect_aact()
        self.logger.info("Connected to both local and AACT databases")

    def execute_raw_query(self, query: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute raw query on aact database"""
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []

        return self.aact_db.execute_raw_query(query, params)

    def get_all_nct_ids_from_local(self) -> List[str]:
        if not self.local_db:
            self.logger.error("Local database not connected")
            return []
        try:
            query = "SELECT nct_id FROM clinical_trials"
            result = self.local_db.execute_raw_query(query)
            nct_ids = [row['nct_id'] for row in result]
            self.logger.info(f"Retrieved {len(nct_ids)} NCT IDs from local database")
            return nct_ids
        except Exception as e:
            self.logger.error(f"Error retrieving NCT IDs from local database: {e}")
            return []

    def get_clinical_trial_from_aact(self, nct_id: str) -> Optional[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return None
        try:
            # Basic trial info
            query_studies = """
            SELECT 
                nct_id, 
                official_title as title, 
                phase, 
                overall_status AS status, 
                enrollment, 
                study_type, 
                start_date, 
                completion_date, 
                study_first_submitted_date AS date_published, 
                plan_to_share_ipd AS plan_to_share
            FROM studies
            WHERE nct_id = :nct_id;
            """

            result_study = self.aact_db.execute_raw_query(query_studies, {'nct_id': nct_id})
            if not result_study:
                self.logger.warning(f"No clinical trial found in AACT for NCT ID: {nct_id}")
                return None
        
            # Combine results
            trial_data = {**result_study[0]}
            return trial_data
        except Exception as e:
            self.logger.error(f"Error retrieving clinical trial {nct_id} from AACT: {e}")
            return None

    def insert_trial_details(self, nct_id: str, trial_data: Dict[str, Any]) -> bool:
        """Insert or update clinical trial details in local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        try:
            with self.local_db.get_session() as session:
                existing = session.query(ClinicalTrial).filter_by(nct_id=nct_id).first()
                
                if existing:
                    # Update existing trial
                    for key, value in trial_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    self.logger.debug(f"Updated clinical trial {nct_id}")
                else:
                    self.logger.warning(f"Clinical trial {nct_id} not found in local database, inserting new record")
                return True
        except Exception as e:
            self.logger.error(f"Error inserting/updating clinical trial {nct_id}: {e}")
            return False
        
    def get_institution(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                id,
                nct_id,
                name,
                city,
                country
            FROM facilities
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving institution for {nct_id} from AACT: {e}")
            return []
        
    def get_investigators(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                affiliation,
                role,
                name
            FROM overall_officials
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving investigators for {nct_id} from AACT: {e}")
            return []
        
    """def check_investigator_institution(self, investigators: Dict) -> bool:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return False
        try:
            # lookup institution if exists in my database
            for investigator in investigators:
                institution_id = investigator.get('institution_id')
                if institution_id:
                    institution = self.get_institution_by_id(institution_id)
                    if self.lookup_id_exists(
                        table = Institution,
                        id = institution_id
                    ):
                        self.logger.debug(f"Found institution for investigator {investigator['name']}: {institution['name']}")
                    else:
                        self.logger.warning(f"Institution not found for investigator {investigator['name']}")
                        self.insert_institution(institution)
            return True
        except Exception as e:
            self.logger.error(f"Error checking investigator institution: {e}")
            return False"""
        
    def get_institution_by_id(self, institution_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve institution details by ID from AACT database"""
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return None
        try:
            query = """
            SELECT
                id,
                name,
                city,
                country
            FROM facilities
            WHERE id = :institution_id;
            """
            result = self.aact_db.execute_raw_query(query, {'institution_id': institution_id})
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error retrieving institution by ID {institution_id} from AACT: {e}")
            return None
        
    def lookup_id_exists(self, table: DeclarativeMeta, id: int) -> bool:
        """Check if a record with the given ID exists in the specified table"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        try:
            with self.local_db.get_session() as session:
                exists = session.query(table).filter_by(id=id).first() is not None
                return exists
        except Exception as e:
            self.logger.error(f"Error checking existence in table {table.__tablename__} for ID {id}: {e}")
            return False

    def insert_institution(self, institution: Dict[str, Any]) -> bool:
        """Insert a new institution record into the local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        try:
            with self.local_db.get_session() as session:
                new_institution = Institution(**institution)
                session.add(new_institution)
                session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error inserting institution {institution['name']}: {e}")
            return False

    def get_design(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                allocation,
                intervention_model,
                observational_model,
                primary_purpose
            FROM designs
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving design for {nct_id} from AACT: {e}")
            return []
        
    def get_outcomes(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                outcome_type
            FROM outcomes
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving outcomes for {nct_id} from AACT: {e}")
            return []

    def get_eligibility_criteria(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                sampling_method,
                gender,
                minimum_age,
                maximum_age,
                healthy_volunteers
            FROM eligibilities
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving eligibility criteria for {nct_id} from AACT: {e}")
            return []

    def get_sponsors(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                name,
                agency_class,
                lead_or_collaborator AS type
            FROM sponsors
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving sponsors for {nct_id} from AACT: {e}")
            return []
        
    def get_countries(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                name
            FROM countries
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving countries for {nct_id} from AACT: {e}")
            return []

    def get_conditions(self, nct_id: str) -> List[str]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                downcase_mesh_term AS condition
            FROM browse_conditions
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving conditions for {nct_id} from AACT: {e}")
            return []
        
    def get_interventions_mesh_terms(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                downcase_mesh_term AS intervention
            FROM browse_interventions
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving intervention mesh terms for {nct_id} from AACT: {e}")
            return []

    def get_interventions(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                intervention_type,
                name
            FROM interventions
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving interventions for {nct_id} from AACT: {e}")
            return []

    def get_documents(self, nct_id: str) -> List[Dict[str, Any]]:
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                nct_id,
                document_type
            FROM documents
            WHERE nct_id = :nct_id;
            """
            result = self.aact_db.execute_raw_query(query, {'nct_id': nct_id})
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving documents for {nct_id} from AACT: {e}")
            return []

    def insert_table(self, nct_id: str, data: List[Dict], table_name: str):
        """Insert data into specified table with NCT ID"""
        if not data:
            self.logger.debug(f"No data to insert for {table_name}")
            return True

        try:
            # Add nct_id to each record
            for record in data:
                record['nct_id'] = nct_id

            # Use the local database bulk_insert method
            success = self.local_db.bulk_insert(table_name, data, nct_id)
            if success:
                self.logger.debug(f"Successfully inserted {len(data)} records into {table_name} for {nct_id}")
            else:
                self.logger.error(f"Failed to insert data into {table_name} for {nct_id}")
            
            return success

        except Exception as e:
            self.logger.error(f"Error inserting data into {table_name} for {nct_id}: {e}")
            return False

    def insert_public_disclosure(self, request_id: str, title: str, doi: str = None, institution: str = None, authors: str = None) -> bool:
        """Insert public disclosure information into the database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False

        try:
            # Clean and validate DOI
            if doi:
                original_doi = doi
                doi = doi.strip().rstrip('.')
                if original_doi != doi:
                    self.logger.debug(f"Cleaned DOI: '{original_doi}' -> '{doi}'")

            # Clean title
            if title:
                title = title.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
                title = title.encode('ascii', 'ignore').decode('ascii')

            with self.local_db.get_session() as session:
                # Check if request exists
                existing_request = session.query(Request).filter_by(request_id=request_id).first()
                
                if not existing_request:
                    self.logger.warning(f"Request {request_id} not found in database")
                    return False

                # Check for existing disclosure with same DOI
                if doi:
                    existing_by_doi = session.query(PublicDisclosure).filter_by(doi=doi).first()
                    if existing_by_doi:
                        self.logger.debug(f"Public disclosure with DOI {doi} already exists")
                        return True

                # Check for existing disclosure with same request_id
                existing_by_request = session.query(PublicDisclosure).filter_by(request_id=request_id).first()
                
                if existing_by_request and not doi:
                    # Update existing record without DOI
                    existing_by_request.title = title
                    self.logger.debug(f"Updated existing public disclosure for request {request_id}")
                else:
                    # Create new disclosure
                    public_disclosure = PublicDisclosure(
                        request_id=request_id,
                        title=title,
                        doi=doi,
                        platform_id=existing_request.platform_id,
                        institution=institution,
                        authors=authors
                    )
                    session.add(public_disclosure)
                    self.logger.debug(f"Inserted new public disclosure: request_id={request_id}, doi={doi}")

                session.commit()
                
                # Verify insertion
                if doi:
                    verification = session.query(PublicDisclosure).filter_by(doi=doi).first()
                    if verification:
                        self.logger.debug(f"Verified DOI {doi} is now in database with ID {verification.id}")
                    else:
                        self.logger.error(f"Failed to verify DOI {doi} insertion")
                        return False
                
                return True

        except Exception as e:
            self.logger.error(f"Error inserting public disclosure for {request_id}: {e}")
            return False

    def insert_focal_reference(self, origin_doi: str, doi:str ) -> bool:
        """Insert focal references into the database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False

        try:
            inserted_count = 0
            with self.local_db.get_session() as session:
                existing = session.query(FocalReference).filter_by(
                    reference_doi=doi,
                    origin_doi=origin_doi
                ).first()

                if not existing:
                    focal_reference = FocalReference(
                        reference_doi=doi,
                        origin_doi=origin_doi
                    )
                    session.add(focal_reference)
                    inserted_count += 1
                    self.logger.debug(f"Inserted focal reference {doi} for origin {origin_doi}")
                else:
                    self.logger.debug(f"Focal reference {doi} for origin {origin_doi} already exists")
                
            session.commit()
            
            if inserted_count > 0:
                self.logger.info(f"Inserted {inserted_count} focal references for origin {origin_doi}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error inserting focal references for {origin_doi}: {e}")
            return False

    def insert_cited_by(self, origin_doi: str, doi: str) -> bool:
        """Insert future citations into the database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False

        try:
            inserted_count = 0
            with self.local_db.get_session() as session:
                existing = session.query(FutureCitation).filter_by(
                    citation_doi=doi,
                    origin_doi=origin_doi
                ).first()

                if not existing:
                    future_citation = FutureCitation(
                        citation_doi=doi,
                        origin_doi=origin_doi
                    )
                    session.add(future_citation)
                    inserted_count += 1
                    self.logger.debug(f"Inserted future citation {doi} for origin {origin_doi}")
                else:
                    self.logger.debug(f"Future citation {doi} for origin {origin_doi} already exists")

            session.commit()
            if inserted_count > 0:
                self.logger.info(f"Inserted {inserted_count} future citations for origin {origin_doi}")
            
            return True

        except Exception as e:
            self.logger.error(f"Error inserting future citations for {origin_doi}: {e}")
            return False

    def update_public_disclosure_details(self, doi: str, authors: List[str] = None, publication_date: str = None) -> bool:
        """Update public disclosure details in the database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False

        try:
            with self.local_db.get_session() as session:
                existing = session.query(PublicDisclosure).filter_by(
                    doi=doi  # FIXED: Use doi instead of request_id
                ).first()

                if not existing:
                    self.logger.warning(f"Public disclosure with DOI {doi} not found")
                    return False
                
                # Update fields if provided
                if publication_date:
                    # Convert string to date if needed
                    if isinstance(publication_date, str):
                        from datetime import datetime
                        try:
                            existing.publication_date = datetime.strptime(publication_date, '%Y-%m-%d').date()
                        except ValueError:
                            self.logger.warning(f"Invalid date format: {publication_date}")
                    else:
                        existing.publication_date = publication_date
                        
                if authors:
                    existing.authors = ', '.join(authors) if isinstance(authors, list) else authors

                session.commit()
                self.logger.debug(f"Updated public disclosure details for DOI {doi}")
                return True

        except Exception as e:
            self.logger.error(f"Error updating public disclosure for {doi}: {e}")
            return False

    def setup_local_database(self):
        """Setup local database with initial data"""
        try:
            # Check current schema first
            self.check_database_schema()
            self.recreate_database_schema()
            
            logging.info("Local database setup completed")
            
        except Exception as e:
            logging.error(f"Error setting up local database: {e}")
            raise

    def setup_aact_database(self):
        """Setup AACT database connection"""
        try:
            self.connect_aact()
            self.logger.info("AACT database setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up AACT database: {e}")
            raise
    
    def recreate_database_schema(self):
        """Drop all tables and recreate them"""
        try:
            self.logger.info("Dropping all existing tables...")
            self.local_db.drop_tables()
            
            self.logger.info("Creating tables from models...")
            self.local_db.create_tables()
            
            # Insert initial platform data
            platforms = [
                {'id': 1, 'name': 'vivli'},
                {'id': 2, 'name': 'csdr'},
                {'id': 3, 'name': 'yoda'}
            ]
            
            with self.local_db.get_session() as session:
                for platform_data in platforms:
                    existing = session.query(Platform).filter_by(id=platform_data['id']).first()
                    if not existing:
                        platform = Platform(**platform_data)
                        session.add(platform)
            
            self.logger.info("Database schema recreated successfully")
            
        except Exception as e:
            self.logger.error(f"Error recreating database schema: {e}")
            raise
    
    def check_database_schema(self):
        """Check if database tables exist and have correct columns"""
        try:
            with self.local_db.get_session() as session:
                # Check if tables exist
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                """
                result = session.execute(text(tables_query))
                existing_tables = [row[0] for row in result.fetchall()]
                
                self.logger.info(f"Existing tables: {existing_tables}")

                required_tables = [
                    'clinical_trials',
                    'mesh_terms',
                    'public_disclosures',
                    'institutions',
                    'investigators'
                ]

                for table in required_tables:
                    if table not in existing_tables:
                        self.logger.warning(f"Missing required table: {table}")
            self.logger.info("Database schema check completed")
        except Exception as e:
            self.logger.error(f"Error checking database schema: {e}")
            raise
    
    def close_all(self):
        """Close all database connections"""
        if self.local_db:
            self.local_db.close()
            self.logger.info("Closed local database connection")
        
        if self.aact_db:
            self.aact_db.close()
            self.logger.info("Closed AACT database connection")
    
    def close_local(self):
        """Close local database connection"""
        if self.local_db:
            self.local_db.close()
            self.local_db = None
            self.logger.info("Closed local database connection")
    
    def close_aact(self):
        """Close AACT database connection"""
        if self.aact_db:
            self.aact_db.close()
            self.aact_db = None
            self.logger.info("Closed AACT database connection")

    def get_all_mesh_terms_from_aact(self) -> List[dict]:
        """Retrieve all MeSH terms from AACT database"""
        if not self.aact_db:
            self.logger.error("AACT database not connected")
            return []
        try:
            query = """
            SELECT
                qualifier,
                tree_number,
                mesh_term,
                downcase_mesh_term
            FROM mesh_terms;
            """
            result = self.aact_db.execute_raw_query(query)
            self.logger.info(f"Retrieved {len(result)} MeSH terms from AACT database")
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving MeSH terms from AACT database: {e}")
            return []

    def insert_mesh_terms(self, mesh_terms: List[dict]) -> bool:
        """Insert MeSH terms into local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        try:
            success = self.local_db.bulk_insert('mesh_terms', mesh_terms, nct_id=None)
            if success:
                self.logger.info(f"Inserted {len(mesh_terms)} MeSH terms into local database")
            else:
                self.logger.error("Failed to insert MeSH terms into local database")
            return success
        except Exception as e:
            self.logger.error(f"Error inserting MeSH terms into local database: {e}")
            return False
    
    # Add delegation methods for common database operations
    def institution_insert(self, institution_name: str) -> bool:
        """Insert institution into database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.institution_insert(institution_name)

    def investigator_insert(self, investigator_name: str, institution_id: int) -> bool:
        """Insert investigator into database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.investigator_insert(investigator_name, institution_id)

    def insert_request(self, request_data: dict) -> bool:
        """Insert request via local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.insert_request(request_data)

    def insert_request_list(self, request_list: List[dict]) -> bool:
        """Insert request list via local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.insert_request_list(request_list)

    def insert_nct_ids(self, nct_ids: List[str]) -> bool:
        """Insert NCT IDs via local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.insert_nct_ids(nct_ids)

    def link_nct_to_platform(self, nct_ids: List[str], platform_name: str) -> bool:
        """Link NCT IDs to platform via local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.link_nct_to_platform(nct_ids, platform_name)

    def insert_nct_request_ids(self, request_id: str, nct_ids: List[str]) -> bool:
        """Insert NCT request IDs via local database"""
        if not self.local_db:
            self.logger.error("Local database not connected")
            return False
        return self.local_db.insert_nct_request_ids(request_id, nct_ids)

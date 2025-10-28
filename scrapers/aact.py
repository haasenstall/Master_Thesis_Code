import re
import os
import pandas as pd
import logging
import requests as req
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from abc import ABC, abstractmethod

from config import REQUEST_SETTINGS, PROJECT_SETTINGS, DATA_PATHS
from db.sqlalchemy_connector import DatabaseManager

class AACTConnector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure AACT connection is established
        if not self.db_manager.aact_db:
            self.logger.info("Connecting to AACT database.")
            self.db_manager.connect_aact()
        
        self.logger.info("Initialized AACTConnector")

    def fetch_trials(self):
        """Connect informations about Trials from AACT database to the already existing Trials in My database."""
        self.logger.info("Get all listed Trials in My Database.")
        
        # Use the DatabaseManager methods directly
        nct_ids = self.db_manager.get_all_nct_ids_from_local()
        if not nct_ids:
            self.logger.warning("No NCT IDs found in local database")
            return

        self.logger.info(f"Found {len(nct_ids)} Trials in My Database.")

        self.logger.info("Starting to fetch Trials from AACT database.")
        successful_updates = 0
        failed_updates = 0

        for nct_id in tqdm(nct_ids, desc="Processing Trials", unit="trial"):
            try:
                # Get Basic information from main table
                trial = self.db_manager.get_clinical_trial_from_aact(nct_id)
                if not trial:
                    self.logger.warning(f"No trial found in AACT for NCT ID: {nct_id}")
                    failed_updates += 1
                    continue

                # Update trial details using DatabaseManager method
                success = self.db_manager.insert_trial_details(nct_id, trial)
                if not success:
                    self.logger.warning(f"Failed to update trial details for {nct_id}")
                    failed_updates += 1
                    continue
                
                # Get additional data using DatabaseManager methods
                institution = self.db_manager.get_institution(nct_id)
                if institution:
                    self.db_manager.insert_table(nct_id, institution, "institution")
                    self.logger.debug(f"Inserted institution for {nct_id}")
                else:
                    self.logger.error(f"No institution data found for {nct_id}")
                

                investigators = self.db_manager.get_investigators(nct_id)
                if investigators:
                    #self.db_manager.check_investigator_institution(investigators)
                    self.db_manager.insert_table(nct_id, investigators, "investigator")
                    self.logger.debug(f"Inserted investigators for {nct_id}")
                else:
                    self.logger.error(f"No investigator data found for {nct_id}")

                design = self.db_manager.get_design(nct_id)
                if design:
                    self.db_manager.insert_table(nct_id, design, "design")

                outcomes = self.db_manager.get_outcomes(nct_id)
                if outcomes:
                    self.db_manager.insert_table(nct_id, outcomes, "outcomes")

                eligibility_criteria = self.db_manager.get_eligibility_criteria(nct_id)
                if eligibility_criteria:
                    self.db_manager.insert_table(nct_id, eligibility_criteria, "eligibility_criteria")

                sponsors = self.db_manager.get_sponsors(nct_id)
                if sponsors:
                    self.db_manager.insert_table(nct_id, sponsors, "sponsors")

                countries = self.db_manager.get_countries(nct_id)
                if countries:
                    self.db_manager.insert_table(nct_id, countries, "countries")

                conditions = self.db_manager.get_conditions(nct_id)
                if conditions:
                    self.db_manager.insert_table(nct_id, conditions, "conditions")

                intervention_mesh_terms = self.db_manager.get_interventions_mesh_terms(nct_id)
                if intervention_mesh_terms:
                    self.db_manager.insert_table(nct_id, intervention_mesh_terms, "interventions_mesh_terms")

                interventions = self.db_manager.get_interventions(nct_id)
                if interventions:
                    self.db_manager.insert_table(nct_id, interventions, "interventions")

                documents = self.db_manager.get_documents(nct_id)
                if documents:
                    self.db_manager.insert_table(nct_id, documents, "documents")

                self.logger.debug(f"Updated trial {nct_id} in My Database.")
                successful_updates += 1

                self.logger.info(f"Successfully updated trial {nct_id} from AACT database.")

            except Exception as e:
                self.logger.error(f"Error processing trial {nct_id}: {e}")
                self.logger.debug("Exception details:", exc_info=True)
                failed_updates += 1

        # Summary
        total = successful_updates + failed_updates
        success_rate = (successful_updates / total * 100) if total > 0 else 0
        self.logger.info(f"AACT fetch completed:")
        self.logger.info(f"  Successful updates: {successful_updates}")
        self.logger.info(f"  Failed updates: {failed_updates}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")

    def fetch_mesh_terms(self):
        """Fetch Mesh Terms from AACT database and store them in local database."""
        self.logger.info("Starting to fetch Mesh Terms from AACT database.")

        try:
            mesh_terms = self.db_manager.get_all_mesh_terms_from_aact()
            if not mesh_terms:
                self.logger.warning("No Mesh Terms found in AACT database.")
                return
            
            inserted_count = self.db_manager.insert_mesh_terms(mesh_terms)
            self.logger.info(f"Inserted {inserted_count} Mesh Terms into local database.")
        except Exception as e:
            self.logger.error(f"Error fetching Mesh Terms: {e}")
            self.logger.debug("Exception details:", exc_info=True)


    def aact_runner(self):
        """Runner method to fetch trials from AACT database."""
        self.logger.info("Starting AACT data fetching process.")
        try:
            self.fetch_trials()
            self.fetch_mesh_terms()
            self.logger.error("AACT data fetching process completed.")
        except Exception as e:
            self.logger.error(f"AACT data fetching process failed: {e}")
            raise
        finally:
            self.close_connections()

    def close_connections(self):
        """Close database connections."""
        self.db_manager.close_all()
        self.logger.info("Closed all database connections.")








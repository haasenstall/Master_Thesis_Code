# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import requests as req
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

from config import REQUEST_SETTINGS, PLATTFORM_IDS, PROJECT_SETTINGS,DATA_PATHS
from scrapers.base import BaseScraper
from model.database_models import Request, RequestTrialLink
from utils.crossref import get_references, _extract_doi_from_url, _clean_doi_from_text, _clean_doi, _debug_row_content

class CSDRScraper(BaseScraper):
    def __init__(self, platform_id: int, db_manager):
        super().__init__(platform_id, db_manager)
        self.session = req.Session()
        self.session.headers.update(REQUEST_SETTINGS['headers'])
        self.max_workers = PROJECT_SETTINGS.get('max_workers', 5)
        self.retry_attempts = PROJECT_SETTINGS.get('max_retries', 3)
        self.retry_delay = PROJECT_SETTINGS.get('retry_delay', 1)
        self.timeout = PROJECT_SETTINGS.get('timeout', 30)

    def nct_fetch(self):
        """Fetch NCT IDs from xls file downloaded from the CSDR website."""
        file_name = "All-Sponsor-Studies.xls"
        path = DATA_PATHS['data_file'] + file_name

        try:
            df = pd.read_excel(path, engine='xlrd')
            
            nct_ids = []
            
            # Column Name for NCT IDs = Trial Registry Identification Number(s)
            if 'Trial Registry Identification Number(s)' in df.columns:
                nct_column = df['Trial Registry Identification Number(s)']
                for entry in nct_column.dropna():
                    ids = re.findall(r'NCT\d{8}', str(entry))
                    nct_ids.extend(ids)
            else:
                self.logger.error(f"Expected column 'Trial Registry Identification Number(s)' not found in {file_name}")
            
            self.logger.info(f"Fetched {len(nct_ids)} NCT IDs from {file_name}")

            if nct_ids:
                self.db.insert_nct_ids(nct_ids)
                self.db.link_nct_to_platform(nct_ids, platform_name="csdr")  # Use lowercase 'csdr'
            else:
                self.logger.warning("No NCT IDs found in the Excel file")

        except Exception as e:
            self.logger.error(f"Error reading NCT IDs from {file_name}: {e}")
            
            try:
                df = pd.read_excel(path, engine='xlrd')
                self.logger.debug(f"Excel file columns: {df.columns.tolist()}")
                self.logger.debug(f"Excel file shape: {df.shape}")
                self.logger.debug(f"First few rows:\n{df.head()}")
            except Exception as e:
                self.logger.error(f"Error reading Excel file structure: {e}")

    def fix_csv_encoding(self):
        """One-time function to fix the CSV encoding"""
        file_path = DATA_PATHS['data_file'] + "Agreed-Proposals.csv"
        
        # Read with windows-1252 and save as UTF-8
        try:
            df = pd.read_csv(file_path, encoding='windows-1252')
            df.to_csv(file_path.replace('.csv', '_utf8.csv'), encoding='utf-8', index=False)
            self.logger.info("CSV converted to UTF-8 successfully")
        except Exception as e:
            self.logger.error(f"Error converting CSV: {e}")

    def scrape_requests(self) -> List[Dict]:
        """Fetch request data from CSDR via csv file downloaded from website."""
        requests = []
        
        self.fix_csv_encoding()

        file_name = "Agreed-Proposals_utf8.csv"
        path = DATA_PATHS['data_file'] + file_name

        try:
            data = pd.read_csv(path)
            self.logger.info(f"Fetched {len(data)} requests from {file_name}")

            for _, row in data.iterrows():
                request_id = str(row.get('Proposal number', '')).strip()
                title = str(row.get('Title', '')).strip()
                institution = str(row.get('Institution', '')).strip() #self.extract_first_institution(str(row.get('Institution', '')))
                investigator = str(row.get('Lead researcher', '')).strip()
                detail_url = row.get('URL Path', '').strip()

                requests.append({
                    'request_id': request_id,
                    'platform_id': self.platform_id,
                    'title': title,
                    'institution': institution,
                    'investigator': investigator,
                    'detail_url': detail_url
                })

        except Exception as e:
            self.logger.error(f"Error reading requests from {file_name}: {e}")

        pbar = tqdm(total=len(requests), desc="Fetching request details", unit="requests")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_request = {executor.submit(self.get_request_details, req): req for req in requests}
            for future in as_completed(future_to_request):
                req_data = future_to_request[future]
                try:
                    updated_request = future.result()
                    if updated_request:
                        index = requests.index(req_data)
                        requests[index] = updated_request
                        self.logger.debug(f"Updated request {req_data.get('request_id')} with additional details")
                    pbar.update(1)
                except Exception as e:
                    self.logger.error(f"Error fetching details for request {req_data.get('request_id')}: {e}")
                    pbar.update(1)  # Still update progress even on error
        pbar.close()
        self.db.insert_request_list(requests)
        return requests  # Add return statement
    
    def extract_first_institution(self, institution_text):
        """Extract only the first institution from a potentially multi-institution string."""
        if not institution_text:
            return ""
        
        # Clean the text
        institution_text = str(institution_text).strip()
        
        # Split by common delimiters that separate multiple institutions
        delimiters = [
            '\n',           # Line breaks
            '\r\n',         # Windows line breaks
            ') 2)',         # Numbered institutions like "1) Institution A 2) Institution B"
            ') and ',       # "Institution A) and Institution B"
            '; ',           # Semicolon separation
            ' & ',          # Ampersand separation
            ' and ',        # "and" separation
            '\t',           # Tab separation
        ]
        
        # Find the first delimiter that appears in the text
        first_split_pos = len(institution_text)
        used_delimiter = None
        
        for delimiter in delimiters:
            pos = institution_text.find(delimiter)
            if pos != -1 and pos < first_split_pos:
                first_split_pos = pos
                used_delimiter = delimiter
        
        # Extract first institution
        if used_delimiter:
            first_institution = institution_text[:first_split_pos].strip()
            first_institution = re.sub(r'^\d+\)\s*', '', first_institution).strip()
        else:
            first_institution = institution_text
        
        # Additional cleanup - remove extra whitespace and trailing punctuation
        first_institution = re.sub(r'\s+', ' ', first_institution).strip()
        first_institution = first_institution.rstrip('.,;')
        
        return first_institution

    def get_request_details(self, req):
        """Fetch additional details for each request, including NCT IDs and published date."""

        response = self.session.get(req['detail_url'], timeout=self.timeout)

        if response.status_code != 200:
            self.logger.error(f"Failed to fetch details for request {req.get('request_id')}: Status code {response.status_code}")
            return req
        
        soup = BeautifulSoup(response.content, 'html.parser')
        nct_ids = []
        study_ids = []

        # Extract NCT IDs and study identifiers
        posting_divs = soup.find_all('div', class_='posting')
        for div in posting_divs:
            # Get the link text (study identifier like GSK-SFCB3024)
            links = div.find_all('a', href=True)
            for link in links:
                study_id = link.get_text().strip()
                study_ids.append(study_id)
            
        req['number_of_trials_requested'] = len(study_ids)
        for id in study_ids:
            if id.startswith('NCT'):
                nct_ids.append(id)
            else:
                continue

        # link id to request
        self.db.insert_nct_request_ids(req.get('request_id'), nct_ids)

        # Extract published date
        published_date = None
        date_element = soup.find('div', id='MainContentPlaceHolder_PostingForm_PROPOSAL_SUMMARY_DATE_DATA_SHARING')

        date_formats = [
                '%d %B %Y',      # 27 September 2019
                '%d %b %Y',      # 27 Sep 2019
                '%B %d, %Y',     # September 27, 2019
                '%b %d, %Y',     # Sep 27, 2019
                '%Y-%m-%d',      # 2019-09-27
                '%d/%m/%Y',      # 27/09/2019
                '%m/%d/%Y'       # 09/27/2019
            ]
            
        if date_element:
            date_text = date_element.get_text().strip()
            try:
                # parse date with different possible formats
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_text, fmt)
                        break
                    except ValueError:
                        continue
                # Convert to desired format (ISO format YYYY-MM-DD)
                published_date = parsed_date.strftime('%Y-%m-%d')
                req['date_of_request'] = published_date
                self.logger.debug(f"Found published date: {published_date} for request {req.get('request_id')}")
            except ValueError as e:
                self.logger.warning(f"Could not parse date '{date_text}' for request {req.get('request_id')}: {e}")
        else:
            self.logger.warning(f"No published date found for request {req.get('request_id')}")     
        
        return req
    
    def fetch_public_disclosure(self):
        """Fetch Public Disclosure from CSDR website with granular DOI extraction"""
        path = REQUEST_SETTINGS['csdr']['public_disclosure_url']
        response = self.session.get(path, timeout=self.timeout)

        if response.status_code != 200:
            self.logger.error(f"Failed to fetch public disclosures: Status code {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        rows = soup.find_all('tr')
        pbar = tqdm(total=len(rows), desc="Processing public disclosures", unit="disclosures")
        all_dois = []

        # Enhanced DOI regex patterns
        doi_patterns = [
            r'10\.\d{4,9}/[^\s<>"\']+',  # Standard DOI pattern
            r'doi\.org/10\.\d{4,9}/[^\s<>"\']+',  # DOI with domain
            r'dx\.doi\.org/10\.\d{4,9}/[^\s<>"\']+',  # Old DOI domain
            r'doi:10\.\d{4,9}/[^\s<>"\']+',  # DOI with prefix
        ]

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 5:
                pbar.update(1)
                continue
            
            request_id = cols[0].get_text().strip()
            title = cols[2].get_text().strip()
            
            # Clean title to handle Unicode characters
            if title:
                title = title.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')

            authors = cols[3].get_text(strip=True)
            instituition = cols[4].get_text(strip=True)

            request_dois = set()  # Use set to avoid duplicates
            
            # Method 1: Extract DOIs from all links in the row
            all_links = row.find_all('a', href=True)
            for link in all_links:
                doi_url = link['href'].strip()
                doi = _extract_doi_from_url(doi_url)
                if doi:
                    request_dois.add(doi)
                    self.logger.debug(f"Found DOI from link: {doi} for request {request_id}")
            
            # Method 2: Extract DOIs from all text content in the row
            row_text = row.get_text()
            for pattern in doi_patterns:
                matches = re.findall(pattern, row_text, re.IGNORECASE)
                for match in matches:
                    doi = _clean_doi_from_text(match)
                    if doi:
                        request_dois.add(doi)
                        self.logger.debug(f"Found DOI from text: {doi} for request {request_id}")
            
            # Method 3: Check specific columns that might contain DOIs
            for col_index in range(len(cols)):
                col = cols[col_index]
                
                # Extract from column text
                col_text = col.get_text().strip()
                for pattern in doi_patterns:
                    matches = re.findall(pattern, col_text, re.IGNORECASE)
                    for match in matches:
                        doi = _clean_doi_from_text(match)
                        if doi:
                            request_dois.add(doi)
                            self.logger.debug(f"Found DOI from column {col_index}: {doi} for request {request_id}")
                
                # Extract from column HTML (in case DOIs are in attributes)
                col_html = str(col)
                for pattern in doi_patterns:
                    matches = re.findall(pattern, col_html, re.IGNORECASE)
                    for match in matches:
                        doi = _clean_doi_from_text(match)
                        if doi:
                            request_dois.add(doi)
                            self.logger.debug(f"Found DOI from column HTML {col_index}: {doi} for request {request_id}")

            # Convert set to list and add to all_dois
            request_dois_list = list(request_dois)
            all_dois.extend(request_dois_list)

            # Insert each DOI as a separate public disclosure record
            for doi in request_dois_list:
                try:
                    success = self.db_manager.insert_public_disclosure(request_id, authors, instituition, title, doi)
                    if success:
                        self.logger.debug(f"Successfully inserted public disclosure for request {request_id} with DOI {doi}")
                    else:
                        self.logger.warning(f"Failed to insert public disclosure for request {request_id} with DOI {doi}")
                except Exception as e:
                    self.logger.error(f"Error inserting public disclosure for request {request_id}, DOI {doi}: {e}")
            
            if request_dois_list:
                self.logger.info(f"Processed request {request_id} with {len(request_dois_list)} DOIs: {request_dois_list}")
            else:
                self.logger.debug(f"No DOIs found for request {request_id}")
            
            pbar.update(1)

        pbar.close()
        
        # Remove duplicates from all_dois
        unique_dois = list(set(all_dois))
        self.logger.info(f"Fetched {len(all_dois)} total DOIs ({len(unique_dois)} unique) from public disclosures")

        # Fetch references for all unique DOIs
        if unique_dois:
            self.logger.info("Starting to fetch references for public disclosures")
            get_references(unique_dois)
            self.logger.info("Completed fetching references for public disclosures")
        else:
            self.logger.warning("No DOIs found to fetch references for")
        
        return unique_dois

    def csdr_runner(self):
        self.logger.info("Starting CSDR scraper")
        self.nct_fetch()
        self.scrape_requests()
        #self.fetch_public_disclosure()
        self.logger.info("CSDR scraper finished")


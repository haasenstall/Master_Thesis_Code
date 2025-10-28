# -*- coding: utf-8 -*-
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import requests as req
from bs4 import BeautifulSoup
from tqdm import tqdm
import datetime

from config import REQUEST_SETTINGS, PLATTFORM_IDS, PROJECT_SETTINGS, NCT_ID_APIS
from scrapers.base import BaseScraper
from utils.crossref import get_references, fetch_cited_by


class VivliScraper(BaseScraper):
    def __init__(self, platform_id: int, db_manager):
        super().__init__(platform_id, db_manager)
        self.session = req.Session()
        self.session.headers.update(REQUEST_SETTINGS['headers'])
        self.max_workers = PROJECT_SETTINGS.get('max_workers', 5)
        self.retry_attempts = PROJECT_SETTINGS.get('max_retries', 3)
        self.retry_delay = PROJECT_SETTINGS.get('retry_delay', 1)
        self.timeout = PROJECT_SETTINGS.get('timeout', 30)
        self.platform_id = PLATTFORM_IDS['vivli']

    def scrape_nct_ids(self, skip: int):
        """
        Fetch NCT IDs from vivli api
        Only getting the id - the details are updated with the AACT Database
        """
        nct_ids = []  
        # Update the skip parameter in the params
        params = NCT_ID_APIS['vivli']['params'].copy()
        params['$skip'] = skip

        self.logger.debug(f"Fetching NCT IDs with skip={skip}")
        self.logger.debug(f"Request params: {params}")
        self.logger.debug(f"Request URL: {NCT_ID_APIS['vivli']['url']}")
        
        try:
            response = self.session.get(
                NCT_ID_APIS['vivli']['url'],
                params=params,
                timeout=self.timeout
            )

            self.logger.debug(f"Response Status Code: {response.status_code}")
            self.logger.debug(f"Response URL: {response.url}")

            if response.status_code == 200:
                data = response.json()

                studies = data.get('value', [])
                self.logger.debug(f"Number of studies fetched: {len(studies)}")

                null_count = 0
                valid_count = 0

                for study in studies:
                    nct_id = study.get('nctId')
                    if nct_id:
                        nct_ids.append(nct_id)
                        valid_count += 1
                        self.logger.debug(f"Found NCT ID: {nct_id}")
                    else:
                        null_count += 1
                self.logger.info(f"Fetched {valid_count} valid NCT IDs with skip={skip}")
                self.logger.info(f"Found {null_count} studies without NCT ID")
            else:
                self.logger.error(f"Failed to fetch NCT IDs: {response.status_code}")
                self.logger.debug(f"Response Content: {response.text}")
        except Exception as e:
            self.logger.error(f"Error fetching NCT IDs with skip={skip}: {e}")
            time.sleep(self.retry_delay)
        
        self.logger.info(f"Fetched {len(nct_ids)} NCT IDs with skip={skip}")
        return nct_ids

    def get_max_records(self) -> int:
        """
        searching for the max number of trials on vivli
        -> now hard coded, later dynamic
        """
        return 9000

    def scrape_request_list(self):
        """Fetch row by row of the approved Request list"""
        base_url = REQUEST_SETTINGS['vivli']['request_url']
        headers = REQUEST_SETTINGS["headers"].copy()
        headers.update({
            "Referer": "https://vivli.org/",
            "Cache-Control": "no-cache",
        })
        response = self.session.get(base_url, headers=headers, timeout=self.timeout)
        
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch request list: {response.status_code}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        if not soup:
            self.logger.error("Failed to parse HTML content")
            return
            
        request_entries = soup.find_all('tr')
        
        if not request_entries:
            request_entries = soup.find_all('td')
            if not request_entries:
                self.logger.error("No request entries found")
                return
                
        self.logger.debug(f"Found {len(request_entries)} request entries")

        request_list = []
        pbar = tqdm(total=len(request_entries), desc="Processing requests", unit="requests")
        
        for entry in request_entries:
            try:
                columns = entry.find_all('td')
                if len(columns) < 4:
                    self.logger.warning("Skipping entry with insufficient columns")
                    pbar.update(1)
                    continue
                    
                request_id = columns[0].get_text(strip=True)
                platform_id = self.platform_id
                investigator = columns[1].get_text(strip=True)
                institution = columns[2].get_text(strip=True)
                detail_url = columns[3].find('a')['href']
                title = columns[3].find('a').text.strip()
                url_request_id = request_id.zfill(8)
                detail_api = f"https://prod-api.vivli.org/api/dataRequests/{url_request_id}/metadata"
                

                request_list.append({
                    'request_id': request_id,
                    'platform_id': platform_id,
                    'investigator': investigator,
                    'institution': institution,
                    'detail_url': detail_url,
                    'detail_api': detail_api,
                    'title': title
                })

                self.logger.debug(f"Processed request ID: {request_id}")
                pbar.update(1)

            except Exception as e:
                self.logger.error(f"Error processing request entry: {e}")
                self.logger.debug("Exception details:", exc_info=True)
                pbar.update(1)
                continue

        pbar.close()
        self.logger.info(f"Total requests found: {len(request_list)}")
        
        # FIXED: First insert basic request information
        self.db_manager.insert_request_list(request_list)
        
        # FIXED: Then fetch additional details and NCT IDs separately
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.add_request_details, request): request['request_id'] for request in request_list}
            
            completed = 0
            failed = 0
            
            for future in as_completed(futures):
                request_id = futures[future]
                try:
                    future.result()  # This will process NCT IDs and update request
                    completed += 1
                    self.logger.debug(f"Completed processing request {request_id}")
                except Exception as e:
                    failed += 1
                    self.logger.error(f"Error processing request {request_id}: {e}")
        
        self.logger.info(f"Request processing completed: {completed} successful, {failed} failed")

    def add_request_details(self, request):
        """Fetch additional details for a single request, including NCT IDs and published date"""
        detail_url = request.get('detail_url')
        detail_api = request.get('detail_api')
        request_id = request.get('request_id')
        
        if not detail_url:
            self.logger.warning(f"No detail URL for request {request_id}")
            return request
        
        def _parse_date_iso(s: str) -> datetime.datetime | None:
            if not s:
                return None
            try:
                # Handle 'Z' suffix
                return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
            except ValueError:
                return None
        
        def _clean_nct_id(nct_id: str) -> str:
            """Clean NCT ID from Unicode artifacts"""
            if not nct_id:
                return ""
            
            # Remove common Unicode artifacts
            cleaned = nct_id.replace('\ufeff', '')  # Byte Order Mark
            cleaned = cleaned.replace('\u200b', '')  # Zero Width Space
            cleaned = cleaned.replace('\u200c', '')  # Zero Width Non-Joiner
            cleaned = cleaned.replace('\u200d', '')  # Zero Width Joiner
            cleaned = cleaned.replace('\xa0', ' ')   # Non-breaking space
            
            # Remove any other invisible characters and normalize whitespace
            import unicodedata
            cleaned = ''.join(char for char in cleaned if unicodedata.category(char) != 'Cf')
            cleaned = ' '.join(cleaned.split())  # Normalize whitespace
            
            return cleaned.strip().upper()
        
        try:
            response = self.session.get(detail_url, timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch request details for {detail_url}: {response.status_code}")
                return request

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # [Your existing date extraction code remains the same...]
            published_date = None
            json_ld_script = soup.find('script', type='application/ld+json')
            if json_ld_script and json_ld_script.string:
                try:
                    data = json.loads(json_ld_script.string)
                    # data might be a dict or a list
                    candidates = []
                    if isinstance(data, dict):
                        if '@graph' in data and isinstance(data['@graph'], list):
                            candidates = data['@graph']
                        else:
                            candidates = [data]
                    elif isinstance(data, list):
                        candidates = data

                    for item in candidates:
                        if isinstance(item, dict) and 'datePublished' in item:
                            published_date = _parse_date_iso(item.get('datePublished'))
                            if published_date:
                                self.logger.debug(f"datePublished (JSON-LD): {published_date}")
                                break
                except Exception as e:
                    self.logger.warning(f"JSON-LD parse failed: {e}")

            # Fallback to meta tags/time tags
            if not published_date:
                meta_patterns = [
                    {'property': 'article:published_time'},
                    {'name': 'article:published_time'},
                    {'property': 'datePublished'},
                    {'name': 'datePublished'},
                    {'name': 'publication_date'},
                    {'property': 'og:article:published_time'}
                ]
                for pattern in meta_patterns:
                    meta_tag = soup.find('meta', pattern)
                    if meta_tag and meta_tag.get('content'):
                        dt = _parse_date_iso(meta_tag.get('content'))
                        if dt:
                            published_date = dt
                            self.logger.debug(f"datePublished (meta): {published_date} via {pattern}")
                            break

                if not published_date:
                    # time tags sometimes have datetime attr
                    time_tag = soup.find('time', datetime=True)
                    if time_tag:
                        dt = _parse_date_iso(time_tag['datetime'])
                        if dt:
                            published_date = dt
                            self.logger.debug(f"datePublished (time tag): {published_date}")

            # FIXED: Extract NCT IDs with proper cleaning
            nct_ids = []
            nct_pattern = re.compile(r'NCT\d{8}', re.IGNORECASE)
            
            try:
                response = self.session.get(detail_api, timeout=self.timeout)
                if response.status_code != 200:
                    self.logger.error(f"Failed to fetch detail API: {response.status_code}")
                    return request
                    
                data = response.json()
                requested_studies = data.get('requestedStudies', [])
                
                self.logger.debug(f"Found {len(requested_studies)} studies in API for request {request_id}")
                
                for study in requested_studies:
                    raw_nct_id = study.get('nctId')
                    if raw_nct_id:
                        # FIXED: Clean the NCT ID before processing
                        clean_nct_id = _clean_nct_id(raw_nct_id)
                        
                        if clean_nct_id and nct_pattern.match(clean_nct_id):
                            if clean_nct_id not in nct_ids:
                                nct_ids.append(clean_nct_id)
                                # Safe logging - no Unicode artifacts
                                self.logger.debug(f"Found clean NCT ID: {clean_nct_id}")
                        else:
                            self.logger.warning(f"Invalid NCT ID format after cleaning: '{clean_nct_id}' (original: '{raw_nct_id}')")
                
            except Exception as api_error:
                self.logger.error(f"Error fetching from detail API for {request_id}: {api_error}")
                # Fallback to HTML scraping if API fails
                try:
                    full_text = soup.get_text()
                    # Clean the full text too
                    cleaned_text = _clean_nct_id(full_text)
                    html_nct_matches = nct_pattern.findall(cleaned_text)
                    
                    for nct_id in html_nct_matches:
                        clean_nct_id = _clean_nct_id(nct_id).upper()
                        if clean_nct_id and clean_nct_id not in nct_ids:
                            nct_ids.append(clean_nct_id)
                            self.logger.debug(f"Found NCT ID from HTML: {clean_nct_id}")
                            
                    self.logger.info(f"Fallback HTML scraping found {len(html_nct_matches)} NCT IDs for {request_id}")
                except Exception as html_error:
                    self.logger.error(f"HTML fallback failed for {request_id}: {html_error}")

            # Remove duplicates and ensure all are cleaned
            unique_nct_ids = []
            seen = set()
            for nct_id in nct_ids:
                clean_id = _clean_nct_id(nct_id)
                if clean_id and clean_id not in seen:
                    unique_nct_ids.append(clean_id)
                    seen.add(clean_id)

            # Store results
            request['number_of_trials_requested'] = len(unique_nct_ids)
            if published_date:
                request['date_of_request'] = published_date
                self.logger.info(f"Found published date for request {request_id}: {published_date}")
            else:
                self.logger.warning(f"No published date found for request {request_id}")

            # Log final results safely
            self.logger.info(f"Request {request_id}: Found {len(unique_nct_ids)} clean NCT IDs")
            if unique_nct_ids:
                # Log NCT IDs in a safe way
                nct_list_str = ', '.join(unique_nct_ids[:5])  # Show first 5
                if len(unique_nct_ids) > 5:
                    nct_list_str += f" (and {len(unique_nct_ids) - 5} more)"
                self.logger.info(f"NCT IDs for {request_id}: {nct_list_str}")

            # Persist to database
            if unique_nct_ids or published_date:
                try:
                    self.db_manager.insert_request(request)
                    if unique_nct_ids:
                        self.db_manager.insert_nct_request_ids(request_id, unique_nct_ids)
                        self.logger.info(f"Successfully linked {len(unique_nct_ids)} NCT IDs to request {request_id}")
                except Exception as db_error:
                    self.logger.error(f"Database error for {request_id}: {db_error}")

            self.logger.debug(f"Completed processing request {request_id}")

        except Exception as e:
            self.logger.error(f"Error processing request details for {request_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        return request

    def parallel_nct_fetch(self):
        """Fetch NCT IDs parallel for faster processing"""
        max_records = self.get_max_records()
        nct_ids = []
        
        # Calculate total number of batches for progress bar
        total_batches = (max_records + 14) // 15  # Round up division
        pbar = tqdm(total=total_batches, desc="Fetching NCT IDs", unit="batches")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for skip in range(0, max_records, 15):
                futures.append(executor.submit(self.scrape_nct_ids, skip))

            for future in as_completed(futures):
                try:
                    batch_nct_ids = future.result()
                    nct_ids.extend(batch_nct_ids)
                    self.logger.debug(f"Added {len(batch_nct_ids)} NCT IDs from batch")
                    pbar.update(1) 
                except Exception as e:
                    self.logger.error(f"Error fetching NCT IDs: {e}")
                    pbar.update(1)  

        pbar.close()

        seen = set()
        unique_nct_ids = []
        for nct_id in nct_ids:
            if nct_id not in seen:
                seen.add(nct_id)
                unique_nct_ids.append(nct_id)

        self.logger.info(f"Total NCT IDs found: {len(unique_nct_ids)}")
        self.db_manager.insert_nct_ids(unique_nct_ids)
        self.db_manager.link_nct_to_platform(unique_nct_ids, platform_name="vivli")

    def fetch_public_disclosure(self):
        """Fetch public disclosure information from vivli"""
        path = REQUEST_SETTINGS['vivli']['public_disclosure_url']

        response = self.session.get(path, timeout=self.timeout)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch public disclosures: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.find_all('tr')
        if not rows:
            self.logger.error("No public disclosure entries found")
            return
        list_of_dois = []

        pbar = tqdm(total=len(rows), desc="Processing public disclosures", unit="entries")

        for row in rows:
            try:
                columns = row.find_all('td')
                if len(columns) < 4:  # Changed from 3 to 4 since you're accessing columns[3]
                    self.logger.warning("Skipping entry with insufficient columns")
                    continue
                
                request_id = columns[0].get_text(strip=True)
                authors = columns[1].get_text(strip=True)
                instituition = columns[2].get_text(strip=True)
                title = columns[3].find('a').text.strip() if columns[3].find('a') else columns[3].get_text(strip=True)
                doi_link = columns[4].find('a')['href'] if columns[4].find('a') else None

                if doi_link.startswith('https://doi.org/'):
                    doi = doi_link.replace('https://doi.org/', '').strip()
                else:
                    doi = doi_link.strip() if doi_link else None

                self.logger.debug(f"Processed disclosure for request ID: {request_id}")

                self.db_manager.insert_public_disclosure(request_id, authors, instituition, title, doi)

                self.logger.info(f"Inserted public disclosure for request ID: {request_id}")

                list_of_dois.append(doi)
                pbar.update(1)

            except Exception as e:
                self.logger.error(f"Error processing disclosure entry: {e}")
                self.logger.debug("Exception details:", exc_info=True)
                continue
        pbar.close()
        self.logger.info(f"Total public disclosures processed: {len(list_of_dois)}")
        self.logger.info("Starting to fetch references for public disclosures")
        get_references(list_of_dois)
        self.logger.info("Completed fetching references for public disclosures")
        return 

    def vivli_runner(self):
        self.logger.info("Starting Vivli Scraper")
        self.parallel_nct_fetch()
        self.scrape_request_list()
        #self.fetch_public_disclosure()
        self.logger.info("Vivli Scraper finished")



# -*- coding: utf-8 -*-
import re
import os
import logging
import requests as req
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm
from PyPDF2 import PdfReader
from datetime import datetime

from config import REQUEST_SETTINGS, PLATTFORM_IDS, PROJECT_SETTINGS
from scrapers.base import BaseScraper
from model.database_models import Request, RequestTrialLink
from utils.crossref import get_references, _extract_doi_from_url, _clean_doi_from_text, _clean_doi, _debug_row_content


class YodaScraper(BaseScraper):
    def __init__(self, platform_id: int, db_manager):
        super().__init__(platform_id, db_manager)
        self.session = req.Session()
        self.session.headers.update(REQUEST_SETTINGS['headers'])
        self.max_workers = PROJECT_SETTINGS.get('max_workers', 3) 
        self.retry_attempts = PROJECT_SETTINGS.get('max_retries', 5)
        self.retry_delay = PROJECT_SETTINGS.get('retry_delay', 5)
        self.timeout = PROJECT_SETTINGS.get('timeout', 30)
        self.max_pages_trials = 51
        self.max_pages_requests = 52
        self.max_empty_pages = 5

    def scrape_nct_ids(self) -> List[str]:
        """Fetch NCT IDs from Yoda platform by scraping HTML pages."""
        nct_ids = []

        pbar = tqdm(total=self.max_pages_trials, desc="Scraping Yoda NCT IDs", unit="page")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self.fetch_nct_from_page, page): page for page in range(1, self.max_pages_trials + 1)
            }
            empty_page_count = 0

            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    page_nct_ids = future.result()
                    if page_nct_ids:
                        nct_ids.extend(page_nct_ids)
                        empty_page_count = 0  # Reset on finding NCT IDs
                    else:
                        empty_page_count += 1
                        if empty_page_count >= self.max_empty_pages:
                            self.logger.info(f"Stopping early at page {page} due to consecutive empty pages.")
                            break
                except Exception as e:
                    self.logger.error(f"Error processing page {page}: {e}")
                finally:
                   pbar.update(1)
        pbar.close()
        self.logger.info(f"Total NCT IDs fetched: {len(nct_ids)}")
        self.db.insert_nct_ids(nct_ids)
        self.db.link_nct_to_platform(nct_ids, platform_name="yoda")  

    def fetch_nct_from_page(self, page: int) -> List[str]:
        """Fetch NCT IDs from HTML of a single page."""

        nct_ids = []
        
        url = REQUEST_SETTINGS['yoda']['trial_url'].format(page=page)

        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch page {page}, status code: {response.status_code}")
            return nct_ids
        
        soup = BeautifulSoup(response.content, 'html.parser')
         # Find NCT IDs in the trial__nct-id divs
        nct_divs = soup.find_all('div', class_='trial__nct-id')
        for div in nct_divs:
            # Find the link within the div
            link = div.find('a', href=True)
            if link:
                nct_text = link.get_text().strip()
                # Validate it's a proper NCT ID format
                if re.match(r'NCT\d{8}', nct_text):
                    nct_ids.append(nct_text)
                    self.logger.debug(f"Found NCT ID: {nct_text} on page {page}")
                else:
                    self.logger.warning(f"Invalid NCT ID format found: {nct_text} on page {page}")
            else:
                self.logger.warning(f"No link found in NCT ID div on page {page}")
        return nct_ids
    
    def scrape_requests(self):
        """Fetch request details from Yoda platform, including NCT IDs and published dates."""
        request_list = []

        pbar = tqdm(total=self.max_pages_requests, desc="Scraping Yoda Requests", unit="page")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self.fetch_requests_from_page, page): page for page in range(1, self.max_pages_requests + 1)
            }
            empty_page_count = 0

            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    page_requests = future.result()
                    if page_requests:
                        request_list.extend(page_requests)
                        empty_page_count = 0  # Reset on finding requests
                    else:
                        empty_page_count += 1
                        if empty_page_count >= self.max_empty_pages:
                            self.logger.info(f"Stopping early at page {page} due to consecutive empty pages.")
                            break
                except Exception as e:
                    self.logger.error(f"Error processing page {page}: {e}")
                finally:
                   pbar.update(1)
        pbar.close()

        pbar = tqdm(total=len(request_list), desc="Processing Yoda Requests", unit="request")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            """pooling fot NCT ids of each request and published date"""
            future_to_request = {executor.submit(self.get_request_details, req): req for req in request_list}
            for future in as_completed(future_to_request):
                req_data = future_to_request[future]
                try:
                    updated_request = future.result()
                    if updated_request:
                        index = request_list.index(req_data)
                        request_list[index] = updated_request
                        self.logger.debug(f"Updated request {req_data.get('request_id')} with additional details")
                    pbar.update(1)
                except Exception as e:
                    self.logger.error(f"Error processing request {req_data.get('request_id')}: {e}")
                    pbar.update(1)
        
        pbar.close()
        self.logger.info(f"Total requests fetched: {len(request_list)}")
        self.db.insert_request_list(request_list)

    def fetch_requests_from_page(self, page: int) -> List[Dict]:
        """Fetch request entries from a single page."""
        requests = []
        url = REQUEST_SETTINGS['yoda']['request_url'].format(page=page)

        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch requests page {page}, status code: {response.status_code}")
            return requests
        
        soup = BeautifulSoup(response.content, 'html.parser')
        container = soup.find_all('div', class_=lambda x: x and 'request-item flex-container flex-wrap' in x)
        
        if not container:
            self.logger.warning(f"No request items found on page {page}")
            return requests

        for item in container:
            try:
                # Extract all data first
                request_id_elem = item.find('span', class_="yoda-caption dark-caption project-number")
                if not request_id_elem:
                    self.logger.warning(f"No request ID found in item on page {page}")
                    continue
                request_id = request_id_elem.text.strip()
                
                title_elem = item.find('h3', class_="request-item__title color-blue display-block")
                if not title_elem:
                    self.logger.warning(f"No title found for request {request_id}")
                    continue
                title = title_elem.text.strip()
                
                investigator = None
                institution = None
                
                pi_caption = item.find('h4', class_="yoda-caption request-item__caption pi-caption investigator")
                if pi_caption:
                    investigator_elem = pi_caption.find_next('span', class_="yoda-caption dark-caption font-700")
                    if investigator_elem:
                        investigator = investigator_elem.text.strip()
                        institution_elem = investigator_elem.find_next('span', class_="yoda-caption")
                        if institution_elem:
                            institution = institution_elem.text.strip()

                pdf_link = None
                pdf_links = item.find_all('a', class_='link-pdf', href=True)
                for link in pdf_links:
                    link_text = link.get_text().strip()
                    if "Due Diligence Assessment" in link_text:
                        pdf_link = link['href']
                        break

                # Create request dictionary
                request_data = {
                    'request_id': request_id,
                    'title': title,
                    'investigator': investigator,
                    'institution': institution,
                    'pdf_url': pdf_link,
                    'platform_id': self.platform_id
                }
                
                requests.append(request_data)
                self.logger.debug(f"Successfully extracted request {request_id}")

            except Exception as e:
                self.logger.error(f"Error processing request item on page {page}: {e}")
                self.logger.debug("Exception details:", exc_info=True)
        
        return requests
    
    def get_request_details(self, req: Dict) -> Dict:
        """fetch requested NCT ids for each request by parsing the Due Diligence Assessment PDF and the published date from the same PDF"""

        nct_ids = []
        published_date = None

        pdf_url = req.get('pdf_url')
        if not pdf_url:
            self.logger.warning(f"No PDF URL found for request {req.get('request_id')}")
            return req

        try:
            response = self.session.get(pdf_url, timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch PDF for request {req.get('request_id')}, status code: {response.status_code}")
                return req
            
            # Save PDF content to a temporary file
            temp_pdf_path = f"temp_{req.get('request_id')}.pdf"
            with open(temp_pdf_path, 'wb') as f:
                f.write(response.content)

            # Read the PDF and extract text
            reader = PdfReader(temp_pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            # Extract NCT IDs using regex
            nct_ids = re.findall(r'NCT\d{8}', full_text)
            req['nct_ids'] = list(set(nct_ids))  # Remove duplicates
            self.logger.debug(f"Extracted NCT IDs for request {req.get('request_id')}: {req['nct_ids']}")

            req['number_of_trials_requested'] = len(req['nct_ids'])

            # Extract published date - try multiple formats
            published_date = self._extract_date_from_text(full_text, req.get('request_id'))
            if published_date:
                req['date_of_request'] = published_date

        except Exception as e:
            self.logger.error(f"Error processing PDF for request {req.get('request_id')}: {e}")
            self.logger.debug("Exception details:", exc_info=True)
        finally:
            # Clean up temporary PDF file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                self.logger.debug(f"Removed temporary PDF file: {temp_pdf_path}")

        # Insert request into database
        success = self.db.insert_request(req)
        if success:
            self.logger.debug(f"Request {req.get('request_id')} inserted into database")
        else:
            self.logger.error(f"Failed to insert request {req.get('request_id')}")
        
        # Insert NCT-request links
        if nct_ids:
            success = self.db.insert_nct_request_ids(req.get('request_id'), nct_ids)
            if success:
                self.logger.debug(f"NCT-request links created for {req.get('request_id')}")
            else:
                self.logger.error(f"Failed to create NCT-request links for {req.get('request_id')}")
    
        return req

    def _extract_date_from_text(self, text: str, request_id: str) -> Optional[datetime.date]:
        """Extract date from text using multiple common date formats"""
        
        # Define multiple date patterns
        date_patterns = [
            # Full month names
            (r'(\w+\s+\d{1,2},\s+\d{4})', '%B %d, %Y'),  # January 1, 2020
            (r'(\d{1,2}\s+\w+\s+\d{4})', '%d %B %Y'),    # 1 January 2020
            (r'(\w+\s+\d{1,2}\s+\d{4})', '%B %d %Y'),    # January 1 2020
            
            # Abbreviated month names
            (r'(\w{3}\s+\d{1,2},\s+\d{4})', '%b %d, %Y'), # Jan 1, 2020
            (r'(\d{1,2}\s+\w{3}\s+\d{4})', '%d %b %Y'),   # 1 Jan 2020
            (r'(\w{3}\s+\d{1,2}\s+\d{4})', '%b %d %Y'),   # Jan 1 2020
            
            # Numeric formats
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),     # 01/15/2020
            (r'(\d{1,2}-\d{1,2}-\d{4})', '%m-%d-%Y'),     # 01-15-2020
            (r'(\d{4}/\d{1,2}/\d{1,2})', '%Y/%m/%d'),     # 2020/01/15
            (r'(\d{4}-\d{1,2}-\d{1,2})', '%Y-%m-%d'),     # 2020-01-15
            
            # European formats
            (r'(\d{1,2}\.\d{1,2}\.\d{4})', '%d.%m.%Y'),   # 15.01.2020
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%d/%m/%Y'),     # 15/01/2020 (try European)
            
            # ISO formats
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),         # 2020-01-15
        ]
        
        for pattern, date_format in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    parsed_date = datetime.strptime(match, date_format).date()
                    # Validate reasonable date range (1990-2030)
                    if 1990 <= parsed_date.year <= 2030:
                        self.logger.debug(f"Extracted date for request {request_id}: {parsed_date} (format: {date_format})")
                        return parsed_date
                except ValueError:
                    continue  # Try next pattern
        
        self.logger.warning(f"No valid date found in PDF for request {request_id}")
        return None
    
    def fetch_public_disclosure(self) -> List[str]:
        """Fetch DOIs from Yoda public disclosures page."""
        doi_list = []
        max_pages = self.max_pages_requests

        pbar = tqdm(total=max_pages, desc="Scraping Yoda Public Disclosures", unit="page")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self.get_public_disclosures, page): page for page in range(1, max_pages + 1)
            }
            empty_page_count = 0

            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    results = future.result()
                    if results:
                        doi_list.extend(re['doi'] for re in results)
                        empty_page_count = 0  # Reset on finding DOIs
                    else:
                        empty_page_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing public disclosure page {page}: {e}")
                finally:
                   pbar.update(1)
        pbar.close()

        doi_list = list(set(doi_list))  # Remove duplicates
        self.logger.info(f"Total DOIs fetched from public disclosures: {len(doi_list)}")
        get_references(doi_list)
        self.logger.info("Completed fetching references for public disclosures")
        return

    def get_public_disclosures(self, page: int) -> List[Dict[str, str]]:
        """Fetch DOIs from Yoda public disclosures page."""
        results = []

        url = REQUEST_SETTINGS['yoda']['request_url'].format(page=page)

        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch requests page {page}, status code: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        container = soup.find_all('div', class_=lambda x: x and 'request-item flex-container flex-wrap' in x)
        
        if not container:
            self.logger.warning(f"No request items found on page {page}")
            return None

        for item in container:
            try:
                request_id_elem = item.find('span', class_="yoda-caption dark-caption project-number")
                if not request_id_elem:
                    self.logger.warning(f"No request ID found in item on page {page}")
                    continue
                request_id = request_id_elem.text.strip()

                investigator = None
                institution = None
                
                pi_caption = item.find('h4', class_="yoda-caption request-item__caption pi-caption investigator")
                if pi_caption:
                    investigator_elem = pi_caption.find_next('span', class_="yoda-caption dark-caption font-700")
                    if investigator_elem:
                        investigator = investigator_elem.text.strip()
                        institution_elem = investigator_elem.find_next('span', class_="yoda-caption")
                        if institution_elem:
                            institution = institution_elem.text.strip()

                publication_links = item.find_all('a', href=True)
                for link in publication_links:
                    href = link['href']
                    doi = _extract_doi_from_url(href)
                    if doi:
                        results.append(self.get_publication_details(request_id=request_id, doi=doi, institution=institution, authors=investigator))

                    elif "pubmed" in href:
                        results.append(self.get_publication_details(request_id=request_id, pubmed_link=href, institution=institution, authors=investigator))

                    else:
                        continue
                if not results:
                    self.logger.warning(f"No valid publication links found for request {request_id} on page {page}")
            except Exception as e:
                self.logger.error(f"Error processing request item on page {page}: {e}")
                self.logger.debug("Exception details:", exc_info=True)


        return results

    def get_publication_details(self, request_id: str, doi: Optional[str] = None, pubmed_link: Optional[str] = None, institution: Optional[str] = None, authors: Optional[str] = None) -> Dict[str, str]:
        """Fetch publication details using DOI or PubMed link."""

        if doi:
            self.logger.debug(f"Found DOI {doi} for request {request_id}")
            path = REQUEST_SETTINGS['crossref']['base_url'].format(doi=doi)
            response = self.session.get(path, timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch CrossRef data for DOI {doi}, status code: {response.status_code}")
                return None
            
            data = response.json()
            title = data.get('title', [None])[0] if data.get('title', [None])[0] else None
            self.db_manager.insert_public_disclosure(request_id=request_id, title=title, doi=doi)
            return {"request_id": request_id, "title": title, "doi": doi}

        if pubmed_link:
            # get PMID from link
            pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', pubmed_link)
            if not pmid_match:
                self.logger.warning(f"Invalid PubMed link format: {pubmed_link} for request {request_id}")
                return None
            pmid = pmid_match.group(1)
            self.logger.debug(f"Found PubMed ID {pmid} for request {request_id}")

            path = REQUEST_SETTINGS['crossref']['pubmed_url'].format(pmid=pmid)

            response = self.session.get(path, timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch CrossRef data for PubMed ID {pmid}, status code: {response.status_code}")
                return None
            data = response.json()
            title = data.get('title', [None])[0] if data.get('title', [None])[0] else None
            doi = data.get('doi', None)
            self.db_manager.insert_public_disclosure(request_id=request_id, title=title, doi=doi, institution=institution, authors=authors)
            return {"request_id": request_id, "title": title, "doi": doi}

    def yoda_runner(self):
        """Main runner to execute both NCT ID scraping and request scraping."""
        try:
            self.logger.info("Starting Yoda scraper")
            
            # Ensure database is properly set up
            self.logger.info("Checking database schema...")
            self.db_manager.check_database_schema()
            
            # Test database connection
            if hasattr(self.db, 'institution_insert'):
                self.logger.debug("Database methods available")
            else:
                self.logger.error("Database methods not available")
                return
            
            # Continue with scraping
            self.scrape_nct_ids()
            self.scrape_requests()
            #self.fetch_public_disclosure()
            
            self.logger.info("Yoda scraper finished successfully")
            
        except Exception as e:
            self.logger.error(f"Error in yoda_runner: {e}")
            self.logger.debug("Exception details:", exc_info=True)
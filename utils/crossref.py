import os
import requests as req
import re
import logging
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Optional, Dict
from config import REQUEST_SETTINGS, PROJECT_SETTINGS, LOGGING_CONFIG, DATA_PATHS
from db.sqlalchemy_connector import DatabaseManager
from db.sqlalchemy_connector import SQLAlchemyConnector
from model.database_models import PublicDisclosure, FocalReference, FutureCitation

def fetch_references(doi: str) -> Optional[List[str]]:
    """Fetch references for a given DOI using the CrossRef API."""
    logger = logging.getLogger('CrossRefFetcher')
    base_url = REQUEST_SETTINGS['crossref']['base_url'].format(doi=doi)
    headers = REQUEST_SETTINGS['headers']

    references = {
        'origin_doi': doi,
        'referenced_work': [],
        'cited_by_link': None,
        'publication_date': None,
        'authors': []
    }

    ref_links = {}

    try:
        response = req.get(base_url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to fetch references for DOI {doi}: Status code {response.status_code}")
            return None
        data = response.json()
        ref_links['referenced_work'] = data.get('referenced_works', [])
        if ref_links['referenced_work']:
            with ThreadPoolExecutor(max_workers=PROJECT_SETTINGS['max_workers']) as executor:
                future_to_doi = {executor.submit(get_reference_doi, ref): ref for ref in ref_links['referenced_work']}
                for future in as_completed(future_to_doi):
                    ref = future_to_doi[future]
                    try:
                        result = future.result()
                        if result:
                            references['referenced_work'].append(result)
                    except Exception as e:
                        logger.error(f"Error processing reference {ref}: {e}")
        else:
            references['referenced_work'] = [ref.get('DOI') for ref in data.get('reference', []) if 'DOI' in ref]
        references['cited_by_link'] = data.get('cited_by_api_url', None)
        references['publication_date'] = data.get('publication_date', None)
        if data.get('authorships'):
            references['authors'] = [author.get('raw_author_name', '') for author in data.get('authorships', [])]
        else:
            references['authors'] = [author.get('given', '') + ' ' + author.get('family', '') for author in data.get('authors', []) if 'family' in author and 'given' in author]
        return references
    except Exception as e:
        logger.error(f"Error fetching references for DOI {doi}: {e}")
        return None
    
def get_reference_doi(ref: str) -> Optional[str]:
    """Fetch doi from crossref reference entry"""
    logger = logging.getLogger('CrossRefFetcher')
    try:
        # Cut prefix "https://" to add the api prefix "https://api."
        if ref.startswith('https://openalex.org/'):
            work_id = ref.replace('https://openalex.org/', '')
            api_url = f"https://api.openalex.org/works/{work_id}"
        else:
            logger.warning(f"Unexpected reference format: {ref}")
            return None
        
        headers = REQUEST_SETTINGS['headers']
        response = req.get(api_url, headers=headers)

        if response.status_code != 200:
            logger.error(f"Failed to fetch reference data from {api_url}: Status code {response.status_code}")
            return None
        
        data = response.json()
        if 'doi' in data:
            doi_url = data['doi']
            if doi_url.startswith('https://doi.org/'):
                doi = doi_url.replace('https://doi.org/', '')
            else:
                doi = doi_url

            logger.debug(f"Extracted DOI {doi} from reference {ref}")
            return doi
        else:
            logger.warning(f"Unexpected response format from {api_url}: {data}")
            return None
    except Exception as e:
        logger.error(f"Error fetching reference DOI from {ref}: {e}")
        return None

def fetch_cited_by(origin_doi: str, api_link: str) -> Optional[List[str]]:
    """Fetch cited by DOIs by the apilink from crossref"""
    logger = logging.getLogger('CrossRefFetcher')

    openalex_id = api_link.replace('https://api.openalex.org/works?filter=cites:', '')

    try:
        response = req.get(api_link, headers=REQUEST_SETTINGS['headers'])

        if response.status_code != 200:
            logger.error(f"Failed to fetch cited_by data from {api_link}: Status code {response.status_code}")
            return None
        
        data = response.json()
        
        if 'results' in data:
            if 'meta' in data and 'count' in data['meta']:
                count = data['meta']['count']
                logger.debug(f"Found {count} citing works for DOI {origin_doi}")

            if count >= data['meta'].get('per-page', 200):
                logger.warning(f"Number of citing works ({count}) exceeds or equals per-page limit. Consider pagination for DOI {origin_doi}")
                return fetch_all_cited_by(openalex_id, count)
            else:
                cited_by_dois = []
                for item in data['results']:
                    if 'doi' in item:
                        doi_url = item['doi']
                        doi = _extract_doi_from_url(doi_url)
                        if doi:
                            cited_by_dois.append(doi)
                return cited_by_dois
            
        else:   
            logger.warning(f"Unexpected response format from {api_link}: {data}")
            return None
    except Exception as e:
        logger.error(f"Error fetching cited_by data from {api_link}: {e}")
        return None
    
def fetch_all_cited_by(api_link: str, total_count: int) -> List[str]:
    """Fetch all cited by DOIs handling pagination"""
    logger = logging.getLogger('CrossRefFetcher')
    cited_by_dois = []
    per_page = 200  
    total_pages = (total_count // per_page) + (1 if total_count % per_page > 0 else 0)
   

    with ThreadPoolExecutor(max_workers=PROJECT_SETTINGS['max_workers']) as executor:
        future_to_page = {}
        for page in range(1, total_pages + 1):
            path = REQUEST_SETTINGS['crossref'].get('cited_by_url').format(page=page, openalex_id=api_link, per_page=per_page)
            future = executor.submit(req.get, path, headers=REQUEST_SETTINGS['headers'])
            future_to_page[future] = page

        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                response = future.result()
                if response.status_code != 200:
                    logger.error(f"Failed to fetch page {page} from {api_link}: Status code {response.status_code}")
                    continue
                
                data = response.json()
                if 'results' in data:
                    for item in data['results']:
                        if 'doi' in item:
                            doi_url = item['doi']
                            doi = _extract_doi_from_url(doi_url)
                            if doi:
                                cited_by_dois.append(doi)
                else:
                    logger.warning(f"Unexpected response format on page {page} from {api_link}: {data}")
            except Exception as e:
                logger.error(f"Error processing page {page} from {api_link}: {e}")

    return cited_by_dois

def get_references(origin_doi_list: list) -> None:  # FIXED: Removed 'self' parameter
    """Fetch the references for a given doi"""
    logger = logging.getLogger('CrossRefFetcher')
    logger.info(f"Fetching references for DOIs: {origin_doi_list}")
    
    db_manager = DatabaseManager()
    db_manager.connect_local()

    if not origin_doi_list:
        logger.warning(f"No DOI links provided for origin DOIs")
        return
    
    total_tasks = len(origin_doi_list) * 2
    pbar = tqdm.tqdm(total=total_tasks, desc="Fetching references and citations", unit="DOIs")  # Fixed tqdm import
    
    max_workers = PROJECT_SETTINGS.get('max_workers', 5)  # FIXED: Get max_workers from config
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all reference fetching tasks first
        ref_futures = {
            executor.submit(fetch_references, doi): ('references', doi) 
            for doi in origin_doi_list
        }
        
        # Collect cited_by_links from the first pass
        cited_by_tasks = []
        
        # Process reference results and collect cited_by_links
        for future in as_completed(ref_futures):
            task_type, origin_doi = ref_futures[future]
            try:
                references = future.result()
                if references:
                    # Insert references
                    if 'referenced_work' in references:
                        for ref in references['referenced_work']:
                            db_manager.insert_focal_reference(origin_doi, ref)
                        logger.debug(f"Fetched {len(references['referenced_work'])} references for DOI {origin_doi}")

                    # Schedule cited_by fetching
                    if 'cited_by_link' in references:
                        cited_by_future = executor.submit(fetch_cited_by, origin_doi, references['cited_by_link'])
                        cited_by_tasks.append((cited_by_future, origin_doi))

                    if 'authors' in references and 'publication_date' in references:
                        db_manager.update_public_disclosure_details(
                            origin_doi,
                            references['authors'],
                            references['publication_date']
                        )
                
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error fetching references for DOI {origin_doi}: {e}")
                pbar.update(1)

        # Process cited_by results
        for cited_future, origin_doi in cited_by_tasks:
            try:
                cited_by = cited_future.result()
                if cited_by:
                    # Handle different return formats
                    if isinstance(cited_by, dict) and 'cited_by' in cited_by:
                        citations = cited_by['cited_by']
                    elif isinstance(cited_by, list):
                        citations = cited_by
                    else:
                        pbar.update(1)
                        continue
                    
                    # Insert each citation
                    for cb in citations:
                        db_manager.insert_cited_by(origin_doi, cb)

                    logger.debug(f"Fetched {len(citations)} cited by citations for DOI {origin_doi}")

                pbar.update(1)

            except Exception as e:
                logger.error(f"Error fetching cited by citations for DOI {origin_doi}: {e}")
                pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    db_manager.close_all()  # FIXED: Close database connections
    logger.info(f"Completed fetching references and citations for {len(origin_doi_list)} DOIs")
    return

def _extract_doi_from_url(doi_url: str) -> str:
    """Extract DOI from various URL formats"""
    if not doi_url:
        return None
    
    doi_url = doi_url.strip()
    doi = None
    
    # Handle different DOI URL formats
    if doi_url.startswith('https://doi.org/'):
        doi = doi_url.replace('https://doi.org/', '')
    elif doi_url.startswith('http://dx.doi.org/'):
        doi = doi_url.replace('http://dx.doi.org/', '')
    elif doi_url.startswith('https://dx.doi.org/'):
        doi = doi_url.replace('https://dx.doi.org/', '')
    elif doi_url.startswith('http://doi.org/'):
        doi = doi_url.replace('http://doi.org/', '')
    elif doi_url.startswith('https://www.doi.org/'):
        doi = doi_url.replace('https://www.doi.org/', '')
    elif doi_url.startswith('http://www.doi.org/'):
        doi = doi_url.replace('http://www.doi.org/', '')
    elif re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi_url, re.IGNORECASE):
        # Direct DOI format
        doi = doi_url
    else:
        # Check if URL contains a DOI pattern
        doi_match = re.search(r'10\.\d{4,9}/[^\s<>"\'&?]+', doi_url, re.IGNORECASE)
        if doi_match:
            doi = doi_match.group()
    
    return _clean_doi(doi) if doi else None

def _clean_doi_from_text(text_match: str) -> str:
    """Clean DOI extracted from text content"""
    if not text_match:
        return None
    
    # Remove common prefixes
    doi = text_match.replace('doi.org/', '').replace('dx.doi.org/', '').replace('doi:', '')
    
    # Remove trailing punctuation that's not part of the DOI
    doi = re.sub(r'[.,;:)}\]>\s]+$', '', doi)
    
    # Validate it's still a valid DOI format
    if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi, re.IGNORECASE):
        return _clean_doi(doi)
    
    return None

def _clean_doi(doi: str) -> str:
    """Clean and validate DOI"""
    if not doi:
        return None
    
    # Remove extra whitespace and trailing periods
    doi = doi.strip().rstrip('.')
    
    # Remove any HTML entities or special characters that got through
    doi = doi.replace('&nbsp;', '').replace('\xa0', '')
    
    # Validate final DOI format
    if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi, re.IGNORECASE):
        return doi
    
    return None

def _debug_row_content(row, request_id: str):
    """Debug function to log all content in a row"""
    logger = logging.getLogger('CrossRefFetcher')
    logger.debug(f"=== Debug content for request {request_id} ===")

    cols = row.find_all('td')
    for i, col in enumerate(cols):
        logger.debug(f"Column {i}: {col.get_text().strip()[:100]}")
        links = col.find_all('a', href=True)
        for j, link in enumerate(links):
            logger.debug(f"  Link {j}: {link['href']}")

    logger.debug(f"Full row text: {row.get_text()[:200]}")
    logger.debug("=== End debug ===")


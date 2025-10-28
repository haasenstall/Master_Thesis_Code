"""
scraping the data from vivli, csdr and yoda for focal publications
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import re
import time
import random
import json
from PyPDF2 import PdfReader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
from pathlib import Path

# Add path to the classes module
sys.path.append(str(Path(__file__).parent.parent / "07_classes"))
from classes import trial, trial_list, trial_request, trial_request_list

def get_vivli_data():
    """
    Scrape data from Vivli platform using the trial_request class structure
    """
    url = "https://vivli.org/approved-research-proposals/"
    
    # Create a trial_request_list object
    request_list = trial_request_list()

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from Vivli. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    for row in soup.find_all('tr'):
        # Skip if no columns found (likely header row)
        if not row.find_all('td'):
            continue
            
        # Extract basic information
        request_id = row.find('td', class_='column-1').text.strip() if row.find('td', class_='column-1') else ""
        lead_investigator = row.find('td', class_='column-2').text.strip() if row.find('td', class_='column-2') else ""
        lead_institution = row.find('td', class_='column-3').text.strip() if row.find('td', class_='column-3') else ""
        
        # Get title and link
        title_cell = row.find('td', class_='column-4')
        study_title = ""
        link = ""
        nct_ids = []
        request_year = ""
        
        if title_cell:
            link_tag = title_cell.find('a')
            if link_tag:
                study_title = link_tag.text.strip()
                link = link_tag['href'] if 'href' in link_tag.attrs else ""
                
                # Get NCT IDs and year
                if link:
                    try:
                        nct_ids, year = vivli_nct_grapper(link)
                        request_year = year if year else "Unknown"
                    except Exception as e:
                        print(f"Error getting NCT IDs for {link}: {e}")
                        nct_ids = []
                        request_year = "Unknown"
        
        # Create a trial_request object and add to the list
        if request_id:
            request = trial_request(
                request_id=request_id,
                title=study_title,
                nct_id=nct_ids,
                request_year=request_year,
                lead_investigator=lead_investigator,
                lead_institution=lead_institution,
                country=None  # You can implement country extraction later
            )
            request_list.add_request(request)

    return request_list.requests

def vivli_nct_grapper(link):
    """
    Extract NCT IDs and publication year from the provided Link on Vivli
    """
    url = link
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from Vivli. Status code: {response.status_code}")
    soup = BeautifulSoup(response.content, 'html.parser')

    NCT_IDs = []
    for p in soup.find_all('p'):
        # Find all NCT IDs in the text
        found_ids = re.findall(r'NCT\d{8}', p.text)
        if found_ids:
            NCT_IDs.extend(found_ids)
        
        # Check for span tags - improved version
        span_tags = p.find_all('span')
        if span_tags:  # Only proceed if span tags are found
            for span in span_tags:
                # Directly use regex to find NCT IDs in the span text
                found_ids = re.findall(r'NCT\d{8}', span.text)
                if found_ids:
                    NCT_IDs.extend(found_ids)
                # Handle potential fragmented NCT IDs across spans
                elif 'NCT' in span.text and not re.findall(r'NCT\d{8}', span.text):
                    # The span contains "NCT" but not a complete NCT ID
                    next_span = span.find_next('span')
                    if next_span:
                        # Check if combining current and next span creates a valid NCT ID
                        combined_text = span.text.strip() + next_span.text.strip()
                        combined_ids = re.findall(r'NCT\d{8}', combined_text)
                        if combined_ids:
                            NCT_IDs.extend(combined_ids)

    # Extract year from JSON-LD script
    year = None
    script_tag = soup.find('script', {'type': 'application/ld+json', 'class': 'yoast-schema-graph'})
    if script_tag:
        try:
            # Parse the JSON content
            json_data = json.loads(script_tag.string)
            
            # Try to find datePublished in the JSON data
            # Look through the graph items for WebPage type
            for item in json_data.get('@graph', []):
                if item.get('@type') == 'WebPage':
                    if 'datePublished' in item:
                        # Extract year from ISO date format (YYYY-MM-DD)
                        date_published = item['datePublished']
                        year = date_published.split('-')[0]  # Get year part
                        break
                    elif 'dateModified' in item:
                        # Fallback to modified date if published isn't available
                        date_modified = item['dateModified']
                        year = date_modified.split('-')[0]  # Get year part
                        break
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error extracting year from JSON-LD: {e}")
    
    # If no year found in JSON-LD, try to find it in other metadata
    if not year:
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date and 'content' in meta_date.attrs:
            year = meta_date['content'].split('-')[0]
    
    return NCT_IDs, year

def get_csdr_data():
    """
    Scrape data from CSDR platform using the trial_request class structure
    """
    url = "https://www.clinicalstudydatarequest.com/Metrics/Agreed-Proposals.aspx"
    
    # Create a trial_request_list object
    request_list = trial_request_list()
    
    # Load existing data if available
    load_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "csdr_data.json")
    if os.path.exists(load_data_path):
        request_list.load_from_file(load_data_path)
        existing_ids = {request['request_id'] for request in request_list.requests}
    else:
        existing_ids = set()
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from CSDR. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'proposals-table'})
    if not table:
        raise Exception("Failed to find the proposals table on CSDR page.")
    
    tbody = table.find('tbody')
    if not tbody:
        raise Exception("Failed to find the table body in the proposals table on CSDR page.")
    
    # Iterate through each row in the table
    for row in tbody.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 6:
            continue
        
        request_id = cells[0].text.strip()
        
        # Skip if already processed
        if request_id in existing_ids:
            print(f"Skipping existing request ID: {request_id}")
            continue
        
        sponsor = cells[1].text.strip()
        study_title = cells[2].text.strip()
        lead_institution = cells[3].text.strip()
        lead_researcher = cells[4].text.strip()
        
        # Get link
        link_tag = cells[5].find('a', href=True)
        link = link_tag['href'] if link_tag else ""
        if link.startswith('../'):
            link = 'https://www.clinicalstudydatarequest.com' + link[2:]
        
        # Get NCT IDs and year
        study_ids = []
        year = "Unknown"
        if link:
            try:
                study_ids, year = csdr_ID_grapper(link)
            except Exception as e:
                print(f"Error getting NCT IDs for {link}: {e}")
        
        # Create a trial_request object
        request = trial_request(
            request_id=request_id,
            title=study_title,
            nct_id=study_ids,
            request_year=year,
            lead_investigator=lead_researcher,
            lead_institution=lead_institution,
            country=None  # You can implement country extraction later
        )
        
        # Add to request list
        request_list.add_request(request)
        
        # Save progress every 5 rows
        if len(request_list.requests) % 5 == 0:
            print(f"Progress: {len(request_list.requests)} rows scraped so far.")
            request_list.save_to_file(load_data_path)
    
    return request_list.requests

def csdr_ID_grapper(link):
    """
    Extract the study ID from the provided Link from CSDR
    """
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    
    # Start browser
    driver = webdriver.Chrome(options=options)
    driver.get(link)
    
    # Wait for JS to load (adjust if needed)
    time.sleep(0.5)

    # Get page source and parse it
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    study_ids = []
    
    # Find all elements that contain study IDS
    for posting in soup.find_all('div', {'class': 'posting'}):
        # print(f"Processing posting")
        a_tag = posting.find('a', href=True)
        if a_tag:
            study_id = a_tag.get_text(strip=True)
            study_link = a_tag['href']
        
        # Check if there is a NCT ID
        nct_id = csdr_nct_grapper(study_link)
        if nct_id != []:
            #print(f"NCT ID found: {nct_id}")
            study_id = nct_id
        
        
        if study_id:
            study_ids.append(study_id)
                
    # Extract year from the data sharing date field
    date_div = soup.find('div', {'id': 'MainContentPlaceHolder_PostingForm_PROPOSAL_SUMMARY_DATE_DATA_SHARING'})
    if date_div:
        date_text = date_div.text.strip()
        # Extract year from date text (assuming format like "26 November 2013")
        date_parts = date_text.split()
        try:
            year = date_parts[-1]  # Get the last part which should be the year
            # Validate it's a 4-digit year
            if len(year) == 4 and year.isdigit():
                print(f"Found year: {year}")
            else:
                year = None
        except:
            year = None
    return study_ids, year

def csdr_nct_grapper(link):
    """
    Extract the NCT ID from the provided link
    """
    url = "https://www.clinicalstudydatarequest.com" + link

     # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    
    # Start browser
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    # Wait for JS to load (adjust if needed)
    time.sleep(0.5)

    # Get page source and parse it
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    
    # Find the element that contains the NCT ID
    nct_id = soup.find('div', {'id': 'MainContentPlaceHolder_PostingForm_CLINICAL_TRIAL_ID'})
    if nct_id:
        nct_id_text = nct_id.text.strip()
        #print(f"NCT ID found: {nct_id_text}")
        # Check if the text matches the NCT ID format
        if re.match(r'NCT\d{8}', nct_id_text):
            return nct_id_text
        else:
            # If the NCT ID is not in the expected format, return an empty list
            return []


def get_yoda_data():
    """
    Scrape data from YODA platform using the trial_request class structure
    """
    # Create a trial_request_list object
    request_list = trial_request_list()
    
    for page in range(1, 51):
        url = f"https://yoda.yale.edu/metrics/submitted-requests-to-use-johnson-johnson-data/data-requests-johnson-and-johnson/?_paged={page}"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data from YODA page {page}. Status code: {response.status_code}")
            continue
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Define class patterns for request items
        class_pattern = [
            'request-item flex-container flex-wrap step-approved_pending_dua_signature',
            'request-item flex-container flex-wrap step-concluded',
            'request-item flex-container flex-wrap step-in_progess',
            'request-item flex-container flex-wrap step-ongoing',
            'request-item flex-container flex-wrap step-published',           
            'request-item flex-container flex-wrap step-submitted_for_yoda_project_review',
            'request-item flex-container flex-wrap step-unknown_revoked'
        ]
        
        requests_items = []
        for pattern in class_pattern:
            items = soup.find_all('div', {'class': pattern})
            requests_items.extend(items)
        
        if not requests_items:
            print(f"No request items found on page {page}.")
            continue
        
        for item in requests_items:
            # Extract request ID and year
            request_id = ""
            request_year = ""
            left_cell = item.find('div', {'class': 'left-cell'})
            if left_cell:
                project_number = left_cell.find('span', {'class': 'yoda-caption dark-caption project-number'})
                if project_number:
                    request_id = project_number.text.strip()
                    request_year = request_id.split('-')[0] if '-' in request_id else ""
            
            # Extract other information
            lead_investigator = ""
            lead_institution = ""
            study_title = ""
            pdf_link = ""
            right_cell = item.find('div', {'class': 'right-cell'})
            if right_cell:
                # Lead investigator
                lead = right_cell.find('span', {'class': 'yoda-caption dark-caption font-700'})
                if lead:
                    lead_investigator = lead.text.strip()
                
                # Lead institution
                institute = lead.find_next('span', {'class': 'yoda-caption'})
                if institute and not institute.find_parent('h4'):
                    lead_institution = institute.text.strip()
                
                # Study title
                title = right_cell.find('h3', {'class': 'request-item__title color-blue display-block'})
                if title:
                    study_title = title.text.strip()
                
                # PDF link
                pdf_div = right_cell.find('div', {'class': 'flex-container flex-wrap project-docs-ctas'})
                if pdf_div:
                    pdf_links = pdf_div.find_all('a', href=True)
                    for link in pdf_links:
                        if 'due' in link['href'].lower():
                            pdf_link = link['href']
                            break
            
            # Get NCT IDs from PDF
            nct_ids = []
            if pdf_link:
                try:
                    nct_ids = yoda_nct_grapper(pdf_link)
                except Exception as e:
                    print(f"Error getting NCT IDs for {pdf_link}: {e}")
            
            # Create a trial_request object
            if request_id:
                request = trial_request(
                    request_id=request_id,
                    title=study_title,
                    nct_id=nct_ids,
                    request_year=request_year,
                    lead_investigator=lead_investigator,
                    lead_institution=lead_institution,
                    country=None  # You can implement country extraction later
                )
                request_list.add_request(request)
    
    return request_list.requests

def yoda_nct_grapper(link):
    """
    Function to extract the NCT IDs from the PDF Request Reports
    for multiple requests
    """
    nct_ids = []

    pdf_link = link
    response = requests.get(pdf_link)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch PDF from YODA. Status code: {response.status_code}")
    
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
    
    pdf_reader = PdfReader('temp.pdf')

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            # Find all NCT IDs in the text
            found_ids = re.findall(r'NCT\d{8}', text)
            nct_ids.extend(found_ids)

    # Remove duplicates
    nct_ids = list(set(nct_ids))
    # Clean up the temporary file
    os.remove('temp.pdf')

    return nct_ids

def main():
    """
    Main function to run the scrapers using the trial_request class structure
    """
    # Create a trial_request_list for each source
    vivli_requests = trial_request_list()
    csdr_requests = trial_request_list()
    yoda_requests = trial_request_list()
    
    # Run the scrapers
    print("Scraping Vivli data...")
    vivli_requests.requests = get_vivli_data()
    
    print("Scraping CSDR data...")
    csdr_requests.requests = get_csdr_data()
    
    print("Scraping YODA data...")
    yoda_requests.requests = get_yoda_data()
    
    # Save the data
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    os.makedirs(output_dir, exist_ok=True)
    
    vivli_file_path = os.path.join(output_dir, "vivli_data.json")
    vivli_requests.save_to_file(vivli_file_path)
    
    csdr_file_path = os.path.join(output_dir, "csdr_data.json")
    csdr_requests.save_to_file(csdr_file_path)
    
    yoda_file_path = os.path.join(output_dir, "yoda_data.json")
    yoda_requests.save_to_file(yoda_file_path)
    
    # Generate simple reports
    print(f"\nData Collection Report:")
    print(f"----------------------")
    print(f"Vivli requests: {len(vivli_requests.requests)}")
    print(f"CSDR requests: {len(csdr_requests.requests)}")
    print(f"YODA requests: {len(yoda_requests.requests)}")
    print(f"Total requests: {len(vivli_requests.requests) + len(csdr_requests.requests) + len(yoda_requests.requests)}")
    

if __name__ == "__main__":
    print("Starting the scraping process...")
    main()
    print("Scraping completed.")
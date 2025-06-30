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

def get_vivli_data():
    """
    Scrape data from Vivli platform
    data requested from the Vivli platform

    data structure:
        {
        request_id: str,
        lead_investigator: str,
        lead_institution: str,
        study_title: str,
        link: str,
        stduy_id: str,
        number_of_studies: int
        }
    """
    url = "https://vivli.org/approved-research-proposals/"
    
    data = []  # Initialize as a list instead of a dictionary

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from Vivli. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    for row in soup.find_all('tr'):
        row_data = {}
        
        # request id
        request_id = row.find('td', class_='column-1')
        if request_id:
            row_data['request_id'] = request_id.text.strip()
        
        # lead investigator
        lead_investigator = row.find('td', class_='column-2')
        if lead_investigator:
            row_data['lead_investigator'] = lead_investigator.text
        
        # lead institution
        lead_institution = row.find('td', class_='column-3')
        if lead_institution:
            row_data['lead_institution'] = lead_institution.text.strip()
        
        # study title and link
        title_cell = row.find('td', class_='column-4')
        if title_cell:
            link_tag = title_cell.find('a')
            if link_tag:
                # Extract the title text
                row_data['study_title'] = link_tag.text.strip()
                
                # Extract the href attribute
                if 'href' in link_tag.attrs:
                    row_data['link'] = link_tag['href']
                    
                    # Get study_id only if link exists
                    try:
                        study_id = vivli_nct_grapper(link_tag['href'])
                        row_data['study_id'] = study_id
                        row_data['number_of_studies'] = len(study_id) if study_id else 0
                    except Exception as e:
                        print(f"Error getting NCT IDs for {link_tag['href']}: {e}")
                        row_data['study_id'] = []
                        row_data['number_of_studies'] = 0
                else:
                    row_data['link'] = ""
                    row_data['study_id'] = []
                    row_data['number_of_studies'] = 0
        
        # Append the row data to the main data list
        if row_data:
            data.append(row_data)

        

    return data

def vivli_nct_grapper(link):
    """
    Extract NCT IDs from the provided Link on Vivli
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
        
        if 'span' in p.text:
            # cheeck for span tags
            span_tags = p.find_all('span')
            for span in span_tags:
                # Find all NCT IDs in the text
                if r'NCT\d{8}' in span.text:
                    found_ids = re.findall(r'NCT\d{8}', span.text)
                    if found_ids:
                        NCT_IDs.extend(found_ids)
                elif 'NCT' in span.text:
                    found_id = span.text.split()
                    next_span = span.find_next('span')
                    if next_span and 'NCT' in next_span.text:
                        found_id.append(next_span.text)
                        if re.match(r'NCT\d{8}', found_id[-1]):
                            NCT_IDs.extend(found_id)
    return NCT_IDs

def get_csdr_data():
    """
    Scrape data from CSDR platform
    data requested from the CSDR platform

    data structure:
        {
        request_id: str,
        lead_investigator: str,
        lead_institution: str,
        study_title: str,
        link: str,
        stduy_id: str,
        number_of_studies: int
        }
    
    """  
    url = "https://www.clinicalstudydatarequest.com/Metrics/Agreed-Proposals.aspx"

    data = []
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from CSDR. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the data
    table = soup.find('table', {'id': 'proposals-table'})
    if not table:
        raise Exception("Failed to find the proposals table on CSDR page.")
    
    tbody = table.find('tbody')
    if not tbody:
        raise Exception("Failed to find the table body in the proposals table on CSDR page.")
    
    # Iterate through each row in the table
    for row in tbody.find_all('tr'):
        row_data = {}
        counter = 0
        for cell in row.find_all('td'):
            if counter == 0:
                # Request ID
                row_data['request_id'] = cell.text.strip()
                # check request id already in CSDR file
                load_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "csdr_data.json")
                if os.path.exists(load_data_path):
                    with open(load_data_path, 'r') as f:
                        existing_data = json.load(f)
                    # check if request_id already exists
                    if any(d['request_id'] == row_data['request_id'] for d in existing_data):
                        print(f"Request ID {row_data['request_id']} already exists in the CSDR data file. Skipping.")
                        row_data = {}  # Reset row_data to skip this row
                        break  # Skip to the next row
            elif counter == 1:
                # Sponsor
                row_data['sponsor'] = cell.text.strip()
            elif counter == 2:
                # title
                row_data['study_title'] = cell.text.strip()
            elif counter == 3:
                # lead_institution
                row_data['lead_institution'] = cell.text.strip()
            elif counter == 4:
                # Lead Researcher
                row_data['lead_researcher'] = cell.text.strip()
            elif counter == 5:
                # Link
                link_tag = cell.find('a', href=True)
                row_data['link'] = link_tag['href']
                # parsing link, because starts with ..
                if row_data['link'].startswith('../'):
                    row_data['link'] = 'https://www.clinicalstudydatarequest.com' + row_data['link'][2:]
                    #print(f"Updated link: {row_data['link']}")
                # get study_id from link
                study_id = csdr_ID_grapper(row_data['link']) 
                
                row_data['study_id'] = study_id
                row_data['number_of_studies'] = len(study_id) if study_id else 0
            
            counter += 1
        # Append the row data to the main data list
        if row_data:
            data.append(row_data)

        # save progressb every 5 rows
        if len(data) % 5 == 0:
            print(f"Progress: {len(data)} rows scraped so far.")
            # attach data to a json file
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
            os.makedirs(output_dir, exist_ok=True)
            csdr_file_path = os.path.join(output_dir, "csdr_data.json")
            with open(csdr_file_path, 'w') as f:
                json.dump(data, f, indent=4)

    if not data:
        raise Exception("No data found in the CSDR page.")
    

    
    return data

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
                
    return study_ids

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
    Scrape data from YODA platform
    data requested from the YODA platform

    data structure:
        {
        request_id: str,
        lead_investigator: str,
        lead_institution: str,
        study_title: str,
        link: str,
        stduy_id: str,
        number_of_studies: int
        }
    
    """
    data = []

    for page in range(1, 51):
        # depending on page count
        url = f"https://yoda.yale.edu/metrics/submitted-requests-to-use-johnson-johnson-data/data-requests-johnson-and-johnson/?_paged={page}"
        
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch data from YODA page {page}. Status code: {response.status_code}")
            continue
        
        soup = BeautifulSoup(response.content, 'html.parser')

        # define the class pattern for request items
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
        # Find all request items
        for pattern in class_pattern:
            items = soup.find_all('div', {'class': pattern})
            requests_items.extend(items)
        if not requests_items:
            print(f"No request items found on page {page}.")
            continue


        for item in requests_items:
            request_data = {}
            
            # Extract from left cell
            left_cell = item.find('div', {'class': 'left-cell'})
            if left_cell:
                # Extract request ID
                project_number = left_cell.find('span', {'class': 'yoda-caption dark-caption project-number'})
                if project_number:
                    request_data['request_ID'] = project_number.text.strip()
                
                # Extract number of trials provided - also in left-cell
                trials_provided = left_cell.find('span', {'class': 'yoda-caption request-item__caption'}, string=lambda text: text and 'No. Trials Provided' in text)
                if trials_provided:
                    num_trials = trials_provided.find_next('span', {'class': 'yoda-caption dark-caption'})
                    if num_trials:
                        request_data['number_of_trial_provided'] = num_trials.text.strip()
            
            # Extract from right cell
            right_cell = item.find('div', {'class': 'right-cell'})
            if right_cell:
                # extract lead investigator
                lead = right_cell.find('span', {'class': 'yoda-caption dark-caption font-700'})
                if lead:
                    request_data['lead_investigator'] = lead.text.strip()

                # extract lead institution
                institute = right_cell.find('span', {'class': 'yoda-caption'})
                if institute and not institute.find_parent('h4'):  # Avoid getting caption labels
                    request_data['lead_institution'] = institute.text.strip()

                # extract study title
                title = right_cell.find('h3', {'class': 'request-item__title color-blue display-block'})
                if title:
                    request_data['title'] = title.text.strip()

                # extract link (PDF link)
                pdf_link = right_cell.find('div', {'class': 'flex-container flex-wrap project-docs-ctas'})
                if pdf_link:
                    pdf_links = pdf_link.find_all('a', href=True)
                    for link in pdf_links:
                        if 'due' in link['href'].lower():
                            request_data['PDF_Link'] = link['href']
                            break

                # Extract study ID from the PDF link
                if link['href']:
                    try:
                        study_ids = yoda_nct_grapper(link['href'])
                        request_data['study_id'] = study_ids
                        request_data['number_of_studies'] = len(study_ids) if study_ids else 0
                    except Exception as e:
                        print(f"Error getting NCT IDs for {link['href']}: {e}")
                        request_data['study_id'] = []
                        request_data['number_of_studies'] = 0

            # Append the request data to the main data list
            if request_data:
                data.append(request_data)

    return data

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
    Main function to run the scrapers
    """
    #vivli_data = get_vivli_data()
    csdr_data = get_csdr_data()
    #yoda_data = get_yoda_data()

    # Save the data as json file
    # Use the correct absolute path to your data directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    #vivli_file_path = os.path.join(output_dir, "vivli_data.json")
    #with open(vivli_file_path, 'w') as f:
    #    json.dump(vivli_data, f, indent=4)

    csdr_file_path = os.path.join(output_dir, "csdr_data.json")
    with open(csdr_file_path, 'w') as f:
        json.dump(csdr_data, f, indent=4)
    #yoda_file_path = os.path.join(output_dir, "yoda_data.json")
    #with open(yoda_file_path, 'w') as f:
    #    json.dump(yoda_data, f, indent=4)

    #print(f"Vivli data saved to {vivli_file_path}")
    #print(f"CSDR data saved to {csdr_file_path}")
    #print(f"YODA data saved to {yoda_file_path}")

    # Print the number of studies scraped
    #print(f"Number of studies scraped from Vivli: {len(vivli_data)}")
    print(f"Number of studies scraped from CSDR: {len(csdr_data)}")
    #print(f"Number of studies scraped from YODA: {len(yoda_data)}")

    #print(f"Total number of studies scraped: {len(vivli_data) + len(csdr_data) + len(yoda_data)}")


if __name__ == "__main__":
    print("Starting the scraping process...")
    main()
    print("Scraping completed.")
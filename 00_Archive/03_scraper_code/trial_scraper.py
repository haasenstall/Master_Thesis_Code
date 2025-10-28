import os
import requests
from bs4 import BeautifulSoup
import json
import re
import time
import pandas as pd
import tqdm
import sys
from pathlib import Path

# Add path to the classes module
sys.path.append(str(Path(__file__).parent.parent / "07_classes"))
from classes import trial, trial_list

def vivli_trial():
    """
    Vivli trial metadata scraper using the trial class structure with dynamic end detection
    """
    # Create a trial_list object to store trials
    trial_collection = trial_list()
    
    # Check for existing data file
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    file_path = os.path.join(dirc, "vivli_trials.json")

    if os.path.exists(file_path):
        trial_collection.load_from_file(file_path)
        print(f"Loaded {len(trial_collection.trials)} existing trials")
    
    # Set initial range
    if len(trial_collection.trials) > 0:
        start = len(trial_collection.trials) + 1
    else:
        start = 1
    max_end = 20000  # Maximum possible end - a safety limit
    
    # Check if the trials list already contains study IDs
    existing_study_ids = {t['nct_id'] for t in trial_collection.trials if 'nct_id' in t}
    
    # Dynamic end detection variables
    consecutive_not_found = 0
    max_consecutive_not_found = 500  # Stop after 200 consecutive IDs without studies
    last_found_id = 0
    
    # Create a progress bar - start with a large range that will be updated
    pbar = tqdm.tqdm(range(start, max_end), desc="Scraping Vivli Trials", unit="study")

    # Loop through the study IDs
    for study_id in pbar:
        # Update progress bar description
        pbar.set_description(f"Scraping Study ID: {study_id:08d} (Last found: {last_found_id:08d})")
        
        # Format study_id with leading zeros
        formatted_study_id = f"{study_id:08d}"
        
        url = f"https://prod-api.vivli.org/api/studies/{formatted_study_id}/metadata/fromdoi"
        try: 
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Extract NCT ID
                nct_id = data.get("nctId", "")
                if nct_id is None or nct_id == "":
                    nct_id = data.get("registryId", "")
                
                # Skip if not a valid NCT ID or already processed
                if not nct_id or not re.match(r'^NCT\d{8}$', nct_id) or nct_id in existing_study_ids:
                    pbar.set_postfix(found=False, status="invalid or duplicate")
                    if nct_id in existing_study_ids:
                        consecutive_not_found = 0  # Reset counter for duplicates
                        continue
                    consecutive_not_found += 1
                    continue
                
                # Valid study found - reset counter and update last found ID
                consecutive_not_found = 0
                last_found_id = study_id
                
                # Create a trial object
                new_trial = trial(
                    nct_id=nct_id,
                    title=data.get("studyTitle", None),
                    start_date=None,  # Skip dates as requested
                    end_date=None,    # Skip dates as requested
                    verification_date=None,  # Skip dates as requested
                    upload_date=data.get("postedDate", None),
                    status=None,
                    phase=data.get("phase", None),
                    condition=None,  # Will be added later from AACT
                    condition_tree_number=None,  # Will be added later from AACT
                    intervention=None,  # Will be added later from AACT
                    intervention_tree_number=None,  # Will be added later from AACT
                    sponsor=data.get("orgName", None),  # Added sponsor from orgName
                    country=None,  # Will be added later from AACT
                    requests=None,
                    request_rate=None
                )
                
                # Add trial to the collection
                trial_collection.add_trial(new_trial)
                existing_study_ids.add(nct_id)
                
                # Update progress bar postfix with success message
                pbar.set_postfix(found=True, nct=nct_id, consecutive_empty=consecutive_not_found)
                
                # Save every 20 trials to avoid data loss
                if len(trial_collection.trials) % 20 == 0:
                    trial_collection.save_to_file(file_path)
                    
            else:
                # No valid data found - increment counter
                consecutive_not_found += 1
                pbar.set_postfix(found=False, status=response.status_code, consecutive_empty=consecutive_not_found)
                time.sleep(0.1)
                
        except Exception as e:
            # Count this as not found as well
            consecutive_not_found += 1
            pbar.set_postfix(error=str(e)[:20], consecutive_empty=consecutive_not_found)
            time.sleep(5)
        
        # Check if we should stop based on consecutive not found count
        if consecutive_not_found >= max_consecutive_not_found:
            print(f"\nStopping scraper: No studies found for {consecutive_not_found} consecutive IDs")
            print(f"Last successful study ID: {last_found_id}")
            break
    
    # Save the complete collection
    trial_collection.save_to_file(file_path)
    print(f"Saved {len(trial_collection.trials)} Vivli trials to {file_path}")
    print(f"Last found study ID: {last_found_id}")
    
    # Save the last found ID to a file for future reference
    last_id_file = os.path.join(dirc, "vivli_last_id.txt")
    with open(last_id_file, 'w') as f:
        f.write(str(last_found_id))
    print(f"Last found ID saved to {last_id_file}")
    
    return trial_collection

def csdr_trial():
    """
    CSDR trial metadata scraper using the trial class structure
    """
    # Create a trial_list object to store trials
    trial_collection = trial_list()
    
    # Load existing data from CSV
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    csv_file_path = os.path.join(dirc, "All-Sponsor-Studies.csv")
    output_file = os.path.join(dirc, "csdr_trials.json")
    
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist.")
        return trial_collection
    
    # Check for existing processed data
    if os.path.exists(output_file):
        trial_collection.load_from_file(output_file)
        print(f"Loaded {len(trial_collection.trials)} existing CSDR trials")
        # Get existing NCT IDs to avoid duplicates
        existing_nct_ids = {t['nct_id'] for t in trial_collection.trials if 'nct_id' in t}
    else:
        existing_nct_ids = set()
    
    # Read CSV data
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    
    # Create progress bar for processing CSV rows
    pbar = tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing CSDR studies")
    
    for index, row in pbar:
        study_id = row['Posting ID']
        pbar.set_description(f"Processing CSDR study {study_id}")
        
        nct_id = row['Trial Registry Identification Number(s)']
        if pd.isna(nct_id) or not re.match(r'^NCT\d{8}$', nct_id) or nct_id in existing_nct_ids:
            pbar.set_postfix(valid_nct=False, reason="invalid or duplicate")
            continue
        
        # Convert date to ISO format if present
        posted_date = None
        if pd.notna(row['Date Added to this Site']):
            try:
                posted_date = pd.to_datetime(row['Date Added to this Site']).isoformat()
            except:
                posted_date = None
        
        # Create a trial object
        new_trial = trial(
            nct_id=nct_id,
            title=row['Study Title'] if pd.notna(row['Study Title']) else None,
            start_date=None,  # Skip dates as requested
            end_date=None,    # Skip dates as requested
            verification_date=None,  # Skip dates as requested
            upload_date=posted_date,
            status=None,  # Not available in CSDR CSV
            phase=row['Phase'] if pd.notna(row['Phase']) else None,
            condition=None,  # Will be added later from AACT
            condition_tree_number=None,  # Will be added later from AACT
            intervention=None,  # Will be added later from AACT
            intervention_tree_number=None,  # Will be added later from AACT
            sponsor=None,
            country=None,  # Will be added later from AACT
            requests=None,
            request_rate=None
        )
        
        # Add trial to the collection
        trial_collection.add_trial(new_trial)
        existing_nct_ids.add(nct_id)
        
        pbar.set_postfix(valid_nct=True)
        
        # Save every 50 trials to avoid data loss
        if len(trial_collection.trials) % 50 == 0:
            trial_collection.save_to_file(output_file)
    
    # Save the complete collection
    trial_collection.save_to_file(output_file)
    print(f"Saved {len(trial_collection.trials)} CSDR trials to {output_file}")
    return trial_collection

def yoda_trial():
    """
    YODA trial metadata scraper using the trial class structure
    """
    # Create a trial_list object to store trials
    trial_collection = trial_list()
    
    # Check for existing data file
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    file_path = os.path.join(dirc, "yoda_trials.json")

    if os.path.exists(file_path):
        trial_collection.load_from_file(file_path)
        print(f"Loaded {len(trial_collection.trials)} existing YODA trials")
        # Get existing NCT IDs to avoid duplicates
        existing_nct_ids = {t['nct_id'] for t in trial_collection.trials if 'nct_id' in t}
    else:
        existing_nct_ids = set()
    
    # Create progress bar for pages
    total_pages = 51   
    pbar_pages = tqdm.tqdm(range(1, total_pages + 1), desc="Processing YODA pages")
    
    for page in pbar_pages:
        pbar_pages.set_description(f"Processing YODA page {page}/{total_pages}")
        
        url = f"https://yoda.yale.edu/trials-search/?_paged={page}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            trial_containers = soup.find_all('div', class_="trial flex-container align-top")

            # Create nested progress bar for trials on the current page
            pbar_trials = tqdm.tqdm(trial_containers, leave=False, desc=f"Page {page} trials")
                
            # Iterate through each trial container
            for row in pbar_trials:
                # Extract the NCT ID
                study_id_div = row.find('div', class_='trial__nct-id trial-cell')
                if not study_id_div:
                    continue
                        
                nct_id_link = study_id_div.find('a')
                if not nct_id_link:
                    continue
                    
                nct_id = nct_id_link.text.strip()
                pbar_trials.set_description(f"Trial {nct_id}")

                # Skip if already processed
                if nct_id in existing_nct_ids:
                    pbar_trials.set_postfix(status="skipped")
                    continue
                
                pbar_trials.set_description(f"Processing {nct_id}")                        

                # Extract title
                title_div = row.find('div', class_='trial__title trial-cell')
                if not title_div:
                    continue
                trial_link = title_div.find('a')
                if not trial_link:
                    continue
                study_title = trial_link.text.strip()

                # Extract further information from trial page
                link = trial_link['href']
                trial_response = requests.get(link)
                if trial_response.status_code != 200:
                    pbar_trials.set_postfix(details="failed")
                    continue
                    
                trial_soup = BeautifulSoup(trial_response.content, 'html.parser')
                
                # Extract study phase
                phase_div = trial_soup.find('div', class_='trial__phase')
                phase = None
                if phase_div:
                    phase_number = phase_div.find('span', class_='phase-number')
                    if phase_number:
                        phase = f"Phase {phase_number.text.strip()}"
                
                # Extract other information
                org_name = None
                published_date = None
                
                # Extract organization name from trial info div
                trial_info_div = trial_soup.find('div', class_='trial__info grid-x trial-half-w-grid')
                if trial_info_div:
                    for cell in trial_info_div.find_all('div', class_='info-cell'):
                        for title_span in cell.find_all('span', class_='info-title'):
                            title = title_span.text.strip()
                            if title == 'Data Partner':
                                text_span = title_span.find_next('span', class_='info-text')
                                if text_span:
                                    org_name = text_span.text.strip()
                
                # Extract published date from schema.org metadata
                schema_div = trial_soup.find('script', {'type': 'application/ld+json'})
                if schema_div:
                    try:
                        schema_data = json.loads(schema_div.string)
                        for item in schema_data.get('@graph', []):
                            if item.get('@type') == 'WebPage' and 'datePublished' in item:
                                published_date = pd.to_datetime(item['datePublished']).isoformat()
                                break
                    except (json.JSONDecodeError, Exception):
                        pass
                
                # Create a trial object
                new_trial = trial(
                    nct_id=nct_id,
                    title=study_title,
                    start_date=None,  # Skip dates as requested
                    end_date=None,    # Skip dates as requested
                    verification_date=None,  # Skip dates as requested
                    upload_date=published_date,
                    status=None,  # Default value for YODA
                    phase=phase,
                    condition=None,  # Will be added later from AACT
                    condition_tree_number=None,  # Will be added later from AACT
                    intervention=None,  # Will be added later from AACT
                    intervention_tree_number=None,  # Will be added later from AACT
                    sponsor=None,
                    country=None,  # Will be added later from AACT
                    requests=None,
                    request_rate=None
                )
                
                # Add trial to the collection
                trial_collection.add_trial(new_trial)
                existing_nct_ids.add(nct_id)
                
                pbar_trials.set_postfix(status="added")
                
                # Save every 10 trials to avoid data loss
                if len(trial_collection.trials) % 10 == 0:
                    trial_collection.save_to_file(file_path)
        
        else:
            pbar_pages.set_postfix(status=f"Error {response.status_code}")
            time.sleep(5)

    # Save the complete collection
    trial_collection.save_to_file(file_path)
    print(f"Saved {len(trial_collection.trials)} YODA trials to {file_path}")
    return trial_collection

def main():
    """
    Main function to run all trial scrapers using the trial class structure
    """
    print("Starting Vivli trial metadata scraping...")
    vivli_collection = vivli_trial()
    print(f"Vivli trial metadata scraping completed with {len(vivli_collection.trials)} trials.")

    print("\nStarting CSDR trial metadata scraping...")
    csdr_collection = csdr_trial()
    print(f"CSDR trial metadata scraping completed with {len(csdr_collection.trials)} trials.")

    print("\nStarting YODA trial metadata scraping...")
    yoda_collection = yoda_trial()
    print(f"YODA trial metadata scraping completed with {len(yoda_collection.trials)} trials.")
    
    # Generate summary report
    print("\n=== Trial Scraping Summary ===")
    print(f"Vivli trials: {len(vivli_collection.trials)}")
    print(f"CSDR trials: {len(csdr_collection.trials)}")
    print(f"YODA trials: {len(yoda_collection.trials)}")
    print(f"Total trials: {len(vivli_collection.trials) + len(csdr_collection.trials) + len(yoda_collection.trials)}")

if __name__ == "__main__":
    main()
import os
import requests
from bs4 import BeautifulSoup
import json
import re
import time
import pandas as pd
import tqdm

def vivli_trial():
    """
    vivli trial metadata scraper
    result structure:
    {
    "study_id": "00000001",
    "nct_id": "NCT00000001",
    "studyTitle": "Example Study Title",
    "overallStatus": "Completed",
    "phase": "Phase 3",
    "studyType": "Interventional",
    "conditions": ["Condition Name"],
    "orgCode": "ORG123",
    "orgName": "Organization Name",
    "status": "Posted",
    "postedDate":"2018-07-09T16:07:01.1765061+00:00",
    "therapeuticAreas":["Cardiovascular Disease"],
    "extractedInterventions":["diet, fat-restricted"],
    "documentType":"Study"
    }
    """
    # create a json file to store the data
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    file_path = os.path.join(dirc, "vivli_trials.json")

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    if data != []:
        trials = data
    else:
        trials = []

    # set range
    start = 1
    end = 10026  # Number of studies listed on Vivli

    # Check if the trials list already contains study IDs
    existing_study_ids = {trial['study_id'] for trial in trials}
    # If the trials list is not empty, set start to the next available study ID
    if existing_study_ids:
        max_study_id = max(int(trial['study_id']) for trial in trials)
        start = max_study_id + 1
        print(f"Starting from study ID: {start:08d}")
    else:
        print("No existing study IDs found. Starting from 00000001.")
    # check if there are more studies then end
    if start >= end:
        start += 10
        url = f"https://prod-api.vivli.org/api/studies/{start:08d}/metadata/fromdoi"
        response = requests.get(url)
        if response.status_code == 404:
            print(f"No studies found from {start:08d} to {end:08d}.")
            return
        else:
            print(f"Found studies from {start:08d} to {end:08d}.")
            end = start + 10

    # create a progress bar
    pbar = tqdm.tqdm(range(start, end), desc="Scraping Vivli Trials", unit="study")

    # Loop through the study IDs from 1 to 10025 - Number of stuides listed on Vivli      
    for study_id in pbar:
        # progress bar
        pbar.set_description(f"Scraping Study ID: {study_id:08d}")
        # check if study_id already exists in the trials list
        if any(trial['study_id'] == f"{study_id:08d}" for trial in trials):
            print(f"Study ID {study_id:08d} already exists in the data.")
            continue
        url = f"https://prod-api.vivli.org/api/studies/{study_id:08d}/metadata/fromdoi"
        try: 
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Extracting the NCT_ID from the metadata
                nct_id = data.get("nctId", "")
                if nct_id is None or nct_id == "":
                    nct_id = data.get("registryId", "")
                if nct_id and re.match(r'^NCT\d{8}$', nct_id):
                    trial = {
                        "study_id": f"{study_id:08d}",
                        "nct_id": nct_id,
                        "studyTitle": data.get("studyTitle", ""),
                        "overallStatus": data.get("overallStatus", ""),
                        "phase": data.get("phase", ""),
                        "studyType": data.get("studyType", ""),
                        "conditions": data.get("conditions", []),
                        "orgCode": data.get("orgCode", ""),
                        "orgName": data.get("orgName", ""),
                        "status": data.get("status", ""),
                        "postedDate": data.get("postedDate", ""),
                        "therapeuticAreas": data.get("therapeuticAreas", []),
                        "extractedInterventions": data.get("extractedInterventions", []),
                        "documentType": data.get("documentType", "")
                    }
                    trials.append(trial)

                    # Update progress bar postfix with success message
                    pbar.set_postfix(found=True, nct=nct_id)

                    # Save every 20 trials to avoid data loss
                    if len(trials) % 20 == 0:
                        with open(file_path, 'w') as f:
                            json.dump(trials, f, indent=4)

                else:
                    # Update progress bar postfix with error message
                    pbar.set_postfix(found=False, status=response.status_code)
                    time.sleep(1)
        except Exception as e:
            # Update progress bar postfix with exception message
            pbar.set_postfix(error=str(e)[:20])
            time.sleep(5)

    # Save the updated trials list to the JSON file
    with open(file_path, 'w') as f:
        json.dump(trials, f, indent=4)

    vivli_trial()

def csdr_trial():
    """
    CSDR trial metadata scraper
    result structure:
    {
    "study_id": "00000001",
    "nct_id": "NCT00000001",
    "studyTitle": "Example Study Title",
    "overallStatus": "Completed",
    "phase": "Phase 3",
    "studyType": "Interventional",
    "conditions": ["Condition Name"],
    "orgCode": "ORG123",
    "orgName": "Organization Name",
    "status": "Posted",
    "postedDate":"2018-07-09T16:07:01.1765061+00:00",
    "therapeuticAreas":["Cardiovascular Disease"],
    "extractedInterventions":["diet, fat-restricted"],
    "documentType":"Study"
    }
    """
    # load data from All-Sponsor-Studies.csv
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    file_path = os.path.join(dirc, "All-Sponsor-Studies.csv")
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    
    df = pd.read_csv(file_path, encoding='utf-8')
    trials = []

    # Create progress bar for processing CSV rows
    pbar = tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing CSDR studies")
    
        

    for index, row in df.iterrows():
        study_id = row['Posting ID']
        nct_id = row['Trial Registry Identification Number(s)']
        if pd.isna(nct_id) or not re.match(r'^NCT\d{8}$', nct_id):
            print(f"Study ID {study_id} does not have a valid NCT ID.")
            continue
        
        trial = {
            "study_id": study_id,
            "nct_id": nct_id,
            "studyTitle": row['Study Title'],
            "overallStatus": '',
            "phase": row['Phase'],
            "studyType": '',
            "conditions": row['Medical Condition'].split(', ') if pd.notna(row['Medical Condition']) else [],
            "orgCode": '',
            "orgName": row['Sponsor'],
            "status": '',
            "postedDate": row['Date Added to this Site'],
            "therapeuticAreas": [],
            "extractedInterventions": [],
            "documentType": ''
        }

        # change postedDate to ISO format
        if pd.notna(trial['postedDate']):
            trial['postedDate'] = pd.to_datetime(trial['postedDate']).isoformat()
        else:
            trial['postedDate'] = ''

        # Add the trial to the list
        trials.append(trial)
        pbar.set_postfix(valid_nct=True)

    # Save the trials to a JSON file
    output_file = os.path.join(dirc, "csdr_trials.json")
    with open(output_file, 'w') as f:
        json.dump(trials, f, indent=4)  

    print(f"Data saved to {output_file}")

def yoda_trial():
    """
    YODA trial metadata scraper
    """
    # create a json file to store the data
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    file_path = os.path.join(dirc, "yoda_trials.json")

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
        data = []
    else:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    
    if data:
        trials = data
    else:
        trials = []

    # Track processed NCT IDs to avoid duplicates
    processed_nct_ids = set(trial['nct_id'] for trial in trials if 'nct_id' in trial)
    
    # Create progress bar for pages
    total_pages = 51   
    pbar_pages = tqdm.tqdm(range(1, total_pages + 1), desc="Processing YODA pages")
    
    for page in pbar_pages:
        pbar_pages.set_description(f"Processing YODA page {page}/{total_pages}")
        
        url = f"https://yoda.yale.edu/trials-search/?_paged={page}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            trial_containers = soup.find_all('div', class_ = "trial flex-container align-top")

            # Create nested progress bar for trials on the current page
            pbar_trials = tqdm.tqdm(trial_containers, leave=False, desc=f"Page {page} trials")
                
            # iterate through each trial flex container
            for row in pbar_trials:
                # Extract the study ID
                study_id_div = row.find('div', class_ = 'trial__nct-id trial-cell')
                if not study_id_div:
                        continue
                        
                nct_id_link = study_id_div.find('a')
                if not nct_id_link:
                    continue
                    
                nct_id = nct_id_link.text.strip()

                pbar_trials.set_description(f"Trial {nct_id}")

                # Skip if already processed
                if nct_id in processed_nct_ids:
                    pbar_trials.set_postfix(status="skipped")
                    continue
                
                # Update progress bar description
                pbar_trials.set_description(f"Processing {nct_id}")                        

                # extract title
                title_div = row.find('div', class_ = 'trial__title trial-cell')
                if not title_div:
                    continue
                trial_link = title_div.find('a')
                if not trial_link:
                    continue
                study_title = trial_link.text.strip()

                # extract further informations from Link
                link = trial_link['href']
                trial_response = requests.get(link)
                if trial_response.status_code != 200:
                        pbar_trials.set_postfix(details="failed")
                        continue
                trial_soup = BeautifulSoup(trial_response.content, 'html.parser')
                
                # extract study phase
                phase_div = trial_soup.find('div', class_='trial__phase color-white flex-container align-middle align-center align-content-center flex-wrap')
                phase_text = phase_div.find('span', class_='display-inline-block text-center phase-number full-width').text.strip() if phase_div else ''
                if phase_text:
                    phase = 'Phase' + phase_text
                else:
                    phase = ''
                    
                # extract other informations
                trial_info_div = trial_soup.find('div', class_='trial__info grid-x trial-half-w-grid')
                if trial_info_div:
                    info_data = {}

                    for info in trial_info_div.find_all('span'):
                        if info.find('span', class_ = 'info-title display-block color-blue font-600'):
                            title = info.text.strip()
                        else:
                            continue
                        text = info.find_next('span').text.strip() if info.find_next('span') else ''

                        info_data.append({
                            title: text
                        })
                
                # get informations from info data
                therapeutic_area = info_data.get('Therapeutic Area', '')
                conditions = info_data.get('Condition Studied', '').split(', ') if info_data.get('Medical Condition') else []
                org_name = info_data.get('Data Partner', '')

                # get info from schema.org graph 
                schema_div = trial_soup.find('script', class_ = 'yoast-schema-graph')
                published_date = ''
                if schema_div:
                    schema_data = json.loads(schema_div.string)
                    # Extract publishedDate 
                    for item in schema_data.get('@graph', []):
                        if item.get('@type') == 'WebPage' and 'datePublished' in item:
                            published_date = item['datePublished']
                            break

                    if published_date:
                        # Convert to ISO format
                        published_date = pd.to_datetime(published_date).isoformat()
                    else:
                        published_date = ''

                    
                trial = {
                    "study_id": nct_id,
                    "nct_id": nct_id,
                    "studyTitle": study_title,
                    "overallStatus": '',
                    "phase": phase,
                    "studyType": '',
                    "conditions": conditions,
                    "orgCode": '',
                    "orgName": org_name,
                    "status": '',
                    "postedDate": published_date,
                    "therapeuticAreas": [therapeutic_area] if therapeutic_area else [],
                    "extractedInterventions": [],
                    "documentType": ''
                }

                trials.append(trial)
                processed_nct_ids.add(nct_id)
                pbar_trials.set_postfix(status="added")

        else:
            pbar_pages.set_postfix(status=f"Error {response.status_code}")
            time.sleep(5)

    # Save the updated trials list to the JSON file
    with open(file_path, 'w') as f:
        json.dump(trials, f, indent=4)
    print(f"Data saved to {file_path}")
    return
    



def main():
    """
    Main function to run the vivli trial scraper
    """
    vivli_trial()
    print("Vivli trial metadata scraping completed.")

    #csdr_trial()
    #print("CSDR trial metadata scraping completed.")

    #yoda_trial()
    #print("YODA trial metadata scraping completed.")


if __name__ == "__main__":
    main()
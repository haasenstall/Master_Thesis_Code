import json
import os
import pandas as pd
import numpy as np
import re

def load_data(file_path):
    """
    Load data from a JSON file.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"File {file_path} does not exist.")
        return []
    
def save_data(file_path, data):
    """
    Save data to a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_path}")

def analyze_data(data, name):
    """
    Analyze data on quantitative aspects to create a data summary.
    """
    summary = []
    num_requests = 0
    num_study_ids = 0
    num_nct_ids = 0
    unique_study_ids = set()
    unique_nct_ids = set()

    for item in data:
        # Count requests - handle comma-separated values
        request = item.get('request_id', '')
        if request:
            if ',' in request:
                # Split by comma and count each request ID
                request_ids = [r.strip() for r in request.split(',') if r.strip()]
                num_requests += len(request_ids)
            else:
                # Single request ID
                num_requests += 1

        # Count study IDs
        study_id = item.get('study_id', [])
        
        
        # Handle study_id based on its type
        if isinstance(study_id, list):
            num_study_ids += len(study_id)
            # Count NCT IDs in list
            for sid in study_id:
                unique_study_ids.add(sid)
                if isinstance(sid, str) and sid.startswith('NCT'):
                    num_nct_ids += 1
                    unique_nct_ids.add(sid)
                    
        elif isinstance(study_id, str):
            # Handle string study_id
            if ',' in study_id:
                # Multiple IDs separated by commas
                ids = [s.strip() for s in study_id.split(',') if s.strip()]
                num_study_ids += len(ids)
                unique_study_ids.add(study_id)
                
                # Count NCT IDs
                for sid in ids:
                    if sid.startswith('NCT'):
                        num_nct_ids += 1
                        unique_nct_ids.add(sid)
            else:
                # Single ID
                num_study_ids += 1
                unique_study_ids.add(study_id)
                if study_id.startswith('NCT'):
                    num_nct_ids += 1
                    unique_nct_ids.add(study_id)

    # Create summary
    summary = {
        'source_name': name,
        'number_of_studies': len(data),
        'number_of_requests': num_requests,
        'number_of_study_ids': num_study_ids,
        'number_of_nct_ids': num_nct_ids,
        'number_unique_study_ids': len(unique_study_ids),
        'number_unique_nct_ids': len(unique_nct_ids)
    }
    
    return summary

def parse_nct_id(data):
    """
    Check if NCT ID is in the right format and fix if necessary.
    """
    for item in data:
        if 'study_id' in item:
            # Handle case where study_id is a list
            if isinstance(item['study_id'], list):
                for i, sid in enumerate(item['study_id']):
                    if isinstance(sid, str) and sid.startswith('NCT'):
                        if not re.match(r'^NCT\d{8}$', sid):
                            # Fix invalid NCT ID
                            item['study_id'][i] = sid[:11]
            
            # Handle case where study_id is a string
            elif isinstance(item['study_id'], str) and 'NCT' in item['study_id']:
                # Process comma-separated IDs
                if ',' in item['study_id']:
                    ids = []
                    for sid in item['study_id'].split(','):
                        sid = sid.strip()
                        if sid.startswith('NCT'):
                            if not re.match(r'^NCT\d{8}$', sid):
                                sid = sid[:11]  # Fix invalid NCT ID
                        ids.append(sid)
                    item['study_id'] = ','.join(ids)
                else:
                    # Single ID
                    if item['study_id'].startswith('NCT') and not re.match(r'^NCT\d{8}$', item['study_id']):
                        item['study_id'] = item['study_id'][:11]
    
    return data  # Return the modified data

def main():
    # file paths
    csdr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "csdr_data.json")
    vivli_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "vivli_data.json")
    yoda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "yoda_data.json")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "data_summary.json")

    # load data 
    csdr_data = load_data(csdr_path)
    vivli_data = load_data(vivli_path)
    yoda_data = load_data(yoda_path)

    # parse NCT IDs - onmly relevant for CSDR data
    parse_nct_id(csdr_data)

    # attach data to one for a total summary
    total_data = []
    total_data.extend(csdr_data)
    total_data.extend(vivli_data)
    total_data.extend(yoda_data)

    # analyze data
    csdr_summary = analyze_data(csdr_data, "CSDR")
    vivli_summary = analyze_data(vivli_data, "Vivli")
    yoda_summary = analyze_data(yoda_data, "YODA")

    total_summary = analyze_data(total_data, "Total")

    # save summaries to one file
    summary = []

    summary.append(csdr_summary)
    summary.append(vivli_summary)
    summary.append(yoda_summary)
    summary.append(total_summary)

    # save data
    save_data(output_path, summary)


if __name__ == "__main__":
    main()
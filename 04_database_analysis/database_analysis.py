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

def analyze_data(data):
    """
    Analyze data on quantative aspects to creat a data summary.
    """
    
    return summary

def parse_nct_id(data):
    """
    check if NCT ID is in the right format.
    Important - there is study id and nct id, so not deleting the the string if its not a NCT ID.
    """

    for item in data:
        if 'study_id' in item and 'NCT' in item['study_id']:
            for id in item['study_id'].split(','):
                if id.startswith('NCT'):
                    if not re.match(r'^NCT\d{8}$', id.strip()):
                        # NCT ID invalid and need to be cut 
                        id = id.strip()[:11]
                    
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
    csdr_summary = analyze_data(csdr_data)
    vivli_summary = analyze_data(vivli_data)
    yoda_summary = analyze_data(yoda_data)

    total_summary = analyze_data(total_data)

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
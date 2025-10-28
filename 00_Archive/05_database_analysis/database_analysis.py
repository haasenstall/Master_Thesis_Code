import json
import os
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add path to the classes module
sys.path.append(str(Path(__file__).parent.parent / "07_classes"))
from classes import trial, trial_list, trial_request, trial_request_list

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

def analyze_data_with_classes(data, name):
    """
    Analyze data using the trial_request_list class for more structured analysis.
    """
    # Create a trial_request_list object
    request_list = trial_request_list(data)
    
    # Calculate metrics
    unique_institutions = request_list.get_list_of_institutions()
    
    # Count NCT IDs
    num_nct_ids = 0
    unique_nct_ids = set()
    
    for request in data:
        nct_ids = request.get('nct_id', [])
        if isinstance(nct_ids, list):
            for nct_id in nct_ids:
                if isinstance(nct_id, str) and nct_id.startswith('NCT'):
                    num_nct_ids += 1
                    unique_nct_ids.add(nct_id)
        elif isinstance(nct_ids, str):
            if nct_ids.startswith('NCT'):
                num_nct_ids += 1
                unique_nct_ids.add(nct_ids)
    
    # Create summary
    summary = {
        'source_name': name,
        'number_of_requests': len(data),
        'number_of_institutions': len(unique_institutions),
        'number_of_nct_ids': num_nct_ids,
        'number_unique_nct_ids': len(unique_nct_ids)
    }
    
    return summary, request_list

def parse_nct_id(data):
    """
    Check if NCT ID is in the right format and fix if necessary.
    Using trial_request_list class to handle the data.
    """
    request_list = trial_request_list(data)
    
    for request in request_list.requests:
        # Handle NCT IDs in list format
        if 'nct_id' in request and isinstance(request['nct_id'], list):
            for i, nct_id in enumerate(request['nct_id']):
                if isinstance(nct_id, str) and nct_id.startswith('NCT'):
                    if not re.match(r'^NCT\d{8}$', nct_id):
                        # Fix invalid NCT ID
                        request['nct_id'][i] = nct_id[:11]
        
        # Handle NCT ID as string
        elif 'nct_id' in request and isinstance(request['nct_id'], str) and 'NCT' in request['nct_id']:
            if not re.match(r'^NCT\d{8}$', request['nct_id']):
                request['nct_id'] = request['nct_id'][:11]
    
    return request_list.requests

def generate_visualizations(csdr_list, vivli_list, yoda_list, combined_list, img_dir):
    """
    Generate various visualizations using the class methods.
    """
    print("Generating visualizations...")

    # Set seaborn style with customized font properties
    sns.set(
        style="whitegrid", 
        palette="muted", 
        font_scale=1.2,
        rc={
            "figure.figsize": (20, 10),
            "font.family": "Arial",
            "font.weight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 16
        }
    )   
    
    # Plot requests by year for each platform
    print("Plotting requests by year...")
    
    # Convert to DataFrames for customized plotting
    csdr_df = pd.DataFrame(csdr_list.requests)
    vivli_df = pd.DataFrame(vivli_list.requests)
    yoda_df = pd.DataFrame(yoda_list.requests)
    combined_df = pd.DataFrame(combined_list.requests)

    # Standardize request_year format
    for df in [csdr_df, vivli_df, yoda_df, combined_df]:
        if 'request_year' in df.columns:
            df['request_year'] = pd.to_datetime(df['request_year'], errors='coerce').dt.year

    # cummulative counts
    csdr_df_grouped_by = csdr_df.groupby('request_year').size().reset_index(name='request_id')
    vivli_df_grouped_by = vivli_df.groupby('request_year').size().reset_index(name='request_id')
    yoda_df_grouped_by = yoda_df.groupby('request_year').size().reset_index(name='request_id')

    csdr_df_grouped_by['request_id'] = csdr_df_grouped_by['request_id'].cumsum()
    vivli_df_grouped_by['request_id'] = vivli_df_grouped_by['request_id'].cumsum()
    yoda_df_grouped_by['request_id'] = yoda_df_grouped_by['request_id'].cumsum()
    
    # Plot requests by year for each platform
    sns.lineplot(data=csdr_df_grouped_by, x='request_year', y='request_id', label='CSDR', estimator='count', color = 'black', linestyle='--', markers ='o')
    sns.lineplot(data=vivli_df_grouped_by, x='request_year', y='request_id', label='Vivli', estimator='count', color = 'black', linestyle=':', markers ='o')
    sns.lineplot(data=yoda_df_grouped_by, x='request_year', y='request_id', label='YODA', estimator='count', color = 'black', linestyle='-', markers ='o')

    # Add value annotations to each point
    for x, y in zip(csdr_df_grouped_by['request_year'], csdr_df_grouped_by['request_id']):
        plt.text(x, y+5, f"{y}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    for x, y in zip(vivli_df_grouped_by['request_year'], vivli_df_grouped_by['request_id']):
        plt.text(x, y+5, f"{y}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    for x, y in zip(yoda_df_grouped_by['request_year'], yoda_df_grouped_by['request_id']):
        plt.text(x, y+5, f"{y}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    #plt.title('Trial Requests by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Requests')
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "requests_by_year.png"), dpi=300)
    plt.clf()  # Clear the current figure for the next plot

    # plot Requests by relative year
    print("Plotting requests by relative year...")
    csdr_df_grouped_by['relative_year'] = csdr_df_grouped_by['request_year'] - csdr_df_grouped_by['request_year'].min()
    vivli_df_grouped_by['relative_year'] = vivli_df_grouped_by['request_year'] - vivli_df_grouped_by['request_year'].min()
    yoda_df_grouped_by['relative_year'] = yoda_df_grouped_by['request_year'] - yoda_df_grouped_by['request_year'].min()

    sns.lineplot(data=csdr_df_grouped_by, x='relative_year', y='request_id', label='CSDR', estimator='count', color = 'black', linestyle='--', markers ='o')
    sns.lineplot(data=vivli_df_grouped_by, x='relative_year', y='request_id', label='Vivli', estimator='count', color = 'black', linestyle=':', markers ='o')
    sns.lineplot(data=yoda_df_grouped_by, x='relative_year', y='request_id', label='YODA', estimator='count', color = 'black', linestyle='-', markers ='o')

    # Add value annotations to each point
    for x, y in zip(csdr_df_grouped_by['relative_year'], csdr_df_grouped_by['request_id']):
        plt.text(x, y+5, f"{y}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for x, y in zip(vivli_df_grouped_by['relative_year'], vivli_df_grouped_by['request_id']):
        plt.text(x, y+5, f"{y}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for x, y in zip(yoda_df_grouped_by['relative_year'], yoda_df_grouped_by['request_id']):
        plt.text(x, y+5, f"{y}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.legend(title='Data Source')
    #plt.title('Trial Requests by Relative Year to First Request')
    plt.xlabel('Year')
    plt.ylabel('Number of Requests')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "requests_by_relative_year.png"), dpi=300)
    plt.clf()  # Clear the current figure for the next plot
    
    # Generate institution analysis
    print("Analyzing institutions...")
    top_institutions = pd.Series(combined_df['lead_institution'].value_counts().head(15))
    sns.barplot(x=top_institutions.values, y=top_institutions.index, color='black')

    # Add value annotations to each bar
    for index, value in enumerate(top_institutions.values):
        plt.text(value + 0.5, index, f"{value}", va='center', fontsize=10, fontweight='bold')

    #plt.title('Top 15 Lead Institutions for Trial Requests (n = {})'.format(len(combined_list.requests)))
    plt.xlabel('Number of Requests')
    plt.ylabel('Institution')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "top_institutions.png"), dpi=300)
    plt.clf()  # Clear the current figure for the next plot

    # Generate Investigator analysis
    print("Analyzing lead investigators...")
    top_investigators = pd.Series(combined_df['lead_investigator'].value_counts().head(15))
    sns.barplot(x=top_investigators.values, y=top_investigators.index, color='black')
    # Add value annotations to each bar
    for index, value in enumerate(top_investigators.values):
        plt.text(value + 0.5, index, f"{value}", va='center', fontsize=10, fontweight='bold')
    #plt.title('Top 15 Lead Investigators for Trial Requests (n = {})'.format(len(combined_list.requests)))
    plt.xlabel('Number of Requests')
    plt.ylabel('Lead Investigator')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "top_investigators.png"), dpi=300)
    plt.clf()  # Clear the current figure for the next plot
    
    print("Visualizations generated and saved to", img_dir)

def get_list_of_institutions(data):
    """
    Get a list of unique institutions from the data.
    """
    institutions = set()
    for request in data:
        institution = request.get('lead_institution', '')
        if institution:
            institutions.add(institution)
    return list(institutions)

def get_list_of_nct_ids(data):
    """
    Get a list of unique NCT IDs from the data.
    """
    nct_ids = set()
    for request in data:
        nct_id = request.get('nct_id', [])
        if isinstance(nct_id, list):
            for id in nct_id:
                if isinstance(id, str) and id.startswith('NCT'):
                    nct_ids.add(id)
        elif isinstance(nct_id, str) and nct_id.startswith('NCT'):
            nct_ids.add(nct_id)
    return list(nct_ids)

def main():
    """
    Main function to run the database analysis using trial_request_list classes.
    """
    # File paths
    csdr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "csdr_data.json")
    vivli_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "vivli_data.json")
    yoda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data", "yoda_data.json")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    summary_path = os.path.join(output_dir, "data_summary.json")

    print("Loading data...")
    # Load data 
    csdr_data = load_data(csdr_path)
    vivli_data = load_data(vivli_path)
    yoda_data = load_data(yoda_path)

    # Parse NCT IDs - only relevant for CSDR data
    print("Parsing NCT IDs...")
    csdr_data = parse_nct_id(csdr_data)  # Use existing function that works with dictionaries

    # Create trial_request_list objects
    print("Creating class objects...")
    csdr_list = trial_request_list(csdr_data)
    vivli_list = trial_request_list(vivli_data)
    yoda_list = trial_request_list(yoda_data)

    # Create combined list
    print("Creating combined data list...")
    combined_list = trial_request_list()
    
    # Add source field to each entry
    for req in csdr_data:
        req['source'] = 'CSDR'
        combined_list.requests.append(req)
    
    for req in vivli_data:
        req['source'] = 'Vivli'
        combined_list.requests.append(req)
    
    for req in yoda_data:
        req['source'] = 'YODA'
        combined_list.requests.append(req)

    # Analyze data using the class structure
    print("Analyzing data...")
    csdr_summary, _ = analyze_data_with_classes(csdr_data, "CSDR")
    vivli_summary, _ = analyze_data_with_classes(vivli_data, "Vivli")
    yoda_summary, _ = analyze_data_with_classes(yoda_data, "YODA")
    total_summary, _ = analyze_data_with_classes(combined_list.requests, "Total")

    # Combine summaries
    summary = [csdr_summary, vivli_summary, yoda_summary, total_summary]

    # Save summary data
    print("Saving summary data...")
    save_data(summary_path, summary)
    
    # Generate visualizations using the class methods
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("Generating visualizations...")
    # Create output directory for images if it doesn't exist
    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")
    os.makedirs(img_dir, exist_ok=True)

    generate_visualizations(csdr_list, vivli_list, yoda_list, combined_list, img_dir)
    
    # get the unique institutions and NCT IDs
    csdr_institutions = get_list_of_institutions(csdr_data)
    vivli_institutions = get_list_of_institutions(vivli_data)
    yoda_institutions = get_list_of_institutions(yoda_data)

    csdr_nct_ids = get_list_of_nct_ids(csdr_data)
    vivli_nct_ids = get_list_of_nct_ids(vivli_data)
    yoda_nct_ids = get_list_of_nct_ids(yoda_data)

    # combine them and save to csv
    combined_institutions = list(set(csdr_institutions + vivli_institutions + yoda_institutions))
    combined_nct_ids = list(set(csdr_nct_ids + vivli_nct_ids + yoda_nct_ids))
    institutions_df = pd.DataFrame(combined_institutions, columns=['Institution'])
    nct_ids_df = pd.DataFrame(combined_nct_ids, columns=['NCT_ID'])
    institutions_path = os.path.join(output_dir, "combined_institutions.csv")
    nct_ids_path = os.path.join(output_dir, "combined_nct_ids.csv")
    institutions_df.to_csv(institutions_path, index=False)
    nct_ids_df.to_csv(nct_ids_path, index=False)

    print("\nDatabase analysis completed!")
    print(f"Summary data saved to: {summary_path}")
    print(f"Unique institutions: CSDR ({len(csdr_list.get_list_of_institutions())}), " +
          f"Vivli ({len(vivli_list.get_list_of_institutions())}), " +
          f"YODA ({len(yoda_list.get_list_of_institutions())})")
    print(f"Total requests: {len(combined_list.requests)}")

if __name__ == "__main__":
    main()
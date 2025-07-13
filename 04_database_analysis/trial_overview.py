import numpy as np
import re
import os
import json
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

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
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14
    }
)

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
    
def trial_overview():
    """
    Create an overview of trials from the scraped data.

    result structure:
    {
    "name": "vivli",
    "total_study_id": 100,
    "total_nct_id": 80,
    "phases": {
        "Phase 1": 10,
        "Phase 2": 20,
        "Phase 3": 30,
        "Phase 4": 40} 
    }
    """

    # load data
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    vivli_file_path = os.path.join(dirc, "vivli_trials.json")
    csdr_file_path = os.path.join(dirc, "csdr_trials.json")
    yoda_file_path = os.path.join(dirc, "yoda_trials.json")

    summary_result = []

    vivli_data = load_data(vivli_file_path)
    csdr_data = load_data(csdr_file_path)
    yoda_data = load_data(yoda_file_path)

    total_trials = []

    # extend only if the trial doesnt already exist
    for trial in vivli_data + csdr_data + yoda_data:
        # Check if the trial already exists in total_trials
        if not any(existing_trial.get('study_id') == trial.get('study_id') for existing_trial in total_trials):
            total_trials.append(trial)  

    # calculate with extra def
    def calculate_overview(data, name):
        result = {}
        total_study_id = 0
        total_nct_id = 0
        phase_1 = 0
        phase_2 = 0
        phase_3 = 0
        phase_4 = 0
        phase_other = 0
        # Iterate through each trial in the data
        for trial in data:
            # Count study IDs
            study_ids = trial.get('study_id', [])
            if study_ids != '':
                total_study_id += 1
            
            # Count NCT IDs
            nct_ids = trial.get('nct_id', [])
            if nct_ids != '':
                total_nct_id += 1
            # sort and count phases
            phase = trial.get('phase', '')
            phase = str(phase)
            if phase.endswith('1'):
                phase_1 += 1
            elif phase.endswith('2'):
                phase_2 += 1
            elif phase.endswith('3'):
                phase_3 += 1
            elif phase.endswith('4'):
                phase_4 += 1
            else:
                phase_other += 1

        total_phases = phase_1 + phase_2 + phase_3 + phase_4 + phase_other

        result = {
            "name": name,
            "total_study_id": total_study_id,
            "total_nct_id": total_nct_id,
            "phases": {
                "Phase 1": phase_1,
                "Phase 2": phase_2,
                "Phase 3": phase_3,
                "Phase 4": phase_4,
                "Other Phases": phase_other,
                "total_phases": total_phases
            }
        }

        return result
    
    # Calculate overview for each data source
    vivli_overview = calculate_overview(vivli_data, "vivli")
    csdr_overview = calculate_overview(csdr_data, "csdr")
    yoda_overview = calculate_overview(yoda_data, "yoda")
    total_overview = calculate_overview(total_trials, "total")

    # Combine all overviews into a list
    summary_result.append(vivli_overview)
    summary_result.append(csdr_overview)
    summary_result.append(yoda_overview)
    summary_result.append(total_overview)

    # Convert the summary result to a DataFrame
    summary_df = pd.DataFrame(summary_result)

    # Save the summary DataFrame to a CSV file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    output_file = os.path.join(output_dir, "trial_overview.csv")
    summary_df.to_csv(output_file, index=False)

    print(f"Trial overview saved to {output_file}")

def plot_different_distributions_by_plattform():
    """
    plotting different distributions of the trials
    better knowledge about trial characteristics
    phase, conditions, orgName, uploads over year
    """
    # load data
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    vivli_file_path = os.path.join(dirc, "vivli_trials.json")
    csdr_file_path = os.path.join(dirc, "csdr_trials.json")
    yoda_file_path = os.path.join(dirc, "yoda_trials.json")

    vivli_data = load_data(vivli_file_path)
    csdr_data = load_data(csdr_file_path)
    yoda_data = load_data(yoda_file_path)

    # plot different distributions
    # phase distribution
    def phase_plot(data, name):
        phase_1 = 0
        phase_2 = 0
        phase_3 = 0
        phase_4 = 0
        phase_other = 0

        # Iterate through each trial in the data
        for trial in data:
            # sort and count phases
            phase = trial.get('phase', '')
            phase = str(phase)
            if phase.endswith('1'):
                phase_1 += 1
            elif phase.endswith('2'):
                phase_2 += 1
            elif phase.endswith('3'):
                phase_3 += 1
            elif phase.endswith('4'):
                phase_4 += 1
            else:
                phase_other += 1

        # plot the phases
        phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Other Phases']
        counts = [phase_1, phase_2, phase_3, phase_4, phase_other]

        image_dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")

        plt.figure(figsize=(20, 10))
        sns.barplot(x=phases, y=counts)
        plt.title(f'Phase Distribution in {name} Trials')
        plt.xlabel('Phase')
        plt.ylabel('Number of Trials')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dirc, f'phase/{name}_phase_distribution.png'))

    def plot_conditions(data, name):
        """
        Plot the distribution of conditions in the trials.
        """
        conditions = []
        for trial in data:
            if 'conditions' in trial and isinstance(trial['conditions'], list):
                conditions.extend(trial['conditions'])

        # Count occurrences of each condition
        condition_counts = pd.Series(conditions).value_counts()

        # Plot the top 10 conditions
        top_conditions = condition_counts.head(10)

        image_dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")

        plt.figure(figsize=(20, 10))
        sns.barplot(x=top_conditions.index, y=top_conditions.values)
        plt.title(f'Top 10 Conditions in {name} Trials')
        plt.xlabel('Condition')
        plt.ylabel('Number of Trials')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dirc, f'conditions/{name}_conditions_distribution.png'))

    def plot_org_name(data, name):
        """
        Plot the distribution of organizations in the trials.
        """
        org_names = []
        for trial in data:
            if 'orgName' in trial and isinstance(trial['orgName'], str):
                org_names.append(trial['orgName'])

        # Count occurrences of each organization
        org_counts = pd.Series(org_names).value_counts()

        # Plot the top 10 organizations
        top_orgs = org_counts.head(10)

        image_dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")

        plt.figure(figsize=(20, 10))
        sns.barplot(x=top_orgs.index, y=top_orgs.values)
        plt.title(f'Top 10 Organizations in {name} Trials')
        plt.xlabel('Organization')
        plt.ylabel('Number of Trials')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dirc, f'orgName/{name}_orgName_distribution.png'))

    def plot_uploads_over_time(data, name):
        """
        Plot the number of uploads over the years.
        """
        upload_dates = []
        for trial in data:
            if 'postedDate' in trial and isinstance(trial['postedDate'], str):
                try:
                    date = pd.to_datetime(trial['postedDate'])
                    upload_dates.append(date)
                except ValueError:
                    continue
        
        # Create a DataFrame with the upload dates
        upload_df = pd.DataFrame(upload_dates, columns=['upload_date'])
        upload_df['year'] = upload_df['upload_date'].dt.year
        upload_df['year_month'] = upload_df['upload_date'].dt.to_period('M')
        upload_counts = upload_df['year_month'].value_counts().sort_index()
        upload_counts.index = upload_counts.index.astype(str)
        # Plot the upload counts over time
        image_dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")

        plt.figure(figsize=(20, 10))
        sns.lineplot(x=upload_counts.index, y=upload_counts.values, marker='o')
        plt.title(f'Uploads Over Time in {name} Trials')
        plt.xlabel('Year-Month')
        plt.ylabel('Number of Uploads')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dirc, f'uploads_over_time/{name}_uploads_over_time.png'))

    # Plot distributions for each data source
    phase_plot(vivli_data, "Vivli")
    phase_plot(csdr_data, "CSDR")
    phase_plot(yoda_data, "YODA")

    plot_conditions(vivli_data, "Vivli")
    plot_conditions(csdr_data, "CSDR")
    plot_conditions(yoda_data, "YODA")
    
    plot_org_name(vivli_data, "Vivli")
    plot_org_name(csdr_data, "CSDR")
    plot_org_name(yoda_data, "YODA")

    plot_uploads_over_time(vivli_data, "Vivli")
    plot_uploads_over_time(csdr_data, "CSDR")
    plot_uploads_over_time(yoda_data, "YODA")

    print("Distributions plotted and saved as images.")

def plot_distributions():
    """
    Plot different distributions of the trials in one plot
    better knowledge about trial characteristics
    phase, conditions, orgName, uploads over year
    """
    # load data
    dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_data")
    vivli_file_path = os.path.join(dirc, "vivli_trials.json")
    csdr_file_path = os.path.join(dirc, "csdr_trials.json")
    yoda_file_path = os.path.join(dirc, "yoda_trials.json")
    image_dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")

    vivli_data = load_data(vivli_file_path)
    csdr_data = load_data(csdr_file_path)
    yoda_data = load_data(yoda_file_path)

    def count_phases(data):
        """
        Count the number of trials in each phase.
        """
        phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Other Phases']
        counts = {phase: 0 for phase in phases}

        for trial in data:
            phase = trial.get('phase', '')
            phase = str(phase)
            if phase.endswith('1'):
                counts['Phase 1'] += 1
            elif phase.endswith('2'):
                counts['Phase 2'] += 1
            elif phase.endswith('3'):
                counts['Phase 3'] += 1
            elif phase.endswith('4'):
                counts['Phase 4'] += 1
            else:
                counts['Other Phases'] += 1

        return counts

    # count phases for each data source
    vivli_phases = count_phases(vivli_data)
    csdr_phases = count_phases(csdr_data)
    yoda_phases = count_phases(yoda_data)

    # Create a DataFrame for plotting
    phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Other Phases']
    data = {
        'Phase': phases,
        'Vivli': [vivli_phases[phase] for phase in phases],
        'CSDR': [csdr_phases[phase] for phase in phases],
        'YODA': [yoda_phases[phase] for phase in phases]
    }

    df = pd.DataFrame(data)

    # plot the data
    image_dirc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_images")
    plt.figure(figsize=(20, 10))
    sns.barplot(
        data=df.melt(id_vars='Phase', var_name='Data Source', value_name='Count'),
        x='Phase',
        y='Count',
        hue='Data Source',
        linewidth=1.5,
        edgecolor='black'
    )
    plt.title('Trial Phases Distribution by Data Source')
    plt.xlabel('Phase')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dirc, 'trial_phases_distribution.png'))

    # plot conditions
    # count conditions for each data source
    def count_conditions(data):
        """
        Count the occurrences of each condition in the trials.
        """
        conditions = []
        for trial in data:
            if 'conditions' in trial and isinstance(trial['conditions'], list):
                conditions.extend(trial['conditions'])

        # Count occurrences of each condition
        condition_counts = pd.Series(conditions).value_counts().head(10)  # Get top 10 conditions
        return condition_counts    

    vivli_conditions = count_conditions(vivli_data)
    csdr_conditions = count_conditions(csdr_data)
    yoda_conditions = count_conditions(yoda_data)
    # Create a DataFrame for plotting
    conditions = list(set(vivli_conditions.index) | set(csdr_conditions.index) | set(yoda_conditions.index))
    data = {
        'Condition': conditions,
        'Vivli': [vivli_conditions.get(cond, 0) for cond in conditions],
        'CSDR': [csdr_conditions.get(cond, 0) for cond in conditions],
        'YODA': [yoda_conditions.get(cond, 0) for cond in conditions]
    }
    df_conditions = pd.DataFrame(data)
    # plot the data
    plt.figure(figsize=(20, 10))
    sns.barplot(
        data=df_conditions.melt(id_vars='Condition', var_name='Data Source', value_name='Count'),
        x='Condition',
        y='Count',
        hue='Data Source',
        linewidth=1.5,
        edgecolor='black'
    )

    plt.title('Conditions Distribution by Data Source')
    plt.xlabel('Condition')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dirc, 'conditions_distribution.png'))

    # plot orgName
    # count orgName for each data source
    def count_org_name(data):
        """
        Count the occurrences of each organization in the trials.
        """
        org_names = []
        for trial in data:
            if 'orgName' in trial and isinstance(trial['orgName'], str):
                org_names.append(trial['orgName'])

        # Count occurrences of each organization
        org_counts = pd.Series(org_names).value_counts().head(10)  # Get top 10 organizations
        return org_counts
    
    vivli_orgs = count_org_name(vivli_data)
    csdr_orgs = count_org_name(csdr_data)
    yoda_orgs = count_org_name(yoda_data)

    # Create a DataFrame for plotting
    orgs = list(set(vivli_orgs.index) | set(csdr_orgs.index) | set(yoda_orgs.index))
    data = {
        'Organization': orgs,
        'Vivli': [vivli_orgs.get(org, 0) for org in orgs],
        'CSDR': [csdr_orgs.get(org, 0) for org in orgs],
        'YODA': [yoda_orgs.get(org, 0) for org in orgs]
    }

    df_orgs = pd.DataFrame(data)
    # plot the data
    plt.figure(figsize=(20, 10))
    sns.barplot(
        data=df_orgs.melt(id_vars='Organization', var_name='Data Source', value_name='Count'),
        x='Organization',
        y='Count',
        hue='Data Source',
        linewidth=1.5,
        edgecolor='black'
    )

    plt.title('Organizations Distribution by Data Source')
    plt.xlabel('Organization')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dirc, 'org_name_distribution.png'))

    # plot uploads over time
    
    def request_time_series_analyze(data):
        """
        Time series of Request with improved year handling
        """
        rows = []
        for entry in data:
            nct_id = entry.get('nct_id')
            # Extract postedDate from the entry
            posted_date = entry.get('postedDate')
            
            # Only process entries with valid postedDate
            if posted_date and isinstance(posted_date, str):
                try:
                    # Convert string date to datetime object
                    date = pd.to_datetime(posted_date)
                    # Extract year
                    year = date.year
                    rows.append({'nct_id': nct_id, 'year': year})
                except ValueError:
                    # Skip entries with invalid dates
                    continue

        # Create a DataFrame
        df = pd.DataFrame(rows)
        
        # Make sure we have data
        if df.empty:
            print(f"Warning: No valid year data found")
            return pd.DataFrame({'relative_year': [], 'request_count': []}), pd.DataFrame({'year': [], 'request_count': []})
        
        print(f"unique years after conversion: {sorted(df['year'].unique())}")

        # Create relative year
        min_year = df['year'].min()
        df['relative_year'] = df['year'] - min_year + 1  # Start from 1

        # Group by relative year and count requests
        df_grouped_relative = df.groupby('relative_year').agg({'nct_id': 'count'}).reset_index()
        df_grouped_relative.rename(columns={'nct_id': 'request_count'}, inplace=True)

        # Group by year and count requests
        df_grouped = df.groupby('year').agg({'nct_id': 'count'}).reset_index()
        df_grouped.rename(columns={'nct_id': 'request_count'}, inplace=True)

        return df_grouped_relative, df_grouped
    
    # plot uploads over time
    vivli_relative, vivli_grouped = request_time_series_analyze(vivli_data)
    csdr_relative, csdr_grouped = request_time_series_analyze(csdr_data)
    yoda_relative, yoda_grouped = request_time_series_analyze(yoda_data)

    # cumulative sum for relative years and grouped years
    vivli_relative['cumulative_requests'] = vivli_relative['request_count'].cumsum()
    csdr_relative['cumulative_requests'] = csdr_relative['request_count'].cumsum()
    yoda_relative['cumulative_requests'] = yoda_relative['request_count'].cumsum()
    vivli_grouped['cumulative_requests'] = vivli_grouped['request_count'].cumsum()
    csdr_grouped['cumulative_requests'] = csdr_grouped['request_count'].cumsum()
    yoda_grouped['cumulative_requests'] = yoda_grouped['request_count'].cumsum()
    
    # plot relative years
    # relative years
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=vivli_relative, x='relative_year', y='cumulative_requests', label='Vivli', marker='o', markersize=4, color='black')
    sns.lineplot(data=csdr_relative, x='relative_year', y='cumulative_requests', label='CSDR', marker='o', markersize=4, linestyle='--', color='black')
    sns.lineplot(data=yoda_relative, x='relative_year', y='cumulative_requests', label='YODA', marker='o', markersize=4, linestyle=':', color='black')

    plt.title('Uploads per Years')
    plt.xlabel('Relative Year')
    plt.ylabel('Uploads')
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dirc, 'uploads_relative_years.png'))

    # plot grouped years
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=vivli_grouped, x='year', y='cumulative_requests', label='Vivli', marker='o', markersize=4, color='black')
    sns.lineplot(data=csdr_grouped, x='year', y='cumulative_requests', label='CSDR', marker='o', markersize=4, linestyle='--', color='black')
    sns.lineplot(data=yoda_grouped, x='year', y='cumulative_requests', label='YODA', marker='o', markersize=4, linestyle=':', color='black')

    plt.title('Uploads per Year')
    plt.xlabel('Year')
    plt.ylabel('Uploads')
    plt.xticks(rotation=45)
    plt.legend(title='Data Source')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dirc, 'uploads_years.png'))


    
    
if __name__ == "__main__":
    #trial_overview()
    #print("Trial overview completed.")

    #plot_different_distributions_by_plattform()
    #print("Different distributions plotted by platform.")

    plot_distributions()
    print("Distributions plotted in one plot.")



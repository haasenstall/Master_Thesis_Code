# packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import json
import networkx as nx
import geopandas as gpd

class trial:
    def __init__(self, nct_id, title, start_date, end_date, verification_date, upload_date, status, phase, condition, condition_tree_number, intervention, intervention_tree_number, sponsor, country, requests = None, request_rate = None):
        self.nct_id = nct_id
        self.title = title
        self.start_date = start_date
        self.end_date = end_date
        self.verification_date = verification_date
        self.upload_date = upload_date
        self.status = status
        self.phase = phase
        self.condition = condition
        self.condition_tree_number = condition_tree_number
        self.intervention = intervention
        self.intervention_tree_number = intervention_tree_number
        self.sponsor = sponsor
        self.country = country
        self.requests = requests
        self.request_rate = request_rate

    def __str__(self):
        return f"Trial({self.nct_id}, {self.title}, {self.start_date}, {self.end_date}, {self.verification_date}, {self.status}, {self.phase}, {self.condition}, {self.condition_tree_number}, {self.intervention}, {self.intervention_tree_number}, {self.sponsor}, {self.country})"

    def __repr__(self):
        return f"Trial(nct_id={self.nct_id}, title={self.title}, start_date={self.start_date}, end_date={self.end_date}, verification_date={self.verification_date}, status={self.status}, phase={self.phase}, condition={self.condition}, condition_tree_number={self.condition_tree_number}, intervention={self.intervention}, intervention_tree_number={self.intervention_tree_number}, sponsor={self.sponsor}, country={self.country})"

    def build(self):
        data = {
            "nct_id": self.nct_id,
            "title": self.title,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "verification_date": self.verification_date,
            "status": self.status,
            "phase": self.phase,
            "condition":{
                "name": self.condition,
                "tree_number": self.condition_tree_number
            },
            "intervention":{
                "name": self.intervention,
                "tree_number": self.intervention_tree_number
            },
            "sponsor": self.sponsor,
            "country": self.country,
            "requests": self.requests,
            "request_rate": self.request_rate
        }
        return data
    
    def add_values(self, nct_id, title, start_date, end_date, verification_date, upload_date, status, phase, condition, condition_tree_number, intervention, intervention_tree_number, sponsor, country):
        self.nct_id = nct_id
        self.title = title
        self.start_date = start_date
        self.end_date = end_date
        self.verification_date = verification_date
        self.upload_date = upload_date
        self.status = status
        self.phase = phase
        self.condition = condition
        self.condition_tree_number = condition_tree_number
        self.intervention = intervention
        self.intervention_tree_number = intervention_tree_number
        self.sponsor = sponsor
        self.country = country
        
class trial_list:
    def __init__(self, trials=None):
        if trials is None:
            self.trials = []
        else:
            self.trials = trials

    def add_trial(self, trial):
        self.trials.append(trial.build())

    def remove_trial(self, nct_id):
        self.trials = [trial for trial in self.trials if trial['nct_id'] != nct_id]

    def get_trial(self, nct_id):
        for trial in self.trials:
            if trial['nct_id'] == nct_id:
                return trial
        return None
    
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.trials, f, indent=4)

    def load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.trials = json.load(f)
    
    def get_trials_by_phase(self, phase):
        return [trial for trial in self.trials if trial['phase'] == phase]
    
    def get_trials_by_condition(self, condition):
        return [trial for trial in self.trials if trial['condition']['name'] == condition]
    
    def get_trials_by_sponsor(self, sponsor):
        return [trial for trial in self.trials if trial['sponsor'] == sponsor]
    
    def get_trials_by_location(self, location):
        return [trial for trial in self.trials if trial['location'] == location]

    def __str__(self):
        return f"TrialList with {len(self.trials)} trials"
    
    def __repr__(self):
        return f"TrialList(trials={self.trials})"

    def plot_phase_distribution(self):
        if not self.trials:
            print("No trials to plot.")
            return
        
        
        df = pd.DataFrame(self.trials)
        sns.set(style="whitegrid")
        
        # plot phases
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='phase', order=df['phase'].value_counts().index)
        plt.title('Distribution of Trial Phases')
        plt.xlabel('Phase')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_condition_network(self):
        """
        Visualize the relationship between conditions by using a network graph.
        Conditions have a tree number (e.g. A01.123.123) which can be used to build a network graph.
        """

        if not self.trials:
            print("No trials to plot.")
            return
        
        G = nx.Graph()
        
        for trial in self.trials:
            condition = trial['condition']['name']
            tree_number = trial['condition']['tree_number']
            G.add_node(condition, tree_number=tree_number)
            
            # Add edges based on tree number similarity
            for other_trial in self.trials:
                if trial != other_trial:
                    other_condition = other_trial['condition']['name']
                    other_tree_number = other_trial['condition']['tree_number']
                    if tree_number.startswith(other_tree_number) or other_tree_number.startswith(tree_number):
                        G.add_edge(condition, other_condition)
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color=range(len(G.nodes())), cmap=plt.cm.Blues, font_size=10, font_color='black', edge_color='gray')
        plt.title('Condition Network Graph')
        plt.show()

    def plot_intervention_network(self):
        """
        Visualize the relationship between interventions by using a network graph.
        Interventions have a tree number (e.g. B01.123.123) which can be used to build a network graph.
        """
        if not self.trials:
            print("No trials to plot.")
            return
        
        G = nx.Graph()
        
        for trial in self.trials:
            intervention = trial['intervention']['name']
            tree_number = trial['intervention']['tree_number']
            G.add_node(intervention, tree_number=tree_number)
            
            # Add edges based on tree number similarity
            for other_trial in self.trials:
                if trial != other_trial:
                    other_intervention = other_trial['intervention']['name']
                    other_tree_number = other_trial['intervention']['tree_number']
                    if tree_number.startswith(other_tree_number) or other_tree_number.startswith(tree_number):
                        G.add_edge(intervention, other_intervention)
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color=range(len(G.nodes())), cmap=plt.cm.Blues, font_size=10, font_color='black', edge_color='gray')
        plt.title('Intervention Network Graph')
        plt.show()

    def plot_sponsor_network(self):
        """
        Visualize the relationship between sponsors by using a network graph.
        Sponsors can be connected if they have trials with similar conditions or interventions.
        """
        if not self.trials:
            print("No trials to plot.")
            return
        
        G = nx.Graph()
        
        for trial in self.trials:
            sponsor = trial['sponsor']
            G.add_node(sponsor)
            
            # Add edges based on shared conditions or interventions
            for other_trial in self.trials:
                if trial != other_trial:
                    other_sponsor = other_trial['sponsor']
                    if trial['condition']['name'] == other_trial['condition']['name'] or trial['intervention']['name'] == other_trial['intervention']['name']:
                        G.add_edge(sponsor, other_sponsor)
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color=range(len(G.nodes())), cmap=plt.cm.Blues, font_size=10, font_color='black', edge_color='gray')
        plt.title('Sponsor Network Graph')
        plt.show()

    def plot_country_network(self):
        """
        Visulize the locations of trials over world map.
        Country of trials is used to plot the trials on a world map.
        for more trials by one country, the filling of the country is darker.
        """
        if not self.trials:
            print("No trials to plot.")
            return
        
        # Load world map    
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        # Create a DataFrame for the trials
        df = pd.DataFrame(self.trials)
        df = df.groupby('country').size().reset_index(name='count')

        # Merge with world map
        world = world.merge(df, how="left", left_on="name", right_on="country")
        world["count"] = world["count"].fillna(0)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.boundary.plot(ax=ax, linewidth=1)
        world.plot(column="count", ax=ax, legend=True,
                   legend_kwds={"label": "Number of Trials",
                                "orientation": "horizontal"},
                   missing_kwds={"color": "lightgrey"},
                   cmap="Blues", edgecolor='black')
        plt.title('Trial Locations by Country')
        plt.show()

    def plot_request_distribution(self):
        """
        Visualize the distribution of trial requests by trial.
        """
        if not self.trials:
            print("No trials to plot.")
            return
        
        df = pd.DataFrame(self.trials)
        sns.set(style="whitegrid")

        # Count requests per trial
        df['request_counts'] = df['requests'].apply(lambda x: len(x) if x else 0)
        # group by request counts
        request_counts = df['request_counts'].value_counts().reset_index()
        request_counts.columns = ['request_count', 'number_of_trials']

        plt.figure(figsize=(12, 6))
        sns.barplot(data=request_counts, x='request_count', y='number_of_trials', palette='viridis')
        plt.title('Distribution of Trial Requests')
        plt.xlabel('Number of Requests')
        plt.ylabel('Number of Trials')
        plt.show()

    def get_highest_requested_trials(self, top_n=10):
        """
        Get the top N trials with the highest number of requests.
        """
        if not self.trials:
            print("No trials to analyze.")
            return []
        
        df = pd.DataFrame(self.trials)
        df['request_counts'] = df['requests'].apply(lambda x: len(x) if x else 0)
        top_trials = df.nlargest(top_n, 'request_counts')
        return top_trials[['nct_id', 'title', 'request_counts']].to_dict(orient='records')

class trial_request:

    def __init__(self, request_id, title, nct_id, request_year, lead_investigator, lead_institution, country = None):
        self.request_id = request_id
        self.title = title
        self.nct_id = nct_id
        self.request_year = request_year
        self.lead_investigator = lead_investigator
        self.lead_institution = lead_institution
        self.country = country

    def __str__(self):
        return f"TrialRequest({self.request_id}, {self.nct_id}, {self.request_year}, {self.lead_investigator}, {self.lead_institution}, {self.country})"
    
    def __repr__(self):
        return f"TrialRequest(request_id={self.request_id}, nct_id={self.nct_id}, request_year={self.request_year}, lead_investigator={self.lead_investigator}, lead_institution={self.lead_institution}, country={self.country})"
    
    def build(self):
        data = {
            "request_id": self.request_id,
            "title": self.title,
            "nct_id": self.nct_id,
            "request_year": self.request_year,
            "lead_investigator": self.lead_investigator,
            "lead_institution": self.lead_institution,
            "country": self.country
        }
        return data
    
    def get_trial_data(self, trial_list):
        """
        Get the trial data for the request from the trial list.
        """
        for trial in trial_list.trials:
            if trial['nct_id'] == self.nct_id:
                return trial
        return None
    
    def clean_nct_id(self):
        """
        Clean the NCT ID by removing any leading or trailing whitespace.
        """
        if isinstance(self.nct_id, list):
            for nct in self.nct_id:
                if nct.startswith("NCT"):
                    # just the first 8 characters NCT12345678
                    self.nct_id = nct.strip()
                    nct = nct[:11]
                elif not nct.startswith("NCT"):
                    # if the NCT ID not starts with "NCT", we can assume it's not in ct.gov included
                    continue
        elif isinstance(self.nct_id, str):
            if self.nct_id.startswith("NCT"):
                # just the first 8 characters NCT12345678
                self.nct_id = self.nct_id.strip()[:11]
            elif not self.nct_id.startswith("NCT"):
                # If the NCT ID does not start with "NCT", we can assume it's invalid
                print(f"Invalid NCT ID: {self.nct_id}. It should start with 'NCT'.")
                self.nct_id = None
        return self.nct_id
    
class trial_request_list:
    def __init__(self, requests=None):
        if requests is None:
            self.requests = []
        else:
            self.requests = requests

    def add_request(self, request):
        self.requests.append(request.build())

    def remove_request(self, request_id):
        self.requests = [request for request in self.requests if request['request_id'] != request_id]

    def get_request(self, request_id):
        for request in self.requests:
            if request['request_id'] == request_id:
                return request
        return None
    
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.requests, f, indent=4)

    def load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.requests = json.load(f)
    
    def __str__(self):
        return f"TrialRequestList with {len(self.requests)} requests"
    
    def __repr__(self):
        return f"TrialRequestList(requests={self.requests})"
    
    def get_list_of_institutions(self):
        """
        Get a list of unique institutions from the requests.
        """
        institutions = set()
        for request in self.requests:
            institutions.add(request['lead_institution'])
        return list(institutions)
    
    def get_list_of_sponsors(self):
        """
        Get a list of unique sponsors from the requests.
        """
        sponsors = set()
        for request in self.requests:
            sponsors.add(request['sponsor'])
        return list(sponsors)
    
    def plot_request_by_year(self):
        """
        Plot the number of requests by year
        """
        if not self.requests:
            print("No requests to plot.")
            return
        
        df = pd.DataFrame(self.requests)
        sns.set(style="whitegrid")
        
        # Count requests per year
        df['request_year'] = pd.to_datetime(df['request_year'], errors='coerce').dt.year
        df = df.dropna(subset=['request_year'])
        
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='request_year', order=df['request_year'].value_counts().index)
        plt.title('Distribution of Trial Requests by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_requests_by_nct_id(self):
        """
        plot the number of nct_ids per request
        number of request with different nct_ids
        1. request with one nct_id
        2. request with two nct_ids
        3. request with three nct_ids
        etc.
        """
        if not self.requests:
            print("No requests to plot.")
            return
        

        df = pd.DataFrame(self.requests)
        sns.set(style="whitegrid")
        
        # count nct_ids per request
        df['nct_id_count'] = df['nct_id'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='nct_id_count', order=df['nct_id_count'].value_counts().index)
        plt.title('Distribution of Requests by NCT ID Count')
        plt.xlabel('NCT ID Count')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_requests_by_sponsor(self):
        """
        Plot the number of requests by sponsor
        """
        if not self.requests:
            print("No requests to plot.")
            return
        
        df = pd.DataFrame(self.requests)
        sns.set(style="whitegrid")
        
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='sponsor', order=df['sponsor'].value_counts().index)
        plt.title('Distribution of Trial Requests by Sponsor')
        plt.xlabel('Count')
        plt.ylabel('Sponsor')
        plt.tight_layout()
        plt.show()

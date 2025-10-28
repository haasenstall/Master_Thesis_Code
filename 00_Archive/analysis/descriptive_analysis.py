# analysis/descriptive_analysis.py

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from db.sqlalchemy_connector import DatabaseManager
from config import DATA_PATHS


class DescriptiveAnalyzer:
    """Performs descriptive statistical analysis of clinical trial request data"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.db_manager.connect_local()
        self.output_dir = Path(DATA_PATHS['image_path'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant data for analysis"""
        self.logger.info("Loading data for descriptive analysis...")
        
        queries = {
            'requests': """
                SELECT r.*, p.name as platform_name
                FROM requests r
                JOIN platforms p ON r.platform_id = p.id
            """,
            
            'request_trials': """
                SELECT r.request_id, r.platform_id, p.name as platform_name,
                       rtl.nct_id, ct.title, ct.status, ct.phase, ct.study_type,
                       ct.enrollment, ct.start_date, ct.completion_date
                FROM requests r
                JOIN platforms p ON r.platform_id = p.id
                LEFT JOIN request_trial_links rtl ON r.request_id = rtl.request_id
                LEFT JOIN clinical_trials ct ON rtl.nct_id = ct.nct_id
            """,
            
            'platform_summary': """
                SELECT p.name, COUNT(r.request_id) as total_requests
                FROM platforms p
                LEFT JOIN requests r ON p.id = r.platform_id
                GROUP BY p.id, p.name
                ORDER BY total_requests DESC
            """
        }
        
        data = {}
        for name, query in queries.items():
            try:
                data[name] = self.db_manager.local_db.query_to_dataframe(query)
                self.logger.info(f"Loaded {name}: {len(data[name])} rows")
            except Exception as e:
                self.logger.error(f"Error loading {name}: {e}")
                data[name] = pd.DataFrame()
        
        return data
    
    def analyze_platform_distribution(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze distribution of requests across platforms"""
        self.logger.info("Analyzing platform distribution...")
        
        requests_df = data['requests']
        
        if requests_df.empty:
            self.logger.warning("No request data available")
            return {}
        
        # Platform distribution
        platform_counts = requests_df['platform_name'].value_counts()
        platform_percentages = requests_df['platform_name'].value_counts(normalize=True) * 100
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        platform_counts.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Number of Requests by Platform')
        ax1.set_xlabel('Platform')
        ax1.set_ylabel('Number of Requests')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        platform_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Requests by Platform')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'platform_counts': platform_counts.to_dict(),
            'platform_percentages': platform_percentages.to_dict()
        }
    
    def analyze_temporal_patterns(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze temporal patterns in request submissions and trial characteristics"""
        self.logger.info("Analyzing temporal patterns...")
        
        request_trials_df = data['request_trials']
        
        if request_trials_df.empty:
            self.logger.warning("No request-trial data available")
            return {}
        
        # Convert date columns
        date_columns = ['start_date', 'completion_date']
        for col in date_columns:
            if col in request_trials_df.columns:
                request_trials_df[col] = pd.to_datetime(request_trials_df[col], errors='coerce')
        
        # Analyze start dates
        if 'start_date' in request_trials_df.columns:
            start_dates = request_trials_df['start_date'].dropna()
            
            if not start_dates.empty:
                # Extract year and create time series
                start_years = start_dates.dt.year
                year_counts = start_years.value_counts().sort_index()
                
                # Plot temporal trends
                plt.figure(figsize=(12, 6))
                year_counts.plot(kind='line', marker='o')
                plt.title('Clinical Trials by Start Year (From Requested Trials)')
                plt.xlabel('Year')
                plt.ylabel('Number of Trials')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                return {
                    'year_distribution': year_counts.to_dict(),
                    'earliest_year': int(start_years.min()),
                    'latest_year': int(start_years.max())
                }
        
        return {}
    
    def analyze_trial_characteristics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze characteristics of requested trials"""
        self.logger.info("Analyzing trial characteristics...")
        
        request_trials_df = data['request_trials']
        
        if request_trials_df.empty:
            self.logger.warning("No request-trial data available")
            return {}
        
        # Remove rows where nct_id is null (requests without specific trials)
        trials_df = request_trials_df.dropna(subset=['nct_id'])
        
        if trials_df.empty:
            self.logger.warning("No trial data available")
            return {}
        
        analysis_results = {}
        
        # Study type distribution
        if 'study_type' in trials_df.columns:
            study_type_counts = trials_df['study_type'].value_counts()
            analysis_results['study_types'] = study_type_counts.to_dict()
            
            # Visualization
            plt.figure(figsize=(10, 6))
            study_type_counts.plot(kind='bar')
            plt.title('Distribution of Study Types in Requested Trials')
            plt.xlabel('Study Type')
            plt.ylabel('Number of Trials')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'study_types.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Phase distribution
        if 'phase' in trials_df.columns:
            phase_counts = trials_df['phase'].value_counts()
            analysis_results['phases'] = phase_counts.to_dict()
            
            # Visualization
            plt.figure(figsize=(10, 6))
            phase_counts.plot(kind='bar')
            plt.title('Distribution of Trial Phases in Requested Trials')
            plt.xlabel('Phase')
            plt.ylabel('Number of Trials')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'trial_phases.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Status distribution
        if 'status' in trials_df.columns:
            status_counts = trials_df['status'].value_counts()
            analysis_results['status'] = status_counts.to_dict()
            
            # Visualization
            plt.figure(figsize=(12, 6))
            status_counts.plot(kind='bar')
            plt.title('Distribution of Trial Status in Requested Trials')
            plt.xlabel('Status')
            plt.ylabel('Number of Trials')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'trial_status.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Enrollment analysis
        if 'enrollment' in trials_df.columns:
            enrollment_data = trials_df['enrollment'].dropna()
            if not enrollment_data.empty:
                analysis_results['enrollment_stats'] = {
                    'mean': float(enrollment_data.mean()),
                    'median': float(enrollment_data.median()),
                    'std': float(enrollment_data.std()),
                    'min': int(enrollment_data.min()),
                    'max': int(enrollment_data.max())
                }
                
                # Enrollment distribution plot
                plt.figure(figsize=(10, 6))
                plt.hist(enrollment_data, bins=50, alpha=0.7, edgecolor='black')
                plt.title('Distribution of Trial Enrollment Numbers')
                plt.xlabel('Enrollment Count')
                plt.ylabel('Frequency')
                plt.axvline(enrollment_data.mean(), color='red', linestyle='--', label=f'Mean: {enrollment_data.mean():.0f}')
                plt.axvline(enrollment_data.median(), color='orange', linestyle='--', label=f'Median: {enrollment_data.median():.0f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.output_dir / 'enrollment_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return analysis_results
    
    def analyze_request_characteristics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze characteristics of the requests themselves"""
        self.logger.info("Analyzing request characteristics...")
        
        requests_df = data['requests']
        
        if requests_df.empty:
            self.logger.warning("No request data available")
            return {}
        
        analysis_results = {}
        
        # Number of trials per request
        request_trial_counts = data['request_trials'].groupby('request_id').size()
        
        if not request_trial_counts.empty:
            analysis_results['trials_per_request_stats'] = {
                'mean': float(request_trial_counts.mean()),
                'median': float(request_trial_counts.median()),
                'std': float(request_trial_counts.std()),
                'min': int(request_trial_counts.min()),
                'max': int(request_trial_counts.max())
            }
            
            # Distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(request_trial_counts, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Number of Trials per Request')
            plt.xlabel('Number of Trials')
            plt.ylabel('Number of Requests')
            plt.axvline(request_trial_counts.mean(), color='red', linestyle='--', 
                       label=f'Mean: {request_trial_counts.mean():.1f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'trials_per_request.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Analyze text fields if available
        text_fields = ['title', 'therapeutic_area', 'conditions_studied', 'intervention_studied']
        
        for field in text_fields:
            if field in requests_df.columns:
                field_data = requests_df[field].dropna()
                if not field_data.empty:
                    # Basic text statistics
                    field_lengths = field_data.str.len()
                    analysis_results[f'{field}_stats'] = {
                        'count': len(field_data),
                        'mean_length': float(field_lengths.mean()),
                        'median_length': float(field_lengths.median()),
                        'max_length': int(field_lengths.max()),
                        'min_length': int(field_lengths.min())
                    }
        
        return analysis_results
    
    def generate_summary_report(self, all_results: Dict) -> str:
        """Generate a comprehensive summary report"""
        self.logger.info("Generating summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("DESCRIPTIVE ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Platform analysis
        if 'platform_distribution' in all_results:
            platform_data = all_results['platform_distribution']
            report.append("PLATFORM DISTRIBUTION:")
            report.append("-" * 40)
            for platform, count in platform_data.get('platform_counts', {}).items():
                percentage = platform_data.get('platform_percentages', {}).get(platform, 0)
                report.append(f"  {platform}: {count} requests ({percentage:.1f}%)")
            report.append("")
        
        # Trial characteristics
        if 'trial_characteristics' in all_results:
            trial_data = all_results['trial_characteristics']
            report.append("TRIAL CHARACTERISTICS:")
            report.append("-" * 40)
            
            if 'study_types' in trial_data:
                report.append("  Study Types:")
                for study_type, count in trial_data['study_types'].items():
                    report.append(f"    {study_type}: {count}")
                report.append("")
            
            if 'enrollment_stats' in trial_data:
                enrollment = trial_data['enrollment_stats']
                report.append("  Enrollment Statistics:")
                report.append(f"    Mean: {enrollment['mean']:.0f}")
                report.append(f"    Median: {enrollment['median']:.0f}")
                report.append(f"    Range: {enrollment['min']} - {enrollment['max']}")
                report.append("")
        
        # Request characteristics
        if 'request_characteristics' in all_results:
            request_data = all_results['request_characteristics']
            if 'trials_per_request_stats' in request_data:
                stats = request_data['trials_per_request_stats']
                report.append("TRIALS PER REQUEST:")
                report.append("-" * 40)
                report.append(f"  Mean: {stats['mean']:.1f}")
                report.append(f"  Median: {stats['median']:.1f}")
                report.append(f"  Range: {stats['min']} - {stats['max']}")
                report.append("")
        
        # Temporal patterns
        if 'temporal_patterns' in all_results:
            temporal_data = all_results['temporal_patterns']
            if 'earliest_year' in temporal_data:
                report.append("TEMPORAL PATTERNS:")
                report.append("-" * 40)
                report.append(f"  Trial years range: {temporal_data['earliest_year']} - {temporal_data['latest_year']}")
                report.append("")
        
        report.append("Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'descriptive_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Report saved to: {report_path}")
        return report_text
    
    def run_all_analyses(self) -> Dict:
        """Run all descriptive analyses"""
        self.logger.info("Starting comprehensive descriptive analysis...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Run individual analyses
            results = {}
            results['platform_distribution'] = self.analyze_platform_distribution(data)
            results['temporal_patterns'] = self.analyze_temporal_patterns(data)
            results['trial_characteristics'] = self.analyze_trial_characteristics(data)
            results['request_characteristics'] = self.analyze_request_characteristics(data)
            
            # Generate summary report
            summary_report = self.generate_summary_report(results)
            results['summary_report'] = summary_report
            
            self.logger.info("Descriptive analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during descriptive analysis: {e}")
            raise e
        
        finally:
            self.db_manager.close_all()

# analysis/quantitative_analysis.py

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from db.sqlalchemy_connector import DatabaseManager
from config import DATA_PATHS


class QuantitativeAnalyzer:
    """Performs quantitative analysis of clinical trial request data"""
    
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
    
    def load_quantitative_data(self) -> pd.DataFrame:
        """Load and prepare data for quantitative analysis"""
        self.logger.info("Loading data for quantitative analysis...")
        
        query = """
        SELECT 
            r.request_id,
            r.platform_id,
            p.name as platform_name,
            r.title,
            r.investigator,
            r.therapeutic_area,
            r.conditions_studied,
            r.intervention_studied,
            COUNT(rtl.nct_id) as num_trials_requested,
            COUNT(CASE WHEN ct.nct_id IS NOT NULL THEN 1 END) as num_trials_with_data,
            
            -- Trial characteristics (aggregated)
            AVG(ct.enrollment) as avg_enrollment,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 1%' THEN 1 END) as phase1_trials,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 2%' THEN 1 END) as phase2_trials,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 3%' THEN 1 END) as phase3_trials,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 4%' THEN 1 END) as phase4_trials,
            COUNT(CASE WHEN ct.study_type = 'Interventional' THEN 1 END) as interventional_trials,
            COUNT(CASE WHEN ct.study_type = 'Observational' THEN 1 END) as observational_trials,
            
            -- Temporal characteristics
            MIN(ct.start_date) as earliest_start_date,
            MAX(ct.start_date) as latest_start_date,
            AVG(EXTRACT(YEAR FROM ct.start_date)) as avg_start_year,
            
            -- Text characteristics
            LENGTH(r.title) as title_length,
            LENGTH(r.therapeutic_area) as therapeutic_area_length,
            LENGTH(r.conditions_studied) as conditions_length,
            LENGTH(r.intervention_studied) as intervention_length
            
        FROM requests r
        JOIN platforms p ON r.platform_id = p.id
        LEFT JOIN request_trial_links rtl ON r.request_id = rtl.request_id
        LEFT JOIN clinical_trials ct ON rtl.nct_id = ct.nct_id
        GROUP BY r.request_id, r.platform_id, p.name, r.title, r.investigator, 
                 r.therapeutic_area, r.conditions_studied, r.intervention_studied
        """
        
        try:
            df = self.db_manager.local_db.query_to_dataframe(query)
            self.logger.info(f"Loaded {len(df)} requests for quantitative analysis")
            
            # Clean and prepare data
            df = self._clean_quantitative_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading quantitative data: {e}")
            return pd.DataFrame()
    
    def _clean_quantitative_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare quantitative data"""
        self.logger.info("Cleaning quantitative data...")
        
        # Fill missing numeric values
        numeric_columns = [
            'num_trials_requested', 'num_trials_with_data', 'avg_enrollment',
            'phase1_trials', 'phase2_trials', 'phase3_trials', 'phase4_trials',
            'interventional_trials', 'observational_trials', 'avg_start_year',
            'title_length', 'therapeutic_area_length', 'conditions_length', 'intervention_length'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Calculate derived variables
        df['data_availability_rate'] = np.where(
            df['num_trials_requested'] > 0,
            df['num_trials_with_data'] / df['num_trials_requested'],
            0
        )
        
        df['total_phase_trials'] = (
            df['phase1_trials'] + df['phase2_trials'] + 
            df['phase3_trials'] + df['phase4_trials']
        )
        
        df['total_text_length'] = (
            df['title_length'] + df['therapeutic_area_length'] + 
            df['conditions_length'] + df['intervention_length']
        )
        
        return df
    
    def perform_correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform correlation analysis between key variables"""
        self.logger.info("Performing correlation analysis...")
        
        # Select numeric variables for correlation
        numeric_vars = [
            'num_trials_requested', 'num_trials_with_data', 'avg_enrollment',
            'data_availability_rate', 'total_phase_trials', 'total_text_length',
            'avg_start_year'
        ]
        
        # Filter available columns
        available_vars = [var for var in numeric_vars if var in df.columns]
        correlation_data = df[available_vars].select_dtypes(include=[np.number])
        
        if correlation_data.empty:
            self.logger.warning("No numeric data available for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Correlation Matrix of Request and Trial Characteristics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Identify strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def perform_platform_comparison(self, df: pd.DataFrame) -> Dict:
        """Compare characteristics across platforms"""
        self.logger.info("Performing platform comparison analysis...")
        
        if 'platform_name' not in df.columns:
            self.logger.warning("Platform information not available")
            return {}
        
        platforms = df['platform_name'].unique()
        comparison_results = {}
        
        # Variables to compare
        comparison_vars = [
            'num_trials_requested', 'avg_enrollment', 'data_availability_rate',
            'total_text_length'
        ]
        
        # Statistical tests for each variable
        for var in comparison_vars:
            if var not in df.columns:
                continue
                
            # Group data by platform
            groups = [df[df['platform_name'] == platform][var].dropna() for platform in platforms]
            groups = [group for group in groups if len(group) > 0]  # Remove empty groups
            
            if len(groups) < 2:
                continue
            
            # Perform ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                
                comparison_results[var] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # Platform means
                platform_means = df.groupby('platform_name')[var].mean()
                comparison_results[var]['platform_means'] = platform_means.to_dict()
                
            except Exception as e:
                self.logger.warning(f"Could not perform ANOVA for {var}: {e}")
        
        # Create comparison visualizations
        self._create_platform_comparison_plots(df, comparison_vars)
        
        return comparison_results
    
    def _create_platform_comparison_plots(self, df: pd.DataFrame, variables: List[str]):
        """Create visualization comparing platforms"""
        
        available_vars = [var for var in variables if var in df.columns]
        
        if not available_vars:
            return
        
        # Create subplots
        n_vars = len(available_vars)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(available_vars):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            
            # Box plot
            sns.boxplot(data=df, x='platform_name', y=var, ax=ax)
            ax.set_title(f'{var} by Platform')
            ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(available_vars), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_clustering_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform clustering analysis to identify request patterns"""
        self.logger.info("Performing clustering analysis...")
        
        # Select features for clustering
        clustering_features = [
            'num_trials_requested', 'avg_enrollment', 'data_availability_rate',
            'total_phase_trials', 'total_text_length'
        ]
        
        # Filter available features
        available_features = [feat for feat in clustering_features if feat in df.columns]
        
        if len(available_features) < 2:
            self.logger.warning("Insufficient features for clustering analysis")
            return {}
        
        # Prepare data
        clustering_data = df[available_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Determine optimal number of clusters using elbow method
        k_range = range(2, min(11, len(df)//2))
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_data, labels))
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'characteristics': {}
            }
            
            for feature in available_features:
                cluster_analysis[f'cluster_{cluster_id}']['characteristics'][feature] = {
                    'mean': float(cluster_data[feature].mean()),
                    'std': float(cluster_data[feature].std())
                }
        
        # Create clustering visualization
        if len(available_features) >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(f'Request Clustering (k={optimal_k})')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'optimal_clusters': optimal_k,
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'cluster_analysis': cluster_analysis,
            'feature_importance': dict(zip(available_features, pca.components_[0])) if len(available_features) >= 2 else {}
        }
    
    def analyze_request_complexity(self, df: pd.DataFrame) -> Dict:
        """Analyze the complexity of requests"""
        self.logger.info("Analyzing request complexity...")
        
        # Define complexity score based on multiple factors
        complexity_factors = []
        
        if 'num_trials_requested' in df.columns:
            complexity_factors.append(df['num_trials_requested'])
        
        if 'total_text_length' in df.columns:
            # Normalize text length
            normalized_text_length = (df['total_text_length'] - df['total_text_length'].min()) / \
                                   (df['total_text_length'].max() - df['total_text_length'].min())
            complexity_factors.append(normalized_text_length)
        
        if 'total_phase_trials' in df.columns:
            # Normalize phase diversity
            normalized_phase_trials = (df['total_phase_trials'] - df['total_phase_trials'].min()) / \
                                    (df['total_phase_trials'].max() - df['total_phase_trials'].min())
            complexity_factors.append(normalized_phase_trials)
        
        if complexity_factors:
            # Calculate composite complexity score
            complexity_score = np.mean(complexity_factors, axis=0)
            
            # Categorize complexity
            complexity_categories = pd.cut(
                complexity_score, 
                bins=3, 
                labels=['Low', 'Medium', 'High']
            )
            
            # Analyze complexity distribution
            complexity_distribution = complexity_categories.value_counts()
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            # Complexity distribution
            plt.subplot(1, 2, 1)
            complexity_distribution.plot(kind='bar')
            plt.title('Distribution of Request Complexity')
            plt.xlabel('Complexity Level')
            plt.ylabel('Number of Requests')
            plt.xticks(rotation=0)
            
            # Complexity score distribution
            plt.subplot(1, 2, 2)
            plt.hist(complexity_score, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Complexity Scores')
            plt.xlabel('Complexity Score')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'request_complexity.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'complexity_distribution': complexity_distribution.to_dict(),
                'complexity_stats': {
                    'mean': float(complexity_score.mean()),
                    'median': float(np.median(complexity_score)),
                    'std': float(complexity_score.std())
                }
            }
        
        return {}
    
    def generate_quantitative_report(self, all_results: Dict) -> str:
        """Generate comprehensive quantitative analysis report"""
        self.logger.info("Generating quantitative analysis report...")
        
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Correlation analysis
        if 'correlation_analysis' in all_results:
            corr_data = all_results['correlation_analysis']
            report.append("CORRELATION ANALYSIS:")
            report.append("-" * 40)
            
            if 'strong_correlations' in corr_data:
                if corr_data['strong_correlations']:
                    report.append("Strong correlations found (|r| > 0.5):")
                    for corr in corr_data['strong_correlations']:
                        report.append(f"  {corr['var1']} <-> {corr['var2']}: r = {corr['correlation']:.3f}")
                else:
                    report.append("No strong correlations found (|r| > 0.5)")
            report.append("")
        
        # Platform comparison
        if 'platform_comparison' in all_results:
            platform_data = all_results['platform_comparison']
            report.append("PLATFORM COMPARISON:")
            report.append("-" * 40)
            
            for var, results in platform_data.items():
                if isinstance(results, dict) and 'p_value' in results:
                    significance = "significant" if results['significant'] else "not significant"
                    report.append(f"  {var}: F = {results['f_statistic']:.3f}, p = {results['p_value']:.3f} ({significance})")
            report.append("")
        
        # Clustering analysis
        if 'clustering_analysis' in all_results:
            cluster_data = all_results['clustering_analysis']
            report.append("CLUSTERING ANALYSIS:")
            report.append("-" * 40)
            report.append(f"  Optimal number of clusters: {cluster_data.get('optimal_clusters', 'N/A')}")
            
            if 'cluster_analysis' in cluster_data:
                for cluster_id, cluster_info in cluster_data['cluster_analysis'].items():
                    report.append(f"  {cluster_id}: {cluster_info['size']} requests")
            report.append("")
        
        # Complexity analysis
        if 'complexity_analysis' in all_results:
            complexity_data = all_results['complexity_analysis']
            report.append("REQUEST COMPLEXITY ANALYSIS:")
            report.append("-" * 40)
            
            if 'complexity_stats' in complexity_data:
                stats = complexity_data['complexity_stats']
                report.append(f"  Mean complexity score: {stats['mean']:.3f}")
                report.append(f"  Median complexity score: {stats['median']:.3f}")
                report.append(f"  Standard deviation: {stats['std']:.3f}")
            
            if 'complexity_distribution' in complexity_data:
                report.append("  Complexity distribution:")
                for level, count in complexity_data['complexity_distribution'].items():
                    report.append(f"    {level}: {count} requests")
            report.append("")
        
        report.append("Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'quantitative_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Quantitative analysis report saved to: {report_path}")
        return report_text
    
    def run_all_analyses(self) -> Dict:
        """Run all quantitative analyses"""
        self.logger.info("Starting comprehensive quantitative analysis...")
        
        try:
            # Load data
            df = self.load_quantitative_data()
            
            if df.empty:
                self.logger.warning("No data available for quantitative analysis")
                return {}
            
            # Run individual analyses
            results = {}
            results['correlation_analysis'] = self.perform_correlation_analysis(df)
            results['platform_comparison'] = self.perform_platform_comparison(df)
            results['clustering_analysis'] = self.perform_clustering_analysis(df)
            results['complexity_analysis'] = self.analyze_request_complexity(df)
            
            # Generate report
            summary_report = self.generate_quantitative_report(results)
            results['summary_report'] = summary_report
            
            self.logger.info("Quantitative analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during quantitative analysis: {e}")
            raise e
        
        finally:
            self.db_manager.close_all()

# analysis/hypothesis_testing.py

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from db.sqlalchemy_connector import DatabaseManager
from config import DATA_PATHS


class HypothesisAnalyzer:
    """Performs hypothesis testing using hierarchical regression analysis"""
    
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
        
        # Define hypotheses
        self.hypotheses = {
            'H1': "Platform characteristics influence request success rate",
            'H2': "Trial complexity affects data sharing willingness", 
            'H3': "Temporal factors impact request patterns",
            'H4': "Request characteristics predict trial selection patterns"
        }
    
    def load_regression_data(self) -> pd.DataFrame:
        """Load and prepare data for regression analysis"""
        self.logger.info("Loading data for hypothesis testing...")
        
        query = """
        SELECT 
            r.request_id,
            r.platform_id,
            p.name as platform_name,
            
            -- Request characteristics (predictors)
            LENGTH(r.title) as title_length,
            LENGTH(r.therapeutic_area) as therapeutic_area_length,
            LENGTH(r.conditions_studied) as conditions_length,
            LENGTH(r.intervention_studied) as intervention_length,
            CASE WHEN r.therapeutic_area IS NOT NULL AND r.therapeutic_area != '' THEN 1 ELSE 0 END as has_therapeutic_area,
            CASE WHEN r.conditions_studied IS NOT NULL AND r.conditions_studied != '' THEN 1 ELSE 0 END as has_conditions,
            CASE WHEN r.intervention_studied IS NOT NULL AND r.intervention_studied != '' THEN 1 ELSE 0 END as has_interventions,
            
            -- Trial characteristics (predictors)
            COUNT(rtl.nct_id) as num_trials_requested,
            COUNT(CASE WHEN ct.nct_id IS NOT NULL THEN 1 END) as num_trials_with_data,
            AVG(ct.enrollment) as avg_enrollment,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 1%' THEN 1 END) as phase1_count,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 2%' THEN 1 END) as phase2_count,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 3%' THEN 1 END) as phase3_count,
            COUNT(CASE WHEN ct.phase LIKE 'Phase 4%' THEN 1 END) as phase4_count,
            COUNT(CASE WHEN ct.study_type = 'Interventional' THEN 1 END) as interventional_count,
            COUNT(CASE WHEN ct.study_type = 'Observational' THEN 1 END) as observational_count,
            
            -- Temporal characteristics (predictors)
            AVG(EXTRACT(YEAR FROM ct.start_date)) as avg_start_year,
            MAX(EXTRACT(YEAR FROM ct.start_date)) - MIN(EXTRACT(YEAR FROM ct.start_date)) as year_span,
            
            -- Outcome variables
            CASE WHEN COUNT(rtl.nct_id) > 0 
                THEN COUNT(CASE WHEN ct.nct_id IS NOT NULL THEN 1 END)::float / COUNT(rtl.nct_id)
                ELSE 0 END as data_availability_rate,
            
            -- Additional outcome variables
            CASE WHEN AVG(ct.enrollment) > 1000 THEN 1 ELSE 0 END as large_trials_focus,
            COUNT(DISTINCT ct.phase) as phase_diversity,
            
            -- Platform dummy variables for regression
            CASE WHEN p.name = 'Vivli' THEN 1 ELSE 0 END as platform_vivli,
            CASE WHEN p.name = 'CSDR' THEN 1 ELSE 0 END as platform_csdr,
            CASE WHEN p.name = 'YODA' THEN 1 ELSE 0 END as platform_yoda
            
        FROM requests r
        JOIN platforms p ON r.platform_id = p.id
        LEFT JOIN request_trial_links rtl ON r.request_id = rtl.request_id
        LEFT JOIN clinical_trials ct ON rtl.nct_id = ct.nct_id
        GROUP BY r.request_id, r.platform_id, p.name, r.title, r.therapeutic_area, 
                 r.conditions_studied, r.intervention_studied
        HAVING COUNT(rtl.nct_id) > 0  -- Only include requests with trials
        """
        
        try:
            df = self.db_manager.local_db.query_to_dataframe(query)
            self.logger.info(f"Loaded {len(df)} requests for regression analysis")
            
            # Clean and prepare data
            df = self._prepare_regression_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading regression data: {e}")
            return pd.DataFrame()
    
    def _prepare_regression_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for regression analysis"""
        self.logger.info("Preparing regression data...")
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Create composite variables
        df['total_text_length'] = (
            df['title_length'] + df['therapeutic_area_length'] + 
            df['conditions_length'] + df['intervention_length']
        )
        
        df['request_completeness'] = (
            df['has_therapeutic_area'] + df['has_conditions'] + df['has_interventions']
        ) / 3
        
        df['total_phases'] = (
            df['phase1_count'] + df['phase2_count'] + 
            df['phase3_count'] + df['phase4_count']
        )
        
        df['study_type_diversity'] = np.where(
            (df['interventional_count'] > 0) & (df['observational_count'] > 0), 1, 0
        )
        
        # Log transform skewed variables
        skewed_vars = ['num_trials_requested', 'avg_enrollment', 'total_text_length']
        for var in skewed_vars:
            if var in df.columns:
                df[f'log_{var}'] = np.log1p(df[var])  # log(1+x) to handle zeros
        
        # Handle outliers (cap at 95th percentile)
        continuous_vars = [
            'num_trials_requested', 'avg_enrollment', 'total_text_length', 
            'year_span', 'avg_start_year'
        ]
        
        for var in continuous_vars:
            if var in df.columns:
                p95 = df[var].quantile(0.95)
                df[var] = np.minimum(df[var], p95)
        
        return df
    
    def test_hypothesis_1(self, df: pd.DataFrame) -> Dict:
        """H1: Platform characteristics influence request success rate"""
        self.logger.info("Testing Hypothesis 1: Platform influence on success rate")
        
        if 'data_availability_rate' not in df.columns:
            self.logger.warning("Data availability rate not available for H1")
            return {}
        
        # Model 1: Platform only
        model1_formula = "data_availability_rate ~ platform_vivli + platform_csdr"
        model1 = ols(model1_formula, data=df).fit()
        
        # Model 2: Platform + controls
        model2_formula = ("data_availability_rate ~ platform_vivli + platform_csdr + "
                         "log_num_trials_requested + request_completeness + total_phases")
        model2 = ols(model2_formula, data=df).fit()
        
        # Compare models
        anova_result = anova_lm(model1, model2)
        
        return {
            'hypothesis': self.hypotheses['H1'],
            'model1_summary': {
                'r_squared': model1.rsquared,
                'f_pvalue': model1.f_pvalue,
                'coefficients': model1.params.to_dict(),
                'pvalues': model1.pvalues.to_dict()
            },
            'model2_summary': {
                'r_squared': model2.rsquared,
                'f_pvalue': model2.f_pvalue,
                'coefficients': model2.params.to_dict(),
                'pvalues': model2.pvalues.to_dict()
            },
            'model_comparison': {
                'f_statistic': anova_result.iloc[1]['F'],
                'p_value': anova_result.iloc[1]['Pr(>F)'],
                'r_squared_change': model2.rsquared - model1.rsquared
            }
        }
    
    def test_hypothesis_2(self, df: pd.DataFrame) -> Dict:
        """H2: Trial complexity affects data sharing willingness"""
        self.logger.info("Testing Hypothesis 2: Trial complexity and data sharing")
        
        # Create complexity index
        complexity_vars = ['log_avg_enrollment', 'phase_diversity', 'study_type_diversity', 'year_span']
        available_vars = [var for var in complexity_vars if var in df.columns]
        
        if len(available_vars) < 2:
            self.logger.warning("Insufficient variables for complexity analysis")
            return {}
        
        # Standardize complexity variables
        scaler = StandardScaler()
        complexity_data = scaler.fit_transform(df[available_vars].fillna(0))
        df['complexity_index'] = np.mean(complexity_data, axis=1)
        
        # Model 1: Complexity only
        model1_formula = "data_availability_rate ~ complexity_index"
        model1 = ols(model1_formula, data=df).fit()
        
        # Model 2: Complexity + platform controls
        model2_formula = ("data_availability_rate ~ complexity_index + platform_vivli + platform_csdr")
        model2 = ols(model2_formula, data=df).fit()
        
        # Model 3: Add interaction
        model3_formula = ("data_availability_rate ~ complexity_index + platform_vivli + platform_csdr + "
                         "complexity_index:platform_vivli + complexity_index:platform_csdr")
        model3 = ols(model3_formula, data=df).fit()
        
        return {
            'hypothesis': self.hypotheses['H2'],
            'complexity_components': available_vars,
            'model1_summary': {
                'r_squared': model1.rsquared,
                'f_pvalue': model1.f_pvalue,
                'coefficients': model1.params.to_dict(),
                'pvalues': model1.pvalues.to_dict()
            },
            'model2_summary': {
                'r_squared': model2.rsquared,
                'f_pvalue': model2.f_pvalue,
                'coefficients': model2.params.to_dict(),
                'pvalues': model2.pvalues.to_dict()
            },
            'model3_summary': {
                'r_squared': model3.rsquared,
                'f_pvalue': model3.f_pvalue,
                'coefficients': model3.params.to_dict(),
                'pvalues': model3.pvalues.to_dict()
            }
        }
    
    def test_hypothesis_3(self, df: pd.DataFrame) -> Dict:
        """H3: Temporal factors impact request patterns"""
        self.logger.info("Testing Hypothesis 3: Temporal factors and request patterns")
        
        if 'avg_start_year' not in df.columns:
            self.logger.warning("Temporal data not available for H3")
            return {}
        
        # Create temporal variables
        df['is_recent_trials'] = np.where(df['avg_start_year'] >= 2015, 1, 0)
        df['trial_age'] = 2024 - df['avg_start_year']  # Assuming current year is 2024
        
        # Model 1: Temporal factors predicting number of trials requested
        model1_formula = "log_num_trials_requested ~ trial_age + year_span"
        model1 = ols(model1_formula, data=df).fit()
        
        # Model 2: Add platform controls
        model2_formula = ("log_num_trials_requested ~ trial_age + year_span + "
                         "platform_vivli + platform_csdr")
        model2 = ols(model2_formula, data=df).fit()
        
        # Model 3: Temporal factors predicting data availability
        model3_formula = "data_availability_rate ~ is_recent_trials + trial_age"
        model3 = ols(model3_formula, data=df).fit()
        
        return {
            'hypothesis': self.hypotheses['H3'],
            'temporal_descriptives': {
                'avg_trial_age': float(df['trial_age'].mean()),
                'recent_trials_pct': float(df['is_recent_trials'].mean() * 100),
                'year_span_avg': float(df['year_span'].mean())
            },
            'model1_summary': {
                'r_squared': model1.rsquared,
                'f_pvalue': model1.f_pvalue,
                'coefficients': model1.params.to_dict(),
                'pvalues': model1.pvalues.to_dict()
            },
            'model2_summary': {
                'r_squared': model2.rsquared,
                'f_pvalue': model2.f_pvalue,
                'coefficients': model2.params.to_dict(),
                'pvalues': model2.pvalues.to_dict()
            },
            'model3_summary': {
                'r_squared': model3.rsquared,
                'f_pvalue': model3.f_pvalue,
                'coefficients': model3.params.to_dict(),
                'pvalues': model3.pvalues.to_dict()
            }
        }
    
    def test_hypothesis_4(self, df: pd.DataFrame) -> Dict:
        """H4: Request characteristics predict trial selection patterns"""
        self.logger.info("Testing Hypothesis 4: Request characteristics and trial selection")
        
        # Model 1: Request completeness predicting large trial focus
        if 'large_trials_focus' in df.columns:
            model1_formula = "large_trials_focus ~ request_completeness + log_total_text_length"
            model1 = ols(model1_formula, data=df).fit()
        else:
            model1 = None
        
        # Model 2: Request characteristics predicting phase diversity
        model2_formula = "phase_diversity ~ request_completeness + log_total_text_length + platform_vivli + platform_csdr"
        model2 = ols(model2_formula, data=df).fit()
        
        # Model 3: Comprehensive model
        model3_formula = ("log_num_trials_requested ~ request_completeness + log_total_text_length + "
                         "has_therapeutic_area + has_conditions + has_interventions + "
                         "platform_vivli + platform_csdr")
        model3 = ols(model3_formula, data=df).fit()
        
        results = {
            'hypothesis': self.hypotheses['H4'],
            'request_descriptives': {
                'avg_completeness': float(df['request_completeness'].mean()),
                'text_length_stats': {
                    'mean': float(df['total_text_length'].mean()),
                    'median': float(df['total_text_length'].median())
                }
            }
        }
        
        if model1:
            results['model1_summary'] = {
                'r_squared': model1.rsquared,
                'f_pvalue': model1.f_pvalue,
                'coefficients': model1.params.to_dict(),
                'pvalues': model1.pvalues.to_dict()
            }
        
        results.update({
            'model2_summary': {
                'r_squared': model2.rsquared,
                'f_pvalue': model2.f_pvalue,
                'coefficients': model2.params.to_dict(),
                'pvalues': model2.pvalues.to_dict()
            },
            'model3_summary': {
                'r_squared': model3.rsquared,
                'f_pvalue': model3.f_pvalue,
                'coefficients': model3.params.to_dict(),
                'pvalues': model3.pvalues.to_dict()
            }
        })
        
        return results
    
    def check_regression_assumptions(self, df: pd.DataFrame) -> Dict:
        """Check key regression assumptions"""
        self.logger.info("Checking regression assumptions...")
        
        # Example model for assumption checking
        model_formula = ("data_availability_rate ~ platform_vivli + platform_csdr + "
                        "log_num_trials_requested + request_completeness")
        
        try:
            model = ols(model_formula, data=df).fit()
            
            # Get residuals and fitted values
            residuals = model.resid
            fitted_values = model.fittedvalues
            
            # Check multicollinearity (VIF)
            predictor_vars = ['platform_vivli', 'platform_csdr', 'log_num_trials_requested', 'request_completeness']
            available_vars = [var for var in predictor_vars if var in df.columns]
            
            vif_data = df[available_vars].fillna(0)
            vif_results = {}
            
            for i, var in enumerate(available_vars):
                try:
                    vif_value = variance_inflation_factor(vif_data.values, i)
                    vif_results[var] = vif_value
                except:
                    vif_results[var] = np.nan
            
            # Create assumption plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Residuals vs Fitted
            axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted Values')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot of Residuals')
            
            # Scale-Location plot
            standardized_residuals = residuals / np.std(residuals)
            axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
            axes[1, 0].set_xlabel('Fitted Values')
            axes[1, 0].set_ylabel('√|Standardized Residuals|')
            axes[1, 0].set_title('Scale-Location Plot')
            
            # Histogram of residuals
            axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Residuals')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'regression_assumptions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'vif_results': vif_results,
                'residual_stats': {
                    'mean': float(residuals.mean()),
                    'std': float(residuals.std()),
                    'min': float(residuals.min()),
                    'max': float(residuals.max())
                },
                'normality_test': {
                    'shapiro_stat': float(stats.shapiro(residuals)[0]),
                    'shapiro_pvalue': float(stats.shapiro(residuals)[1])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking regression assumptions: {e}")
            return {}
    
    def create_regression_visualizations(self, results: Dict):
        """Create visualizations for regression results"""
        self.logger.info("Creating regression visualizations...")
        
        # Create coefficient plot for all hypotheses
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        hypothesis_names = ['H1', 'H2', 'H3', 'H4']
        
        for i, hyp in enumerate(hypothesis_names):
            if hyp in results and 'model2_summary' in results[hyp]:
                model_results = results[hyp]['model2_summary']
                coefficients = model_results['coefficients']
                pvalues = model_results['pvalues']
                
                # Filter out intercept and get significant coefficients
                coef_data = [(var, coef, pvalues.get(var, 1.0)) 
                           for var, coef in coefficients.items() 
                           if var != 'Intercept' and var in pvalues]
                
                if coef_data:
                    variables, coefs, pvals = zip(*coef_data)
                    colors = ['red' if p < 0.05 else 'blue' for p in pvals]
                    
                    bars = axes[i].barh(variables, coefs, color=colors, alpha=0.7)
                    axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    axes[i].set_title(f'{hyp}: {results[hyp]["hypothesis"][:50]}...')
                    axes[i].set_xlabel('Coefficient Value')
                    
                    # Add significance indicators
                    for j, (bar, pval) in enumerate(zip(bars, pvals)):
                        if pval < 0.001:
                            axes[i].text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                                       '***', ha='left' if bar.get_width() > 0 else 'right', va='center')
                        elif pval < 0.01:
                            axes[i].text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                                       '**', ha='left' if bar.get_width() > 0 else 'right', va='center')
                        elif pval < 0.05:
                            axes[i].text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                                       '*', ha='left' if bar.get_width() > 0 else 'right', va='center')
                else:
                    axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                               transform=axes[i].transAxes)
                    axes[i].set_title(f'{hyp}: No results')
            else:
                axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{hyp}: No results')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_testing_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_hypothesis_report(self, results: Dict) -> str:
        """Generate comprehensive hypothesis testing report"""
        self.logger.info("Generating hypothesis testing report...")
        
        report = []
        report.append("=" * 80)
        report.append("HYPOTHESIS TESTING REPORT - HIERARCHICAL REGRESSION ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        for hyp_name, hyp_text in self.hypotheses.items():
            report.append(f"{hyp_name}: {hyp_text}")
            report.append("-" * 60)
            
            if hyp_name in results:
                hyp_results = results[hyp_name]
                
                # Report key findings
                if 'model2_summary' in hyp_results:
                    model = hyp_results['model2_summary']
                    report.append(f"  Model R²: {model['r_squared']:.4f}")
                    report.append(f"  Model p-value: {model['f_pvalue']:.4f}")
                    
                    # Significant predictors
                    significant_predictors = [
                        var for var, pval in model['pvalues'].items() 
                        if pval < 0.05 and var != 'Intercept'
                    ]
                    
                    if significant_predictors:
                        report.append(f"  Significant predictors (p < 0.05): {', '.join(significant_predictors)}")
                    else:
                        report.append("  No significant predictors found")
                
                # Model comparison if available
                if 'model_comparison' in hyp_results:
                    comparison = hyp_results['model_comparison']
                    report.append(f"  Model improvement: ΔR² = {comparison['r_squared_change']:.4f}")
                    report.append(f"  Comparison p-value: {comparison['p_value']:.4f}")
                
                report.append("")
            else:
                report.append("  No results available for this hypothesis")
                report.append("")
        
        # Summary of regression assumptions
        if 'assumptions' in results:
            report.append("REGRESSION ASSUMPTIONS CHECK:")
            report.append("-" * 40)
            
            assumptions = results['assumptions']
            
            if 'vif_results' in assumptions:
                report.append("  Multicollinearity (VIF values):")
                for var, vif in assumptions['vif_results'].items():
                    if not np.isnan(vif):
                        concern = "High" if vif > 5 else "Moderate" if vif > 2.5 else "Low"
                        report.append(f"    {var}: {vif:.2f} ({concern})")
            
            if 'normality_test' in assumptions:
                norm_test = assumptions['normality_test']
                normality = "Normal" if norm_test['shapiro_pvalue'] > 0.05 else "Non-normal"
                report.append(f"  Residual normality: {normality} (Shapiro-Wilk p = {norm_test['shapiro_pvalue']:.4f})")
            
            report.append("")
        
        # Overall conclusions
        report.append("OVERALL CONCLUSIONS:")
        report.append("-" * 40)
        report.append("  [To be interpreted based on specific research context]")
        report.append("  • Statistical significance does not imply practical significance")
        report.append("  • Consider effect sizes alongside p-values")
        report.append("  • Verify assumptions are met for valid inference")
        report.append("")
        
        report.append("Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'hypothesis_testing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Hypothesis testing report saved to: {report_path}")
        return report_text
    
    def run_hierarchical_regression(self) -> Dict:
        """Run complete hierarchical regression analysis for all hypotheses"""
        self.logger.info("Starting hierarchical regression analysis...")
        
        try:
            # Load data
            df = self.load_regression_data()
            
            if df.empty:
                self.logger.warning("No data available for regression analysis")
                return {}
            
            self.logger.info(f"Running regression analysis on {len(df)} observations")
            
            # Test all hypotheses
            results = {}
            results['H1'] = self.test_hypothesis_1(df)
            results['H2'] = self.test_hypothesis_2(df)
            results['H3'] = self.test_hypothesis_3(df)
            results['H4'] = self.test_hypothesis_4(df)
            
            # Check regression assumptions
            results['assumptions'] = self.check_regression_assumptions(df)
            
            # Create visualizations
            self.create_regression_visualizations(results)
            
            # Generate comprehensive report
            summary_report = self.generate_hypothesis_report(results)
            results['summary_report'] = summary_report
            
            self.logger.info("Hierarchical regression analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during hierarchical regression analysis: {e}")
            raise e
        
        finally:
            self.db_manager.close_all()

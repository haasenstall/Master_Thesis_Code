import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
import sys
import datetime
from typing import List, Dict, Union, Optional, Tuple
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import PROJECT_SETTINGS, DATA_PATHS, LOGGING_CONFIG

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pandas.api.types import is_numeric_dtype
from scipy.stats import ttest_ind, chi2_contingency


class PlotManager:
    """Style and layout manager for plots"""

    def __init__(self, style_name: str = 'master_thesis'):
        self.style_name = style_name
        self.setup_style()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"PlotManager initialized with style: {self.style_name}")

    def setup_style(self):
        """Set up the plotting style"""

        plt.style.use('seaborn-v0_8-paper')

        # custom colors
        self.colors = {
            'primary': "#2D3E50", # dark blue
            'secondary': '#4ECDC4', # light blue
            'accent': '#FF6B6B', # coral
            'success': '#27AE60',
            'neutral': '#BDC3C7',
            'light': '#F4F6F7',
            'dark': '#1A252F'
        }
        
        # custom color palette
        self.color_palette = [
            '#2D3E50',
            '#3f4f60',
            '#526170',
            '#667381',
            '#7a8692', 
            '#8f99a4',
            '#a5adb5', 
            '#bbc1c7', 
            '#d1d5da',
            '#e8eaec',
        ]

        plt.rcParams.update({
            'figure.figsize': (16, 8),
            'font.family': 'sans-serif',
            'font.size': 18,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 20,
            'figure.titlesize': 20,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': self.colors['light'],
            'axes.edgecolor': self.colors['dark'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 2,
            'lines.linewidth': 1,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'figure.max_open_warning': 0,
            'figure.constrained_layout.use': True
        })

class MasterPlotter:
    """Class for creating standardized plots"""

    def __init__(self, style_manager: PlotManager = None, output_dir: str = DATA_PATHS['img']['default']):
        self.style_manager = style_manager or PlotManager()
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("MasterPlotter initialized")

    def save_plot(self, fig, filename: str, path: str = None, dpi: int = 300, bbox_inches: str = 'tight'):
        """Save the plot to the output directory"""
        if not path:
            path = self.output_dir
        filepath = os.path.join(path, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, edgecolor=None)
        self.logger.info(f"Plot saved to {filepath}")
        plt.close(fig)

    def bar_plot(self, data: Union[pd.DataFrame, Dict], 
                 x: str = None, y: str = None,
                 xlabel: str = None, ylabel: str = None,
                 horizontal: bool = False, stacked: bool = False,
                 color_column: str = None, figsize: Tuple = (16, 8),
                 save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create a bar plot from the data"""
        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(data, dict):
            df = pd.DataFrame(list(data.items()), columns=[x or 'Category', y or 'Value'])
            x, y = x or 'Category', y or 'Value'  
        else:
            df = data.copy()

        # Initialize colors variable BEFORE the conditional logic
        colors = self.style_manager.color_palette
        primary_color = self.style_manager.colors['primary']

        # Only modify colors if color_column exists and is valid
        if color_column and color_column in df.columns:
            unique_colors = df[color_column].unique()
            if len(unique_colors) < 3:
                color_palette = [self.style_manager.colors['primary'], 
                                 self.style_manager.colors['secondary']]
                # Create color mapping for unique values
                colors = dict(zip(unique_colors, color_palette))
            else:
                colors = self.style_manager.color_palette[:len(unique_colors)]

        if horizontal:
            if color_column and color_column in df.columns:
               sns.barplot(data=df, y=x, x=y, hue=color_column, ax=ax, palette=colors, **kwargs)
            else:
                sns.barplot(data=df, y=x, x=y, ax=ax, color=primary_color, **kwargs)
        else:
            if color_column and color_column in df.columns:
                sns.barplot(data=df, x=x, y=y, hue=color_column, ax=ax, palette=colors, **kwargs)
            else:
                sns.barplot(data=df, x=x, y=y, ax=ax, color=primary_color, **kwargs)

        # styling
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        # Rotate x-axis labels
        if not horizontal and x in df.columns and len(df[x].unique()) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # FIXED: Dynamic formatting based on value ranges
        def get_dynamic_format(values):
            """Return appropriate format based on value range"""
            max_val = max(abs(v) for v in values if pd.notna(v) and v != 0)
            
            if max_val < 1:
                return '%.2f'  # Show 2 decimal places for values < 1
            elif max_val < 10:
                return '%.1f'  # Show 1 decimal place for values < 10
            else:
                return '%.0f'  # Show no decimal places for values >= 10

        # add value labels with dynamic formatting
        for container in ax.containers:
            # Get the values from the container
            values = [v.get_height() if not horizontal else v.get_width() for v in container]
            values = [v for v in values if pd.notna(v)]  # Remove NaN values
            
            if values:  # Only if we have valid values
                fmt = get_dynamic_format(values)
                ax.bar_label(container, fmt=fmt, padding=2)

        if color_column and color_column in df.columns and len(df[color_column].unique()) > 1:
            ax.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')

        # FIXED: Consistent layout handling
        self._handle_layout()

        if save_name and path:
            self.save_plot(fig, save_name, path)

        return fig

    def _handle_layout(self):
        """Handle layout consistently across all plot types"""
        try:
            if plt.rcParams.get('figure.constrained_layout.use', False):
                # Constrained layout is enabled, don't use tight_layout
                pass
            else:
                plt.tight_layout()
        except Exception:
            # Fallback to manual adjustment
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.85)

    def line_plot(self, data: pd.DataFrame, x: str, y: str, hue: str = None, 
                  xlabel: str = None, ylabel: str = None, 
                  markers: bool = True, figsize: Tuple = (16, 8), 
                  save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create a line plot from the data"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if hue is not None:
            unique_hues = data[hue].unique()
            palette = self.style_manager.color_palette[:len(unique_hues)]
        else:
            palette = [self.style_manager.colors['primary']]

    # Create the base lineplot with hue
        sns.lineplot(
            data=data, 
            x=x, 
            y=y, 
            hue=hue,
            ax=ax, 
            palette=palette,  
            marker='o' if markers else None,
            markersize=6 if markers else 0,
            linewidth=2,
            **kwargs
    )
        
        # Apply different line styles manually to each line
        if hue is not None:
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
            lines = ax.get_lines()
            
            for i, line in enumerate(lines):
                linestyle = line_styles[i % len(line_styles)]
                line.set_linestyle(linestyle)

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        # Format legend
        if hue is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, 
                     title=hue.replace('_', ' ').title(), 
                     bbox_to_anchor=(1.05, 1), loc='upper right')

        # Use consistent layout handling
        self._handle_layout()

        if save_name:
            self.save_plot(fig, save_name, path)
        return fig

    def histogram(self, data: Union[pd.DataFrame, pd.Series, list, np.ndarray], 
          column: str = None, x: str = None,
          xlabel: str = None, ylabel: str = "Count",
          bins: int = 30, kde: bool = True, density: bool = False,
          figsize: Tuple = (16, 8), save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create customizable histograms using matplotlib"""

        fig, ax = plt.subplots(figsize=figsize)

        try:
            col = column or x
            
            # Extract the actual data values
            if isinstance(data, pd.DataFrame):
                if col not in data.columns:
                    self.logger.error(f"Column '{col}' not found in DataFrame. Available columns: {data.columns.tolist()}")
                    return None
                values_series = data[col].copy()
            elif isinstance(data, pd.Series):
                values_series = data.copy()
                col = col or data.name or 'values'
            else:
                values_series = pd.Series(data)
                col = col or 'values'

            # Ensure values are numeric and clean
            values_series = pd.to_numeric(values_series, errors='coerce').dropna()
    
            if len(values_series) == 0:
                self.logger.error("No valid numeric data found for histogram")
                return None
    
            self.logger.info(f"Creating histogram for {len(values_series)} values")
    
            # Plot histogram using seaborn
            sns.histplot(
                values_series,
                bins=bins,
                kde=False,
                stat='density' if density else 'count',
                color=self.style_manager.colors['primary'],
                ax=ax,
                **kwargs
            )
    
            # Add KDE if requested and data is suitable
            kde_added = False
            if kde:
                try:
                    # Check if KDE is feasible
                    if len(values_series.unique()) > 1 and values_series.std() > 1e-10:
                        from scipy.stats import gaussian_kde
                        kde_data = gaussian_kde(values_series)
                        x_range = np.linspace(values_series.min(), values_series.max(), 100)
                        kde_values = kde_data(x_range)
                
                        # Scale KDE to match histogram
                        if not density:
                            # Get histogram info for scaling
                            counts, bin_edges = np.histogram(values_series, bins=bins)
                            kde_values = kde_values * len(values_series) * (bin_edges[1] - bin_edges[0])
                
                        ax.plot(x_range, kde_values, 
                           color=self.style_manager.colors['accent'], 
                           linewidth=3, 
                           label='KDE')
                        kde_added = True
                    else:
                        self.logger.warning("KDE skipped: insufficient data variance for reliable estimation")
                except Exception as kde_error:
                    self.logger.warning(f"KDE calculation failed: {kde_error}")

            # Add statistical lines
            mean_val = values_series.mean()
            median_val = values_series.median()
        
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.0f}', alpha=0.8)
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.0f}', alpha=0.8)
        
            # Create custom legend with statistics
            legend_elements = []
            
            # Add KDE to legend if it was successfully added
            if kde_added:
                legend_elements.append(plt.Line2D([0], [0], color=self.style_manager.colors['accent'], 
                                                linewidth=3, label='KDE'))
            
            # Add statistical lines
            legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                            linewidth=2, label=f'Mean: {mean_val:.0f}'))
            legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', 
                                            linewidth=2, label=f'Median: {median_val:.0f}'))
            
            # Add statistics as text entries
            legend_elements.extend([
                plt.Line2D([0], [0], color='none', label=f'Count: {len(values_series)}'),
                plt.Line2D([0], [0], color='none', label=f'Std: {values_series.std():.0f}')
            ])
            
            # Create legend with all elements
            ax.legend(handles=legend_elements, loc='upper right', 
                    frameon=True, fancybox=True, shadow=True, 
                    framealpha=0.9, facecolor='white', edgecolor='gray')
        
            # Styling
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.4, axis='y')
        
            self._handle_layout()

            if save_name and path:
                self.save_plot(fig, save_name, path)
                self.logger.info(f"Histogram saved as {save_name}")

        except Exception as e:
            self.logger.error(f"Error creating histogram: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

        return fig

    def pie_chart(self, data: Union[pd.DataFrame, Dict], 
                  values: str = None, labels: str = None,
                  figsize: Tuple = (8, 8),
                  explode: List[float] = None, autopct: str = '%1.1f%%',
                  save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create customizable pie charts"""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(data, dict):
            labels_list = list(data.keys())
            values_list = list(data.values())
        else:
            labels_list = data[labels].tolist()
            values_list = data[values].tolist()
        
        colors = self.style_manager.color_palette[:len(labels_list)]

        wedges, texts, autotexts = ax.pie(values_list, labels=labels_list, colors=colors, autopct=autopct, explode=explode, startangle=90)

        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')        
        plt.tight_layout()
        
        if save_name:
            self.save_plot(fig, save_name, path)
        
        return fig

    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, xlabel: str = None, ylabel: str = None, color_column: str = None, size_column:str = None, figsize: Tuple = (16, 8), save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create a scatter plot from the data"""
        fig, ax = plt.subplots(figsize=figsize)
        colors = self.style_manager.color_palette

        scatter_kwargs = {
            'data': data,
            'x': x,
            'y': y,
            'ax': ax,
            's': 100,
            'alpha': 0.7,
            'edgecolor': 'w',
            **kwargs
        }

        if color_column and color_column in data.columns:
            scatter_kwargs['hue'] = color_column
            scatter_kwargs['palette'] = self.style_manager.color_palette
        else:
            scatter_kwargs['color'] = self.style_manager.colors['primary']

        if size_column and size_column in data.columns:
            scatter_kwargs['size'] = size_column
            scatter_kwargs['sizes'] = (50, 300)

        sns.scatterplot(**scatter_kwargs)

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        plt.tight_layout()

        if save_name:
            self.save_plot(fig, save_name, path)

        return fig

    def heatmap(self, data: pd.DataFrame, 
            xlabel: str = None, ylabel: str = None,
            annot: bool = True, fmt: str = '.2f', cmap: str = 'Blues', 
            figsize: Tuple = (12, 10), save_name: str = None, path: str = None, 
            **kwargs) -> plt.Figure:
        """Create a heatmap from the data"""
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            data,  # Use DataFrame directly
            annot=annot, 
            fmt=fmt,
            cmap=cmap, 
            ax=ax,
            **kwargs
        )
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Rotate labels if they're long
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Handle layout
        self._handle_layout()
        
        if save_name and path:
            self.save_plot(fig, save_name, path)

        plt.close(fig)
        return fig

    def box_plot(self, data: pd.DataFrame, x: str, y: str, xlabel: str = None, ylabel: str = None, figsize: Tuple = (16, 8), save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create Customizable box plots"""

        fig, ax = plt.subplots(figsize=figsize)
        colors = self.style_manager.color_palette

        sns.boxplot(data=data, x=x, y=y, ax=ax, palette=colors, **kwargs)

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        if x and len(data[x].unique()) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()

        if save_name:
            self.save_plot(fig, save_name, path)

        return fig

    def violin_plot(self, data: pd.DataFrame, x: str, y: str, xlabel: str = None, ylabel: str = None, figsize: Tuple = (16, 8), save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create Customizable violin plots"""

        fig, ax = plt.subplots(figsize=figsize)
        colors = self.style_manager.color_palette

        sns.violinplot(data=data, x=x, y=y, ax=ax, palette=colors, **kwargs)

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        if x and len(data[x].unique()) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_name:
            self.save_plot(fig, save_name, path)

        return fig

    def correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson', figsize: Tuple = (12, 10), cmap: str = None, annot: bool = True, save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create a correlation matrix heatmap"""

        fig, ax = plt.subplots(figsize=figsize)
        cmap = cmap or 'Blues'

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(data.corr(), dtype=bool))

        corr = data.corr(method=method)
        sns.heatmap(corr, cmap=cmap, annot=annot, fmt=".2f", ax=ax, mask=mask, square=True, cbar = {'shrink': .8}, **kwargs)

        plt.tight_layout()

        if save_name:
            self.save_plot(fig, save_name, path)

        return fig
    
    def multi_subplot(self, plot_configs: List[Dict], 
                      figsize: Tuple = (15, 10), save_name: str = None, path: str = None) -> plt.Figure:
        """Create multiple subplots in one figure"""
        
        n_plots = len(plot_configs)
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, config in enumerate(plot_configs):
            ax = axes[i]
            plot_type = config.get('type', 'bar')
            data = config['data']
            
            if plot_type == 'bar':
                if isinstance(data, dict):
                    ax.bar(data.keys(), data.values(), 
                          color=self.style_manager.color_palette[i % len(self.style_manager.color_palette)])
                else:
                    x_col = config.get('x', data.columns[0])
                    y_col = config.get('y', data.columns[1])
                    sns.barplot(x=data[x_col], y=data[y_col],
                          color=self.style_manager.color_palette[i % len(self.style_manager.color_palette)], ax=ax)

            elif plot_type == 'line':
                x_col = config.get('x', data.columns[0])
                y_col = config.get('y', data.columns[1])
                sns.lineplot(x=data[x_col], y=data[y_col], 
                       color=self.style_manager.color_palette[i % len(self.style_manager.color_palette)], ax=ax)

            elif plot_type == 'hist':
                column = config.get('column', data.columns[0])
                sns.histplot(data[column], bins=config.get('bins', 20),
                       color=self.style_manager.color_palette[i % len(self.style_manager.color_palette)],
                       alpha=0.7, ax=ax)
            
            ax.set_xlabel(config.get('xlabel', ''))
            ax.set_ylabel(config.get('ylabel', ''))
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_name:
            self.save_plot(fig, save_name, path)
        
        return fig

    def logistic_regression_plot(self, data: pd.DataFrame, x: str, y: str,
                           xlabel: str = None, ylabel: str = None,
                           show_stats: bool = False, figsize: Tuple = (16, 8), 
                           save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create a logistic regression plot for binary classification"""
        fig, ax = plt.subplots(figsize=figsize)

        # prepare data
        data[y] = pd.to_numeric(data[y], errors="coerce")
        # x numeric
        data[x] = pd.to_numeric(data[x], errors="coerce")
        data = data.dropna(subset=[x, y])

        used_seaborn = False
        try:
            # Create logistic regression plot
            sns.regplot(
                data=data, x=x, y=y, ax=ax,
                logistic=True, 
                scatter_kws={'alpha': 0.6, 's': 60, 'color': self.style_manager.colors['primary']},
                line_kws={'color': self.style_manager.colors['accent'], 'linewidth': 3},
                y_jitter=0.03,
                **kwargs
            )
            used_seaborn = True
        except Exception as e:
            X = data[[x]]
            y_vals = data[y]
            log_reg = LogisticRegression(solver='lbfgs', max_iter=200)
            log_reg.fit(X, y_vals)
            x_range = np.linspace(X[x].min(), X[x].max(), 200).reshape(-1, 1)
            p_grid = log_reg.predict_proba(x_range)[:, 1]

            ax.scatter(data[x], data[y] + np.random.uniform(-0.03, 0.03, size=len(data)),
                       alpha=0.6, s=40, color=self.style_manager.colors['primary'])
            ax.plot(x_range, p_grid, color=self.style_manager.colors['accent'], linewidth=3)
            self.logger.warning(f"Seaborn regplot failed, used manual logistic regression fit. Error: {e}")

        # Calculate statistics if requested
        if show_stats:            
            X = data[[x]]
            y_vals = data[y]
            
            # Fit logistic regression
            log_reg = LogisticRegression(solver='lbfgs', max_iter=200)
            log_reg.fit(X, y_vals)
            
            # Make predictions
            y_pred = log_reg.predict(X)
            y_proba = log_reg.predict_proba(X)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_vals, y_pred)
            precision = precision_score(y_vals, y_pred, zero_division=0)
            recall = recall_score(y_vals, y_pred, zero_division=0)
            f1 = f1_score(y_vals, y_pred, zero_division=0)
            
            # Create statistics text
            stats_text = (
                f'Accuracy: {accuracy:.3f}\n'
                f'Precision: {precision:.3f}\n'
                f'Recall: {recall:.3f}\n'
                f'F1-Score: {f1:.3f}\n'
                f'Coefficient: {log_reg.coef_[0][0]:.3f}\n'
                f'N: {len(data)}'
            )
            
            # Add statistics box
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or 'Probability of Request' if used_seaborn else 'Requests (0/1)')
        ax.set_ylim(-0.1, 1.1)
        
        self._handle_layout()
        
        if save_name and path:
            self.save_plot(fig, save_name, path)
        
        return fig

    def classification_summary(self, model_results: Dict, 
                      figsize: Tuple = (16, 8), save_name: str = None, path: str = None) -> plt.Figure:
        """Create comprehensive classification summary with confusion matrix and metrics"""
        
        # Plot 1: Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))  # Fixed: proper figure creation
        confusion_matrix = model_results.get('confusion_matrix', [[0, 0], [0, 0]])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Request', 'Request'], 
                    yticklabels=['No Request', 'Request'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        cm_save_name = save_name.replace('.png', '_confusion_matrix.png') if save_name else None
        if cm_save_name and path:
            self.save_plot(fig, cm_save_name, path)
        
        # Plot 2: Classification Metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            model_results.get('accuracy', 0),
            model_results.get('precision', 0),
            model_results.get('recall', 0),
            model_results.get('f1_score', 0)
        ]
        self.bar_plot(
            data=dict(zip(metrics, values)), 
            x='Metric', y='Score', 
            ylabel='Score', 
            figsize=(8, 6), 
            save_name=save_name.replace('.png', '_classification_metrics.png') if save_name else None, 
            path=path
        )

        # Plot 3: Feature Coefficients
        if 'coefficients' in model_results:
            coef_data = model_results['coefficients']
            features = list(coef_data.keys())
            coef_values = list(coef_data.values())

            self.bar_plot(
                data=dict(zip(features, coef_values)), 
                x='Feature', y='Coefficient',  
                ylabel='Coefficient Value', 
                horizontal=True, 
                figsize=(8, 6), 
                save_name=save_name.replace('.png', '_feature_coefficients.png') if save_name else None, 
                path=path
            )
        
        # Plot 4: ROC Curve (if data available)
        if 'fpr' in model_results and 'tpr' in model_results:
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))  # Fixed: proper figure creation
            fpr = model_results['fpr']
            tpr = model_results['tpr']
            auc_score = model_results.get('auc', 0)
            
            ax_roc.plot(fpr, tpr, color=self.style_manager.colors['accent'], linewidth=2, 
                    label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax_roc.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random Classifier')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)
            
            self._handle_layout()
            
            roc_save_name = save_name.replace('.png', '_roc_curve.png') if save_name else None
            if roc_save_name and path:
                self.save_plot(fig_roc, roc_save_name, path)
        
        # Plot 5: Prediction Distribution (if data available)
        if 'predicted_probabilities' in model_results:
            fig_dist, ax_dist = plt.subplots(figsize=(8, 6))  # Fixed: proper figure creation
            probas = model_results['predicted_probabilities']
            actual = model_results.get('actual_labels', [])
            
            # Separate probabilities by actual class
            if len(actual) > 0:
                no_request_probas = [p for p, a in zip(probas, actual) if a == 0]
                request_probas = [p for p, a in zip(probas, actual) if a == 1]
                
                ax_dist.hist(no_request_probas, bins=20, alpha=0.7, label='No Request', 
                        color=self.style_manager.colors['neutral'])
                ax_dist.hist(request_probas, bins=20, alpha=0.7, label='Request', 
                        color=self.style_manager.colors['primary'])
                ax_dist.legend()
            else:
                ax_dist.hist(probas, bins=20, alpha=0.7, color=self.style_manager.colors['primary'])
            
            ax_dist.set_xlabel('Predicted Probability')
            ax_dist.set_ylabel('Frequency')
            
            self._handle_layout()
            
            dist_save_name = save_name.replace('.png', '_prediction_distribution.png') if save_name else None
            if dist_save_name and path:
                self.save_plot(fig_dist, dist_save_name, path)
        
        # Plot 6: Model Statistics Summary
        fig_stats, ax_stats = plt.subplots(figsize=(8, 6))  # Fixed: proper figure creation
        ax_stats.axis('off')
        
        stats_text = f"""Model Performance Summary:

            Accuracy: {model_results.get('accuracy', 0):.3f}
            Precision: {model_results.get('precision', 0):.3f}
            Recall: {model_results.get('recall', 0):.3f}
            F1-Score: {model_results.get('f1_score', 0):.3f}
            AUC-ROC: {model_results.get('auc', 0):.3f}

            Log-Likelihood: {model_results.get('log_likelihood', 0):.2f}
            AIC: {model_results.get('aic', 0):.2f}

            Sample Size: {model_results.get('n_samples', 0)}
            Positive Class: {model_results.get('n_positive', 0)}
            Negative Class: {model_results.get('n_negative', 0)}
            """
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        
        self._handle_layout()
        
        stats_save_name = save_name.replace('.png', '_model_statistics.png') if save_name else None
        if stats_save_name and path:
            self.save_plot(fig_stats, stats_save_name, path)
        
        return fig  # Return the confusion matrix figure as the main figure

    def binary_feature_comparison(self, data: pd.DataFrame, features: List[str], target: str,
                                figsize: Tuple = (16, 8), save_name: str = None, path: str = None) -> plt.Figure:
        """Compare features between two classes in `target` with p-values (t-test for numeric, chi-square for categorical)."""

        # Identify the two classes in the target column
        classes = pd.Series(data[target]).dropna().unique()
        if len(classes) != 2:
            raise ValueError(f"`{target}` must be binary; found classes: {classes}")

        # Keep a stable order for plotting/grouping
        # If the target is numeric/bool, sort; for strings, keep encountered order
        try:
            classes = np.sort(classes)
        except Exception:
            pass

        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, feature in enumerate(features):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            # Decide numeric vs categorical
            numeric = is_numeric_dtype(data[feature])

            # PLOT
            if numeric:
                # violin by class
                sns.violinplot(
                    data=data,
                    x=target, y=feature, ax=ax,
                    order=classes.tolist(),
                    palette=[self.style_manager.colors['neutral'], self.style_manager.colors['primary']]
                )
                # mean markers
                means = data.groupby(target, observed=True)[feature].mean()
                for j, cls in enumerate(classes):
                    if cls in means.index and pd.notna(means.loc[cls]):
                        ax.scatter(j, means.loc[cls], color='red', s=100, marker='D', zorder=10)
            else:
                # countplot by feature with hue=target
                sns.countplot(
                    data=data,
                    x=feature, hue=target, ax=ax,
                    hue_order=classes.tolist(),
                    palette=[self.style_manager.colors['neutral'], self.style_manager.colors['primary']]
                )
                ax.tick_params(axis='x', rotation=45)


            # STATS
            p_text = None
            try:
                if numeric:
                    g0 = data.loc[data[target] == classes[0], feature].dropna()
                    g1 = data.loc[data[target] == classes[1], feature].dropna()
                    if len(g0) > 1 and len(g1) > 1:
                        # Welch’s t-test + ignore NaNs (already dropped)
                        stat, p_val = ttest_ind(g0, g1, equal_var=False)
                        p_text = f"t-test p = {p_val:.3g}" if p_val >= 0.001 else "t-test p < 0.001"
                    else:
                        p_text = "t-test: insufficient data"
                else:
                    # Chi-square test of independence
                    ct = pd.crosstab(data[feature], data[target])
                    # ensure both classes present as columns
                    for cls in classes:
                        if cls not in ct.columns:
                            ct[cls] = 0
                    ct = ct[classes]  # order columns
                    if ct.values.sum() > 0 and ct.shape[0] > 1:
                        chi2, p_val, dof, exp = chi2_contingency(ct)
                        p_text = f"χ² p = {p_val:.3g}" if p_val >= 0.001 else "χ² p < 0.001"
                    else:
                        p_text = "χ²: insufficient data"
            except Exception as e:
                p_text = f"stat error: {e}"

            # annotate (axes-fraction coords)
            if p_text:
                ax.text(
                    0.02, 0.98, p_text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    zorder=20
                )

        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes[r, c].set_visible(False)

        # leave space for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_name and path:
            self.save_plot(fig, save_name, path)

        return fig

    def probability_calibration_plot(self, model_results: Dict,
                               figsize: Tuple = (16, 8), save_name: str = None, path: str = None) -> plt.Figure:
        """Plot probability calibration for binary classification"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'predicted_probabilities' in model_results and 'actual_labels' in model_results:
            from sklearn.calibration import calibration_curve
            
            y_true = model_results['actual_labels']
            y_prob = model_results['predicted_probabilities']
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                color=self.style_manager.colors['primary'], linewidth=2, markersize=8,
                label='Model Calibration')
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add Brier score if available
            if 'brier_score' in model_results:
                ax.text(0.05, 0.95, f'Brier Score: {model_results["brier_score"]:.3f}', 
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        self._handle_layout()
        
        if save_name and path:
            self.save_plot(fig, save_name, path)
        
        return fig

    def stacked_bar_plot(self, data: pd.DataFrame,
                     xlabel: str = None, ylabel: str = None,
                     horizontal: bool = False, figsize: Tuple = (16, 8),
                     save_name: str = None, path: str = None, **kwargs) -> plt.Figure:
        """Create a stacked bar plot from pivot table data"""
        fig, ax = plt.subplots(figsize=figsize)


        # Use color palette for different categories
        colors = self.style_manager.color_palette[:len(data.columns)]
    
        if horizontal:
            data.plot(kind='barh', stacked=True, ax=ax, color=colors, **kwargs)
        else:
            data.plot(kind='bar', stacked=True, ax=ax, color=colors, **kwargs)
    
        # Styling
        ax.set_xlabel(xlabel or data.index.name or 'Categories')
        ax.set_ylabel(ylabel or 'Values')
    
        # Rotate x-axis labels if needed
        if not horizontal and len(data.index) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='center', fontsize=8)
    
        self._handle_layout()
    
        if save_name and path:
            self.save_plot(fig, save_name, path)
    
        return fig


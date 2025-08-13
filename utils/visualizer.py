import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import base64
import io
from typing import Tuple, Optional

class Visualizer:
    def __init__(self):
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def create_scatterplot_with_regression(self, df: pd.DataFrame, x_col: str, y_col: str,
                                         title: str = "Scatterplot with Regression Line",
                                         max_size_kb: int = 100) -> str:
        """
        Create a scatterplot with dotted red regression line
        Returns base64 encoded image as data URI
        """
        try:
            # Find actual column names (case-insensitive)
            actual_x_col = self._find_column(df, x_col)
            actual_y_col = self._find_column(df, y_col)
            
            if not actual_x_col or not actual_y_col:
                raise Exception(f"Columns {x_col} or {y_col} not found")
            
            # Clean data
            clean_df = df[[actual_x_col, actual_y_col]].dropna()
            x = pd.to_numeric(clean_df[actual_x_col], errors='coerce')
            y = pd.to_numeric(clean_df[actual_y_col], errors='coerce')
            
            # Remove any remaining NaN values
            valid_mask = ~(x.isna() | y.isna())
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) == 0:
                raise Exception("No valid data points found")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create scatterplot
            ax.scatter(x, y, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
            
            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
            
            # Plot dotted red regression line
            ax.plot(line_x, line_y, color='red', linestyle='--', linewidth=2, 
                   label=f'Regression Line (R²={r_value**2:.3f})')
            
            # Customize plot
            ax.set_xlabel(actual_x_col.replace('_', ' ').title())
            ax.set_ylabel(actual_y_col.replace('_', ' ').title())
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert to base64
            return self._fig_to_base64(fig, max_size_kb)
            
        except Exception as e:
            print(f"Error creating scatterplot: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
        finally:
            plt.close('all')
    
    def create_court_delay_plot(self, df: pd.DataFrame, title: str = "Court Case Delay Analysis",
                               max_size_kb: int = 100) -> str:
        """
        Create a plot for court case delay analysis
        """
        try:
            # Ensure we have the right columns
            if 'year' not in df.columns or 'avg_delay_days' not in df.columns:
                # Try to create the required data
                if 'decision_date' in df.columns and 'date_of_registration' in df.columns:
                    df = self._calculate_delay_by_year(df)
                else:
                    raise Exception("Required columns for delay analysis not found")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = df['year']
            y = df['avg_delay_days']
            
            # Create scatterplot
            ax.scatter(x, y, s=80, c='blue', alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add regression line
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line_x = np.linspace(x.min(), x.max(), 100)
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, color='red', linestyle='--', linewidth=2,
                       label=f'Regression Line (slope={slope:.2f})')
            
            # Customize plot
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Delay (Days)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis to show years properly
            ax.set_xticks(x)
            ax.set_xticklabels([int(year) for year in x])
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, max_size_kb)
            
        except Exception as e:
            print(f"Error creating court delay plot: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
        finally:
            plt.close('all')
    
    def _calculate_delay_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average delay by year
        """
        # Convert dates
        df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
        df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
        
        # Calculate delay
        df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
        
        # Filter valid delays
        df = df[df['delay_days'] >= 0]
        
        # Group by year and calculate average
        df['year'] = df['decision_date'].dt.year
        result = df.groupby('year')['delay_days'].mean().reset_index()
        result.columns = ['year', 'avg_delay_days']
        
        return result
    
    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                        title: str = "Bar Chart", max_size_kb: int = 100) -> str:
        """
        Create a bar chart
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = df[x_col]
            y = df[y_col]
            
            bars = ax.bar(x, y, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color bars with a gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
            
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if they're too long
            if max(len(str(label)) for label in x) > 10:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, max_size_kb)
            
        except Exception as e:
            print(f"Error creating bar chart: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
        finally:
            plt.close('all')
    
    def create_line_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                        title: str = "Line Plot", max_size_kb: int = 100) -> str:
        """
        Create a line plot
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = df[x_col]
            y = df[y_col]
            
            ax.plot(x, y, marker='o', linewidth=2, markersize=6)
            
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, max_size_kb)
            
        except Exception as e:
            print(f"Error creating line plot: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
        finally:
            plt.close('all')
    
    def _find_column(self, df: pd.DataFrame, col_name: str) -> Optional[str]:
        """
        Find column name case-insensitively
        """
        col_name_lower = col_name.lower()
        for col in df.columns:
            if col_name_lower == col.lower() or col_name_lower in col.lower():
                return col
        return None
    
    def _fig_to_base64(self, fig, max_size_kb: int = 100) -> str:
        """
        Convert matplotlib figure to base64 data URI with size optimization
        """
        # Try different formats and qualities to get under size limit
        formats = [('png', {}), ('webp', {}), ('png', {'optimize': True})]
        
        for fmt, kwargs in formats:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=fmt, dpi=100, bbox_inches='tight', 
                       facecolor='white', **kwargs)
            
            # Check size
            buffer_size = buffer.tell()
            if buffer_size <= max_size_kb * 1024:
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode('utf-8')
                return f"data:image/{fmt};base64,{encoded}"
            
            buffer.close()
        
        # If still too large, reduce DPI and try again
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=50, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        
        return f"data:image/png;base64,{encoded}"
    
    def _create_error_plot(self, error_message: str) -> str:
        """
        Create a simple error message plot
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, error_message, ha='center', va='center', 
                fontsize=12, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Visualization Error')
        
        plt.tight_layout()
        
        result = self._fig_to_base64(fig)
        plt.close(fig)
        return result
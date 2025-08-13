import pandas as pd
import numpy as np
from scipy import stats
import duckdb
import json
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import io

class DataProcessor:
    def __init__(self):
        self.connection = None
    
    def process_csv(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Process CSV files with robust error handling
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    content = file_content.decode(encoding)
                    df = pd.read_csv(io.StringIO(content))
                    break
                except:
                    continue
            
            if df is None:
                raise Exception("Could not decode CSV file")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Try to infer data types
            df = self._infer_types(df)
            
            return df
            
        except Exception as e:
            print(f"Error processing CSV {filename}: {str(e)}")
            raise
    
    def process_excel(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Process Excel files
        """
        try:
            df = pd.read_excel(io.BytesIO(file_content))
            df.columns = df.columns.str.strip()
            df = self._infer_types(df)
            return df
        except Exception as e:
            print(f"Error processing Excel {filename}: {str(e)}")
            raise
    
    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently infer and convert data types
        """
        for col in df.columns:
            # Skip if already numeric
            if df[col].dtype in ['int64', 'float64']:
                continue
            
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                # If more than 50% can be converted to numeric, do it
                if numeric_series.notna().sum() > len(df) * 0.5:
                    df[col] = numeric_series
                    continue
            
            # Try to convert to datetime
            try:
                if df[col].dtype == 'object':
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    if datetime_series.notna().sum() > len(df) * 0.5:
                        df[col] = datetime_series
                        continue
            except:
                pass
        
        return df
    
    def analyze_films_data(self, df: pd.DataFrame, questions: List[str]) -> List[Any]:
        """
        Specific analysis for film gross data
        """
        results = []
        
        for question in questions:
            try:
                if "how many" in question.lower() and "2 bn" in question.lower() and "before 2000" in question.lower():
                    # Question 1: How many $2bn movies were released before 2000?
                    result = self._count_high_gross_before_year(df, 2_000_000_000, 2000)
                    results.append(result)
                
                elif "earliest" in question.lower() and "1.5 bn" in question.lower():
                    # Question 2: Which is the earliest film that grossed over $1.5bn?
                    result = self._find_earliest_high_gross(df, 1_500_000_000)
                    results.append(result)
                
                elif "correlation" in question.lower() and "rank" in question.lower() and "peak" in question.lower():
                    # Question 3: Correlation between Rank and Peak
                    result = self._calculate_correlation(df, 'Rank', 'Peak')
                    results.append(result)
                
                elif "scatterplot" in question.lower() or "scatter plot" in question.lower():
                    # Question 4: Scatterplot - will be handled by visualizer
                    results.append("PLOT_REQUIRED")
                
                else:
                    results.append("Question not recognized")
                    
            except Exception as e:
                print(f"Error analyzing question '{question}': {str(e)}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def _count_high_gross_before_year(self, df: pd.DataFrame, gross_threshold: float, year_threshold: int) -> int:
        """
        Count movies with gross above threshold before a certain year
        """
        # Find gross columns
        gross_columns = [col for col in df.columns if 'gross' in col.lower() or 'worldwide' in col.lower()]
        year_columns = [col for col in df.columns if 'year' in col.lower()]
        
        if not gross_columns or not year_columns:
            return 0
        
        gross_col = gross_columns[0]
        year_col = year_columns[0]
        
        # Filter data
        mask = (pd.to_numeric(df[gross_col], errors='coerce') >= gross_threshold) & \
               (pd.to_numeric(df[year_col], errors='coerce') < year_threshold)
        
        return len(df[mask])
    
    def _find_earliest_high_gross(self, df: pd.DataFrame, gross_threshold: float) -> str:
        """
        Find the earliest film that grossed over the threshold
        """
        gross_columns = [col for col in df.columns if 'gross' in col.lower() or 'worldwide' in col.lower()]
        year_columns = [col for col in df.columns if 'year' in col.lower()]
        title_columns = [col for col in df.columns if 'title' in col.lower() or 'film' in col.lower()]
        
        if not gross_columns or not year_columns or not title_columns:
            return "Required columns not found"
        
        gross_col = gross_columns[0]
        year_col = year_columns[0]
        title_col = title_columns[0]
        
        # Filter high-grossing films
        high_gross_mask = pd.to_numeric(df[gross_col], errors='coerce') >= gross_threshold
        high_gross_films = df[high_gross_mask].copy()
        
        if len(high_gross_films) == 0:
            return "No films found above threshold"
        
        # Convert year to numeric and find earliest
        high_gross_films[year_col] = pd.to_numeric(high_gross_films[year_col], errors='coerce')
        earliest_film = high_gross_films.loc[high_gross_films[year_col].idxmin()]
        
        return str(earliest_film[title_col])
    
    def _calculate_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """
        Calculate correlation between two columns
        """
        # Find matching columns (case-insensitive)
        actual_col1 = None
        actual_col2 = None
        
        for col in df.columns:
            if col1.lower() in col.lower():
                actual_col1 = col
            if col2.lower() in col.lower():
                actual_col2 = col
        
        if not actual_col1 or not actual_col2:
            return 0.0
        
        # Convert to numeric
        series1 = pd.to_numeric(df[actual_col1], errors='coerce')
        series2 = pd.to_numeric(df[actual_col2], errors='coerce')
        
        # Calculate correlation
        correlation = series1.corr(series2)
        
        return round(correlation, 6) if not pd.isna(correlation) else 0.0
    
    def query_duckdb(self, query: str) -> pd.DataFrame:
        """
        Execute DuckDB query
        """
        try:
            if not self.connection:
                self.connection = duckdb.connect()
                # Install required extensions
                self.connection.execute("INSTALL httpfs;")
                self.connection.execute("LOAD httpfs;")
                self.connection.execute("INSTALL parquet;")
                self.connection.execute("LOAD parquet;")
            
            result = self.connection.execute(query).df()
            return result
            
        except Exception as e:
            print(f"DuckDB query error: {str(e)}")
            raise
    
    def analyze_court_data(self, questions: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze Indian court judgment data
        """
        results = {}
        
        for question, _ in questions.items():
            try:
                if "high court disposed the most cases from 2019" in question.lower():
                    # Query for most cases disposed
                    query = """
                    SELECT court, COUNT(*) as case_count
                    FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
                    WHERE year BETWEEN 2019 AND 2022
                    GROUP BY court
                    ORDER BY case_count DESC
                    LIMIT 1
                    """
                    result_df = self.query_duckdb(query)
                    results[question] = result_df.iloc[0]['court'] if len(result_df) > 0 else "No data"
                
                elif "regression slope" in question.lower():
                    # Calculate regression slope for delay analysis
                    query = """
                    SELECT 
                        year,
                        AVG(DATEDIFF('day', CAST(date_of_registration AS DATE), decision_date)) as avg_delay_days
                    FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
                    WHERE court = '33_10' 
                    AND date_of_registration IS NOT NULL 
                    AND decision_date IS NOT NULL
                    AND year BETWEEN 2019 AND 2023
                    GROUP BY year
                    ORDER BY year
                    """
                    result_df = self.query_duckdb(query)
                    
                    if len(result_df) > 1:
                        # Calculate regression slope
                        x = result_df['year'].values
                        y = result_df['avg_delay_days'].values
                        slope, _, _, _, _ = stats.linregress(x, y)
                        results[question] = round(slope, 6)
                    else:
                        results[question] = "Insufficient data"
                
                elif "plot" in question.lower() and "scatterplot" in question.lower():
                    # This will be handled by visualizer
                    results[question] = "PLOT_REQUIRED"
                
                else:
                    results[question] = "Question type not recognized"
                    
            except Exception as e:
                print(f"Error processing court data question '{question}': {str(e)}")
                results[question] = f"Error: {str(e)}"
        
        return results
    
    def calculate_delay_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate delay statistics for court cases
        """
        try:
            # Convert date columns
            df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
            df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
            
            # Calculate delay in days
            df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
            
            # Remove invalid delays
            df = df[df['delay_days'] >= 0]
            
            stats_dict = {
                'mean_delay': df['delay_days'].mean(),
                'median_delay': df['delay_days'].median(),
                'max_delay': df['delay_days'].max(),
                'min_delay': df['delay_days'].min(),
                'std_delay': df['delay_days'].std()
            }
            
            return stats_dict
            
        except Exception as e:
            print(f"Error calculating delay statistics: {str(e)}")
            return {}
    
    def group_and_aggregate(self, df: pd.DataFrame, group_cols: List[str], 
                          agg_col: str, agg_func: str = 'count') -> pd.DataFrame:
        """
        Generic grouping and aggregation function
        """
        try:
            if agg_func == 'count':
                result = df.groupby(group_cols).size().reset_index(name='count')
            elif agg_func == 'sum':
                result = df.groupby(group_cols)[agg_col].sum().reset_index()
            elif agg_func == 'mean':
                result = df.groupby(group_cols)[agg_col].mean().reset_index()
            elif agg_func == 'max':
                result = df.groupby(group_cols)[agg_col].max().reset_index()
            elif agg_func == 'min':
                result = df.groupby(group_cols)[agg_col].min().reset_index()
            else:
                result = df.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
            
            return result
            
        except Exception as e:
            print(f"Error in grouping and aggregation: {str(e)}")
            return pd.DataFrame()
    
    def find_extremes(self, df: pd.DataFrame, column: str, 
                     extreme_type: str = 'max') -> Any:
        """
        Find maximum or minimum values
        """
        try:
            if column not in df.columns:
                # Try to find similar column names
                similar_cols = [col for col in df.columns if column.lower() in col.lower()]
                if similar_cols:
                    column = similar_cols[0]
                else:
                    return None
            
            if extreme_type == 'max':
                return df[column].max()
            elif extreme_type == 'min':
                return df[column].min()
            elif extreme_type == 'idxmax':
                return df.loc[df[column].idxmax()]
            elif extreme_type == 'idxmin':
                return df.loc[df[column].idxmin()]
            else:
                return None
                
        except Exception as e:
            print(f"Error finding extremes: {str(e)}")
            return None
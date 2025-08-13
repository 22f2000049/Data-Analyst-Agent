import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from typing import Dict, List, Optional

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_wikipedia_highest_grossing_films(self, url: str) -> pd.DataFrame:
        """
        Scrape the Wikipedia page for highest grossing films
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main table with highest grossing films
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            # Usually the first sortable table contains the main data
            main_table = None
            for table in tables:
                if 'sortable' in table.get('class', []):
                    main_table = table
                    break
            
            if not main_table:
                main_table = tables[0] if tables else None
                
            if not main_table:
                raise Exception("Could not find the main data table")
            
            # Extract table data
            headers = []
            header_row = main_table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text().strip())
            
            # Extract rows
            rows = []
            for tr in main_table.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all(['td', 'th']):
                    text = td.get_text().strip()
                    # Clean up the text (remove references, etc.)
                    text = re.sub(r'\[.*?\]', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    row.append(text)
                if row:
                    rows.append(row)
            
            # Create DataFrame
            if headers and rows:
                # Ensure all rows have the same length as headers
                max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
                
                # Pad headers if necessary
                while len(headers) < max_cols:
                    headers.append(f'Column_{len(headers)}')
                
                # Pad rows if necessary
                for row in rows:
                    while len(row) < max_cols:
                        row.append('')
                
                df = pd.DataFrame(rows, columns=headers[:max_cols])
                return self._clean_film_data(df)
            else:
                raise Exception("Could not extract table data")
                
        except Exception as e:
            print(f"Error scraping Wikipedia: {str(e)}")
            raise
    
    def _clean_film_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the film data
        """
        try:
            # Common column name mappings
            column_mapping = {
                'Rank': 'Rank',
                'Peak': 'Peak',
                'Title': 'Title',
                'Worldwide gross': 'Worldwide_Gross',
                'Year': 'Year',
                'Reference': 'Reference'
            }
            
            # Find and rename columns
            new_columns = {}
            for col in df.columns:
                for key, value in column_mapping.items():
                    if key.lower() in col.lower():
                        new_columns[col] = value
                        break
                else:
                    # If no mapping found, keep original but clean it
                    new_columns[col] = col.replace(' ', '_').replace('/', '_')
            
            df = df.rename(columns=new_columns)
            
            # Clean numeric columns
            numeric_columns = ['Rank', 'Peak', 'Year']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
            
            # Clean gross amounts (convert to numeric)
            gross_columns = [col for col in df.columns if 'gross' in col.lower() or 'box' in col.lower()]
            for col in gross_columns:
                if col in df.columns:
                    # Extract numeric value from strings like "$2.798 billion"
                    df[col] = df[col].astype(str).apply(self._extract_gross_amount)
            
            # Extract year from title if Year column doesn't exist
            if 'Year' not in df.columns and 'Title' in df.columns:
                df['Year'] = df['Title'].astype(str).str.extract(r'\((\d{4})\)')[0]
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error cleaning film data: {str(e)}")
            return df
    
    def _extract_gross_amount(self, text: str) -> float:
        """
        Extract gross amount from text like "$2.798 billion" and convert to numeric
        """
        try:
            if pd.isna(text) or text == '':
                return None
            
            text = str(text).lower()
            
            # Extract number
            number_match = re.search(r'(\d+\.?\d*)', text)
            if not number_match:
                return None
            
            number = float(number_match.group(1))
            
            # Apply multiplier
            if 'billion' in text:
                return number * 1_000_000_000
            elif 'million' in text:
                return number * 1_000_000
            else:
                return number
                
        except:
            return None
    
    def scrape_generic_table(self, url: str) -> pd.DataFrame:
        """
        Generic table scraper for other websites
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Try pandas read_html first (often works well)
            try:
                tables = pd.read_html(response.content)
                if tables:
                    return tables[0]  # Return first table
            except:
                pass
            
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if not tables:
                raise Exception("No tables found on the page")
            
            # Process first table
            table = tables[0]
            rows = []
            headers = []
            
            # Get headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Get data rows
            for tr in table.find_all('tr')[1:]:
                row = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if row:
                    rows.append(row)
            
            if headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                return df
            else:
                raise Exception("Could not extract table data")
                
        except Exception as e:
            print(f"Error scraping table: {str(e)}")
            raise
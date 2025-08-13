import google.generativeai as genai
from config import Config
import json
import re
from typing import Dict, Any, Optional

class GeminiHelper:
    def __init__(self):
        if Config.GEMINI_API_KEY:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            print("Warning: Gemini API key not found. AI features will be limited.")
    
    def analyze_question(self, question: str, data_info: str = "") -> Dict[str, Any]:
        """
        Analyze the question to understand what kind of analysis is needed.
        Only use AI when absolutely necessary to save tokens.
        """
        # First try rule-based analysis (no AI tokens used)
        analysis = self._rule_based_analysis(question)
        
        # Only use AI if rule-based analysis is insufficient
        if analysis['confidence'] < 0.7 and self.model:
            try:
                analysis = self._ai_analysis(question, data_info)
            except Exception as e:
                print(f"AI analysis failed: {e}")
                # Fall back to rule-based
                pass
        
        return analysis
    
    def _rule_based_analysis(self, question: str) -> Dict[str, Any]:
        """
        Rule-based question analysis to save AI tokens
        """
        question_lower = question.lower()
        
        analysis = {
            'type': 'unknown',
            'needs_scraping': False,
            'needs_visualization': False,
            'data_source': None,
            'output_format': 'text',
            'confidence': 0.5
        }
        
        # Check for scraping needs
        if any(keyword in question_lower for keyword in ['scrape', 'wikipedia', 'website', 'url', 'web']):
            analysis['needs_scraping'] = True
            analysis['confidence'] += 0.2
        
        # Check for visualization needs
        if any(keyword in question_lower for keyword in ['plot', 'chart', 'graph', 'visuali', 'scatter', 'histogram', 'bar chart']):
            analysis['needs_visualization'] = True
            analysis['output_format'] = 'image'
            analysis['confidence'] += 0.2
        
        # Check for specific analysis types
        if any(keyword in question_lower for keyword in ['correlation', 'regression', 'slope']):
            analysis['type'] = 'statistical'
            analysis['confidence'] += 0.2
        
        if any(keyword in question_lower for keyword in ['count', 'how many', 'number of']):
            analysis['type'] = 'counting'
            analysis['confidence'] += 0.2
        
        if any(keyword in question_lower for keyword in ['earliest', 'latest', 'first', 'last', 'minimum', 'maximum']):
            analysis['type'] = 'extremes'
            analysis['confidence'] += 0.2
        
        # Check for specific data sources
        if 'wikipedia' in question_lower:
            analysis['data_source'] = 'wikipedia'
            analysis['confidence'] += 0.1
        
        if any(keyword in question_lower for keyword in ['duckdb', 'sql', 'query', 'parquet']):
            analysis['data_source'] = 'database'
            analysis['confidence'] += 0.1
        
        return analysis
    
    def _ai_analysis(self, question: str, data_info: str = "") -> Dict[str, Any]:
        """
        AI-based question analysis (uses tokens - use sparingly!)
        """
        prompt = f"""
        Analyze this data analysis question and provide a structured response:
        
        Question: {question}
        Data Context: {data_info}
        
        Provide analysis in this exact JSON format:
        {{
            "type": "statistical|counting|extremes|visualization|comparison",
            "needs_scraping": true|false,
            "needs_visualization": true|false,
            "data_source": "wikipedia|database|csv|other",
            "output_format": "text|number|image|json",
            "key_operations": ["operation1", "operation2"],
            "confidence": 0.9
        }}
        
        Only respond with the JSON, no extra text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip())
            return result
        except Exception as e:
            print(f"AI analysis error: {e}")
            return self._rule_based_analysis(question)
    
    def generate_code_snippet(self, question: str, data_description: str) -> str:
        """
        Generate Python code for data analysis (use only when really stuck!)
        """
        if not self.model:
            return "# AI helper not available - implement manually"
        
        prompt = f"""
        Generate Python code to answer this question:
        Question: {question}
        Data: {data_description}
        
        Requirements:
        - Use pandas for data manipulation
        - Use matplotlib/seaborn for plotting
        - Return the answer in the required format
        - Handle errors gracefully
        - Code should be production-ready
        
        Only provide the code, no explanations:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Code generation error: {e}")
            return "# Error generating code"
    
    def interpret_results(self, question: str, results: Any) -> str:
        """
        Interpret analysis results (use sparingly!)
        """
        # Only use AI for complex interpretations
        if not self.model or len(str(results)) > 1000:
            return str(results)  # Return as-is for simple cases
        
        prompt = f"""
        Question: {question}
        Results: {results}
        
        Provide a brief, clear interpretation of these results in relation to the question.
        Keep it under 100 words and be precise.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Interpretation error: {e}")
            return str(results)

# Global instance
gemini_helper = GeminiHelper()
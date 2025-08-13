# Data Analyst Agent

A Flask API that uses AI to source, prepare, analyze, and visualize data automatically.

## Features

- **Web Scraping**: Automatically scrape data from Wikipedia and other sources
- **Data Processing**: Handle CSV, Excel, and other data formats
- **Statistical Analysis**: Perform correlations, regressions, and other statistical operations
- **Visualization**: Create charts, plots, and graphs with automatic encoding to base64
- **AI Integration**: Uses Gemini AI for intelligent question analysis and code generation
- **Database Queries**: Support for DuckDB queries on large datasets

## API Usage

### Endpoint
```
POST /api/
```

### Request Format
Send a multipart form with:
- `questions.txt` (required): Contains the analysis questions
- Additional files (optional): CSV, Excel, images, etc.

### Example Request
```bash
curl "https://your-api-endpoint.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

### Response Format
- For array questions: Returns JSON array with answers
- For object questions: Returns JSON object with key-value pairs

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd data-analyst-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **Run the application**
```bash
python app.py
```

## Deployment

### Using Railway (Recommended)

1. Create account at [Railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variable: `GEMINI_API_KEY`
4. Deploy automatically

### Using Render

1. Create account at [Render.com](https://render.com)
2. Connect your GitHub repository  
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Add environment variable: `GEMINI_API_KEY`

### Using Heroku

1. Create account at [Heroku.com](https://heroku.com)
2. Install Heroku CLI
3. Create app: `heroku create your-app-name`
4. Set config: `heroku config:set GEMINI_API_KEY=your_key`
5. Deploy: `git push heroku main`

## Supported Analysis Types

### Wikipedia Film Analysis
- Scrapes highest grossing films data
- Counts movies by criteria
- Finds earliest/latest films
- Calculates correlations
- Creates visualizations

### Court Data Analysis
- Queries Indian High Court judgment database
- Analyzes case disposal patterns
- Calculates delay statistics
- Creates trend visualizations

### CSV Analysis
- Processes uploaded CSV files
- Performs statistical analysis
- Creates various chart types
- Handles missing data automatically

## Project Structure

```
data-analyst-agent/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Dependencies
├── utils/                # Utility modules
│   ├── scraper.py        # Web scraping
│   ├── data_processor.py # Data analysis
│   ├── visualizer.py     # Chart creation
│   └── ai_helper.py      # AI integration
├── tests/                # Test files
├── uploads/              # Temporary storage
└── LICENSE               # MIT License
```

## Environment Variables

- `GEMINI_API_KEY`: Google Gemini API key for AI features
- `FLASK_ENV`: Set to `production` for deployment
- `PORT`: Port number (automatically set by hosting platforms)

## Error Handling

The API includes comprehensive error handling:
- Invalid file formats
- Missing required fields
- Analysis timeouts
- AI API failures
- Data processing errors

All errors return appropriate HTTP status codes and error messages.

## Performance

- **Timeout**: 5 minutes per request
- **File Size**: Maximum 50MB per file
- **Image Output**: Automatically optimized under 100KB
- **Caching**: Built-in data caching for repeated queries

## Testing

Test your deployment:

```bash
curl -X POST "https://your-api-endpoint.com/api/" \
  -F "questions.txt=@tests/test_questions.txt"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the error logs in your deployment platform
2. Verify your Gemini API key is valid
3. Ensure all required files are present
4. Check the file format matches requirements

## Changelog

### v1.0.0
- Initial release
- Wikipedia scraping support
- Court data analysis
- CSV processing
- Basic visualization
- Gemini AI integration

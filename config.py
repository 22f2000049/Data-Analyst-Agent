import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_FILE_SIZE', 50000000))  # 50MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    
    # API settings
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 180))  # 3 minutes
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'pdf'}

# Create upload directory if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
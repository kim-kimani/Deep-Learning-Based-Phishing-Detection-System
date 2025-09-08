#!/usr/bin/env python3
"""
Quick Setup Script for Multi-Modal Phishing Detection System
Installs dependencies and configures the system for multi-modal input processing
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"📋 {description}")
    print(f"🔄 Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required for this system")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version compatible: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install all required dependencies"""
    print("🚀 Installing Multi-Modal Dependencies...")
    
    # Core requirements
    core_deps = [
        "Django>=5.2",
        "djangorestframework",
        "django-cors-headers",
        "django-filter",
        "requests",
        "colorama",
        "beautifulsoup4",
        "tldextract",
        "joblib",
        "pandas",
        "scikit-learn",
        "numpy",
        "scipy"
    ]
    
    # Multi-modal specific dependencies
    multimodal_deps = [
        "Pillow>=10.0.0",
        "pytesseract>=0.3.10",
        "opencv-python>=4.8.0",
        "easyocr>=1.7.0",
        "SpeechRecognition>=3.10.0",
        "pydub>=0.25.1",
        "PyPDF2>=3.0.1",
        "python-docx>=0.8.11",
        "python-magic>=0.4.27",
        "qrcode>=7.4.2",
        "pyzbar>=0.1.9",
        "selenium>=4.15.0",
        "webdriver-manager>=4.0.1"
    ]
    
    # Background processing
    async_deps = [
        "celery>=5.3.0",
        "redis>=5.0.0",
        "django-celery-beat>=2.5.0",
        "django-celery-results>=2.5.1"
    ]
    
    all_deps = core_deps + multimodal_deps + async_deps
    
    # Install all dependencies
    for dep in all_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    print("✅ Core dependencies installation completed")

def install_system_dependencies():
    """Install system-level dependencies"""
    print("🔧 Installing System Dependencies...")
    
    # Tesseract OCR
    print("📋 Installing Tesseract OCR...")
    print("   For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   For Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print("   For macOS: brew install tesseract")
    
    # FFmpeg for audio processing
    print("📋 Installing FFmpeg...")
    print("   For Windows: Download from https://ffmpeg.org/download.html")
    print("   For Ubuntu/Debian: sudo apt-get install ffmpeg")
    print("   For macOS: brew install ffmpeg")
    
    # Chrome/Chromium for web automation
    print("📋 Chrome/Chromium for web automation...")
    print("   Make sure Google Chrome or Chromium is installed")
    print("   WebDriver will be managed automatically by webdriver-manager")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "static/js",
        "static/css", 
        "static/images",
        "media/uploads",
        "media/temp",
        "logs",
        "temp_uploads",
        "temp_multimodal"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_django_setup():
    """Run Django migrations and setup"""
    print("🗄️  Setting up Django...")
    
    commands = [
        ("python manage.py makemigrations", "Creating migrations"),
        ("python manage.py migrate", "Running migrations"),
        ("python manage.py collectstatic --noinput", "Collecting static files")
    ]
    
    for command, description in commands:
        run_command(command, description)

def test_capabilities():
    """Test multi-modal capabilities"""
    print("🧪 Testing Multi-Modal Capabilities...")
    
    try:
        # Test image processing
        import PIL
        print("✅ PIL (Image processing) available")
    except ImportError:
        print("❌ PIL not available")
    
    try:
        # Test OCR
        import pytesseract
        print("✅ Pytesseract (OCR) available")
    except ImportError:
        print("❌ Pytesseract not available")
    
    try:
        # Test speech recognition
        import speech_recognition
        print("✅ SpeechRecognition available")
    except ImportError:
        print("❌ SpeechRecognition not available")
    
    try:
        # Test document processing
        import PyPDF2
        import docx
        print("✅ Document processing (PDF, DOCX) available")
    except ImportError:
        print("❌ Document processing libraries not available")
    
    try:
        # Test web automation
        import selenium
        print("✅ Selenium (Web automation) available")
    except ImportError:
        print("❌ Selenium not available")

def create_sample_data():
    """Create sample data for testing"""
    print("📋 Creating sample test data...")
    
    # Create a simple test script
    test_script = '''#!/usr/bin/env python3
"""
Multi-Modal System Test Script
"""

import requests
import os

def test_text_analysis():
    """Test text analysis endpoint"""
    url = "http://localhost:8000/predict/multimodal/"
    data = {
        'text': 'URGENT! Your account has been suspended. Click here to verify: http://suspicious-link.com',
        'prediction_type': 'email'
    }
    
    try:
        response = requests.post(url, data=data)
        print(f"Text Analysis Test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {result.get('result', {}).get('prediction', 'Unknown')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Text analysis test failed: {e}")
        return False

def test_capabilities():
    """Test capabilities endpoint"""
    url = "http://localhost:8000/capabilities/"
    
    try:
        response = requests.get(url)
        print(f"Capabilities Test: {response.status_code}")
        if response.status_code == 200:
            caps = response.json()
            print(f"Multi-modal available: {caps.get('multimodal_available', False)}")
            print(f"Enhanced service available: {caps.get('enhanced_service_available', False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Capabilities test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Multi-Modal System Tests...")
    print("Make sure the Django server is running on localhost:8000")
    
    tests = [
        ("Text Analysis", test_text_analysis),
        ("Capabilities", test_capabilities)
    ]
    
    for test_name, test_func in tests:
        print(f"\\n🔄 Running {test_name} test...")
        success = test_func()
        print(f"{'✅' if success else '❌'} {test_name} test {'passed' if success else 'failed'}")
'''
    
    with open('test_multimodal.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_multimodal.py - use this to test the system")

def main():
    """Main setup function"""
    print("🛡️  Multi-Modal Phishing Detection System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Show system dependencies info
    install_system_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Run Django setup
    run_django_setup()
    
    # Test capabilities
    test_capabilities()
    
    # Create sample data
    create_sample_data()
    
    print("\n🎉 Setup Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Install system dependencies (Tesseract, FFmpeg, Chrome)")
    print("2. Start the Django server: python manage.py runserver")
    print("3. Visit http://localhost:8000/multimodal/ for the multi-modal interface")
    print("4. Visit http://localhost:8000/ for the original interface")
    print("5. Run python test_multimodal.py to test the system")
    print("\n🔧 Optional: Start Redis and Celery for background processing:")
    print("   redis-server")
    print("   celery -A ml_security_detector worker --loglevel=info")

if __name__ == "__main__":
    main()

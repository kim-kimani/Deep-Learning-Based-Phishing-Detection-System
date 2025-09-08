# Multi-Modal Phishing Detection System

## 🚀 Overview

This enhanced phishing detection system now supports multiple input methods beyond traditional text analysis. Users can upload images, record voice messages, submit documents, and analyze URLs with advanced AI-powered threat detection.

## 🎯 New Features

### 1. Multi-Modal Input Support
- **Image Analysis**: Upload screenshots, photos of screens, or any image containing suspicious text
- **Voice Input**: Record suspicious voice messages or upload audio files for analysis
- **Document Processing**: Analyze PDFs, Word documents, emails (.eml), and text files
- **Enhanced URL Analysis**: Capture website screenshots and analyze visual content
- **QR Code Scanning**: Extract and analyze URLs from QR codes in images

### 2. Advanced Processing Capabilities
- **OCR (Optical Character Recognition)**: Extract text from images using Tesseract and EasyOCR
- **Speech-to-Text**: Convert audio content to text for analysis
- **Document Parsing**: Extract content from various document formats
- **Web Automation**: Automatically capture website screenshots for visual analysis
- **Batch Processing**: Analyze multiple files or URLs simultaneously

### 3. Enhanced User Interface
- **Drag-and-Drop File Upload**: Intuitive file handling with visual feedback
- **Voice Recording**: Built-in microphone recording with real-time feedback
- **Progress Indicators**: Visual progress tracking for file processing
- **Multi-Tab Interface**: Organized input methods for different content types
- **Real-Time Previews**: Preview uploaded content before analysis

## 📁 Project Structure

```
spam-detection-system/
├── detector/
│   ├── multimodal_service.py      # Core multi-modal processing engine
│   ├── multimodal_views.py        # Django views for multi-modal endpoints
│   ├── enhanced_deep_learning_service.py  # ML/AI analysis service
│   └── models.py                  # Database models
├── templates/detector/
│   └── multimodal.html           # Multi-modal interface template
├── static/js/
│   └── multimodal.js             # Frontend JavaScript logic
├── requirements/
│   └── multimodal_requirements.txt  # Additional dependencies
├── setup_multimodal.py          # Quick setup script
└── test_multimodal.py           # System testing script
```

## 🛠️ Installation & Setup

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd spam-detection-system

# Run the automated setup script
python setup_multimodal.py
```

### Manual Setup

1. **Install Python Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements/multimodal_requirements.txt
```

2. **Install System Dependencies**

**Windows:**
- Download Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
- Download FFmpeg: https://ffmpeg.org/download.html
- Install Google Chrome

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr ffmpeg
sudo apt-get install google-chrome-stable
```

**macOS:**
```bash
brew install tesseract ffmpeg
brew install --cask google-chrome
```

3. **Configure Django**
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic
```

4. **Start the Server**
```bash
python manage.py runserver
```

## 🎮 Usage Guide

### Accessing the Multi-Modal Interface

1. **Traditional Interface**: http://localhost:8000/
2. **Multi-Modal Interface**: http://localhost:8000/multimodal/

### Input Methods

#### 1. File Upload
- **Supported Formats**: 
  - Images: JPG, PNG, GIF, WebP, BMP
  - Audio: MP3, WAV, M4A, OGG
  - Documents: PDF, DOCX, DOC, TXT, EML, MSG
- **File Size Limits**:
  - Images: 50MB
  - Audio: 100MB
  - Documents: 20MB

#### 2. Voice Input
- Click the record button to start recording
- Speak your message clearly
- Click stop when finished
- Alternative: Upload pre-recorded audio files

#### 3. Text Analysis
- Direct text input with auto-detection of content type
- Manual content type selection (Email, SMS, URL)
- Supports large text blocks

#### 4. URL Analysis
- Enter suspicious URLs
- Option to capture screenshots
- OCR analysis on captured screenshots
- Automatic threat detection

#### 5. Batch Processing
- Upload multiple files simultaneously
- Analyze multiple URLs at once
- Progress tracking for large batches
- Comprehensive results summary

## 🔧 API Endpoints

### Multi-Modal Analysis
```
POST /predict/multimodal/
Content-Type: multipart/form-data

Parameters:
- file: Upload file (optional)
- text: Direct text input (optional)  
- url: URL to analyze (optional)
- prediction_type: auto|email|sms|url
- enable_ocr: boolean
- enhanced_processing: boolean
- capture_screenshot: boolean
- ocr_screenshot: boolean
```

### Batch Processing
```
POST /predict/batch/
Content-Type: multipart/form-data

Parameters:
- files: Multiple files
- urls: JSON array of URLs
- capture_screenshots: boolean
- parallel_processing: boolean
```

### System Capabilities
```
GET /capabilities/

Response:
{
  "multimodal_available": boolean,
  "enhanced_service_available": boolean,
  "capabilities": {
    "image_processing": boolean,
    "ocr_processing": boolean,
    "speech_processing": boolean,
    "document_processing": boolean,
    "qr_processing": boolean,
    "web_automation": boolean
  },
  "supported_formats": {
    "images": [...],
    "audio": [...],
    "documents": [...]
  }
}
```

## 🧠 AI/ML Processing Pipeline

### Two-Stage Analysis System

1. **Deep Learning Analysis**
   - Initial threat detection using neural networks
   - If malicious: Return result immediately
   - If safe: Proceed to LLM analysis

2. **LLM Enhancement (Groq AI)**
   - Advanced linguistic analysis
   - Context understanding
   - Final threat assessment
   - Detailed explanation

### Processing Flow

```
Input → Validation → Content Extraction → AI Analysis → Results
  ↓         ↓             ↓                ↓           ↓
File     Security    OCR/Speech/      Deep Learning  Enhanced
Upload   Check       Document         + LLM         Response
                    Parsing
```

## 🔒 Security Features

### File Security
- File type validation and sanitization
- Malware signature detection
- Size limitations to prevent abuse
- Secure temporary file handling
- Automatic cleanup of processed files

### Input Validation
- Comprehensive file header verification
- Content type validation
- Security scanning for executable content
- Rate limiting for API endpoints

### Data Protection
- No permanent storage of uploaded content
- Encrypted temporary file storage
- Automatic cleanup after processing
- Privacy-focused design

## 📊 Performance & Scalability

### Optimization Features
- **Parallel Processing**: Multiple files processed simultaneously
- **Caching**: Intelligent caching of OCR and analysis results
- **Background Tasks**: Celery integration for heavy processing
- **Progressive Loading**: Streaming results for large batches

### Resource Management
- **Memory Efficient**: Streaming file processing
- **CPU Optimization**: Multi-threaded analysis
- **Storage Management**: Automatic cleanup and size limits
- **Network Optimization**: Compressed responses

## 🧪 Testing

### Automated Testing
```bash
# Run the comprehensive test suite
python test_multimodal.py

# Test specific capabilities
python -c "from detector.multimodal_service import multimodal_processor; print(multimodal_processor.get_capabilities())"
```

### Manual Testing
1. Upload test images with text
2. Record voice messages
3. Submit sample documents
4. Analyze suspicious URLs
5. Test batch processing

## 🚨 Troubleshooting

### Common Issues

**1. OCR Not Working**
- Ensure Tesseract is installed and in PATH
- Check image quality and resolution
- Verify supported image formats

**2. Speech Recognition Fails**
- Check microphone permissions
- Ensure audio file format is supported
- Verify internet connection for Google Speech API

**3. Document Processing Errors**
- Check file format compatibility
- Verify file is not corrupted
- Ensure sufficient disk space

**4. Web Automation Issues**
- Install Google Chrome/Chromium
- Check network connectivity
- Verify webdriver-manager setup

### Debug Mode
Enable debug logging in Django settings:
```python
LOGGING = {
    'loggers': {
        'detector': {
            'level': 'DEBUG',
        },
    },
}
```

## 📈 Monitoring & Analytics

### System Metrics
- Processing success rates
- Average processing times
- Resource utilization
- Error rates by input type

### Usage Analytics
- Input method popularity
- File type distribution
- Processing volume trends
- User interaction patterns

## 🔮 Future Enhancements

### Planned Features
- **Real-time Analysis**: Live processing during file upload
- **Advanced OCR**: Support for handwritten text
- **Multi-language Support**: International threat detection
- **Mobile App**: Dedicated mobile application
- **API Rate Limiting**: Advanced quota management
- **Cloud Storage**: Integration with cloud storage providers

### Experimental Features
- **Video Analysis**: Processing of video content
- **Social Media Integration**: Direct platform analysis
- **Browser Extension**: Real-time web protection
- **AI Model Training**: Custom model development

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Follow PEP 8 coding standards
5. Add comprehensive tests
6. Submit pull request

### Code Quality
- **Linting**: Use flake8 and pylint
- **Testing**: Maintain test coverage above 80%
- **Documentation**: Update docs for new features
- **Security**: Follow OWASP guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Tesseract OCR**: Google's open-source OCR engine
- **EasyOCR**: JaidedAI's OCR library
- **OpenCV**: Computer vision library
- **Django**: Web framework
- **Groq**: LLM API services
- **Contributors**: All developers who contributed to this project

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting guide
- Review the API documentation
- Join our community discussions

---

**Multi-Modal Phishing Detection System** - Protecting users through advanced AI and comprehensive input analysis.

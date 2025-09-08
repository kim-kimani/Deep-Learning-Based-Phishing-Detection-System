"""
Multi-Modal Input Processing Service
Handles image, voice, document, and enhanced URL analysis for phishing detection
"""

import os
import io
import base64
import tempfile
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import mimetypes
import hashlib
import json

# Core Django imports
from django.core.files.uploadedfile import UploadedFile
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

# Image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# OCR engines
try:
    import pytesseract
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Speech processing
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    import librosa
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Document processing
try:
    import PyPDF2
    from docx import Document
    import openpyxl
    DOCUMENT_AVAILABLE = True
except ImportError:
    DOCUMENT_AVAILABLE = False

# QR Code processing
try:
    from pyzbar import pyzbar
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# Web automation
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    WEB_AUTOMATION_AVAILABLE = True
except ImportError:
    WEB_AUTOMATION_AVAILABLE = False

# File validation
try:
    import magic
    FILE_VALIDATION_AVAILABLE = True
except ImportError:
    FILE_VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiModalProcessor:
    """Main class for processing multi-modal inputs"""
    
    # Supported file types and limits
    SUPPORTED_IMAGE_TYPES = {
        'image/jpeg': '.jpg',
        'image/png': '.png', 
        'image/gif': '.gif',
        'image/webp': '.webp',
        'image/bmp': '.bmp'
    }
    
    SUPPORTED_AUDIO_TYPES = {
        'audio/mpeg': '.mp3',
        'audio/wav': '.wav',
        'audio/x-m4a': '.m4a',
        'audio/ogg': '.ogg',
        'audio/webm': '.webm'
    }
    
    SUPPORTED_DOCUMENT_TYPES = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/msword': '.doc',
        'text/plain': '.txt',
        'message/rfc822': '.eml',
        'application/vnd.ms-outlook': '.msg'
    }
    
    # File size limits (in bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB  
    MAX_DOCUMENT_SIZE = 20 * 1024 * 1024  # 20MB
    
    def __init__(self):
        """Initialize the multi-modal processor"""
        self.temp_dir = Path(tempfile.gettempdir()) / "spam_detector_temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize OCR readers
        self.easyocr_reader = None
        if OCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
        
        # Initialize speech recognizer
        self.speech_recognizer = None
        if SPEECH_AVAILABLE:
            self.speech_recognizer = sr.Recognizer()
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return available processing capabilities"""
        return {
            'image_processing': VISION_AVAILABLE,
            'ocr_processing': OCR_AVAILABLE,
            'speech_processing': SPEECH_AVAILABLE,
            'document_processing': DOCUMENT_AVAILABLE,
            'qr_processing': QR_AVAILABLE,
            'web_automation': WEB_AUTOMATION_AVAILABLE,
            'file_validation': FILE_VALIDATION_AVAILABLE
        }
    
    def validate_file(self, file: UploadedFile) -> Dict[str, Any]:
        """Validate uploaded file for security and type compliance"""
        result = {
            'valid': False,
            'file_type': None,
            'detected_type': None,
            'size': file.size,
            'security_check': False,
            'errors': []
        }
        
        try:
            # Check file size
            if file.size > max(self.MAX_IMAGE_SIZE, self.MAX_AUDIO_SIZE, self.MAX_DOCUMENT_SIZE):
                result['errors'].append(f"File too large: {file.size} bytes")
                return result
            
            # Detect actual file type
            if FILE_VALIDATION_AVAILABLE:
                file_header = file.read(2048)
                file.seek(0)  # Reset file pointer
                detected_mime = magic.from_buffer(file_header, mime=True)
                result['detected_type'] = detected_mime
            else:
                detected_mime = mimetypes.guess_type(file.name)[0]
                result['detected_type'] = detected_mime
            
            # Check if file type is supported
            all_supported = {**self.SUPPORTED_IMAGE_TYPES, **self.SUPPORTED_AUDIO_TYPES, **self.SUPPORTED_DOCUMENT_TYPES}
            
            if detected_mime in all_supported:
                result['file_type'] = detected_mime
                result['valid'] = True
                
                # Additional size checks by type
                if detected_mime in self.SUPPORTED_IMAGE_TYPES and file.size > self.MAX_IMAGE_SIZE:
                    result['errors'].append(f"Image too large: {file.size} bytes (max: {self.MAX_IMAGE_SIZE})")
                    result['valid'] = False
                elif detected_mime in self.SUPPORTED_AUDIO_TYPES and file.size > self.MAX_AUDIO_SIZE:
                    result['errors'].append(f"Audio too large: {file.size} bytes (max: {self.MAX_AUDIO_SIZE})")
                    result['valid'] = False
                elif detected_mime in self.SUPPORTED_DOCUMENT_TYPES and file.size > self.MAX_DOCUMENT_SIZE:
                    result['errors'].append(f"Document too large: {file.size} bytes (max: {self.MAX_DOCUMENT_SIZE})")
                    result['valid'] = False
            else:
                result['errors'].append(f"Unsupported file type: {detected_mime}")
            
            # Basic security check (file header validation)
            result['security_check'] = self._basic_security_check(file)
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"File validation error: {e}")
        
        return result
    
    def _basic_security_check(self, file: UploadedFile) -> bool:
        """Perform basic security checks on uploaded file"""
        try:
            # Check for executable file signatures
            file.seek(0)
            header = file.read(512)
            file.seek(0)
            
            # Common executable signatures to block
            dangerous_signatures = [
                b'MZ',  # Windows PE
                b'\x7fELF',  # Linux ELF
                b'\xca\xfe\xba\xbe',  # Java class
                b'PK\x03\x04',  # ZIP (could contain executables)
            ]
            
            # Allow ZIP only for Office documents
            if header.startswith(b'PK\x03\x04'):
                # Check if it's an Office document
                if not (file.name.endswith(('.docx', '.xlsx', '.pptx'))):
                    return False
            
            for sig in dangerous_signatures[:-1]:  # Exclude ZIP check
                if header.startswith(sig):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security check error: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Convert to numpy array for OpenCV processing
            if VISION_AVAILABLE:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(cv_image, (3, 3), 0)
                
                # Convert to grayscale
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to get better text definition
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Convert back to PIL Image
                image = Image.fromarray(thresh)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return image
    
    def extract_text_from_image(self, file: UploadedFile) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        result = {
            'success': False,
            'text': '',
            'confidence': 0,
            'method': None,
            'qr_codes': [],
            'error': None
        }
        
        if not OCR_AVAILABLE:
            result['error'] = 'OCR libraries not available'
            return result
        
        try:
            # Load and preprocess image
            image = Image.open(file)
            processed_image = self.preprocess_image(image)
            
            # Try EasyOCR first (generally more accurate)
            if self.easyocr_reader:
                try:
                    img_array = np.array(processed_image)
                    ocr_results = self.easyocr_reader.readtext(img_array)
                    
                    extracted_text = []
                    total_confidence = 0
                    
                    for (bbox, text, confidence) in ocr_results:
                        if confidence > 0.5:  # Filter low-confidence results
                            extracted_text.append(text)
                            total_confidence += confidence
                    
                    if extracted_text:
                        result['text'] = ' '.join(extracted_text)
                        result['confidence'] = total_confidence / len(ocr_results) * 100
                        result['method'] = 'EasyOCR'
                        result['success'] = True
                        
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            # Fallback to Tesseract if EasyOCR failed
            if not result['success']:
                try:
                    text = pytesseract.image_to_string(processed_image)
                    if text.strip():
                        result['text'] = text.strip()
                        result['confidence'] = 75  # Default confidence for Tesseract
                        result['method'] = 'Tesseract'
                        result['success'] = True
                except Exception as e:
                    logger.warning(f"Tesseract failed: {e}")
            
            # Extract QR codes if available
            if QR_AVAILABLE:
                try:
                    qr_codes = pyzbar.decode(np.array(image))
                    for qr in qr_codes:
                        result['qr_codes'].append({
                            'data': qr.data.decode('utf-8'),
                            'type': qr.type
                        })
                except Exception as e:
                    logger.warning(f"QR code extraction failed: {e}")
            
            if not result['success']:
                result['error'] = 'No text could be extracted from image'
                
        except Exception as e:
            result['error'] = f'Image processing error: {str(e)}'
            logger.error(f"Image text extraction error: {e}")
        
        return result
    
    def extract_text_from_audio(self, file: UploadedFile) -> Dict[str, Any]:
        """Convert speech to text from audio file"""
        result = {
            'success': False,
            'text': '',
            'confidence': 0,
            'duration': 0,
            'method': None,
            'error': None
        }
        
        if not SPEECH_AVAILABLE:
            result['error'] = 'Speech processing libraries not available'
            return result
        
        temp_file = None
        try:
            # Save uploaded file temporarily
            temp_file = self.temp_dir / f"audio_{hashlib.md5(str(timezone.now()).encode()).hexdigest()}{Path(file.name).suffix}"
            with open(temp_file, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            
            # Load audio with pydub for format conversion
            audio = AudioSegment.from_file(str(temp_file))
            result['duration'] = len(audio) / 1000  # Duration in seconds
            
            # Convert to WAV for speech recognition
            wav_file = temp_file.with_suffix('.wav')
            audio.export(str(wav_file), format='wav')
            
            # Use speech recognition
            with sr.AudioFile(str(wav_file)) as source:
                audio_data = self.speech_recognizer.record(source)
                
                # Try Google Web Speech API (free tier)
                try:
                    text = self.speech_recognizer.recognize_google(audio_data)
                    result['text'] = text
                    result['confidence'] = 85  # Default confidence for Google
                    result['method'] = 'Google Speech API'
                    result['success'] = True
                except sr.UnknownValueError:
                    result['error'] = 'Could not understand audio'
                except sr.RequestError as e:
                    # Fallback to offline recognition
                    try:
                        text = self.speech_recognizer.recognize_sphinx(audio_data)
                        result['text'] = text
                        result['confidence'] = 70  # Lower confidence for offline
                        result['method'] = 'CMU Sphinx (offline)'
                        result['success'] = True
                    except:
                        result['error'] = f'Speech recognition failed: {str(e)}'
                        
        except Exception as e:
            result['error'] = f'Audio processing error: {str(e)}'
            logger.error(f"Audio text extraction error: {e}")
        finally:
            # Cleanup temporary files
            if temp_file and temp_file.exists():
                temp_file.unlink()
            wav_file = temp_file.with_suffix('.wav') if temp_file else None
            if wav_file and wav_file.exists():
                wav_file.unlink()
        
        return result
    
    def extract_text_from_document(self, file: UploadedFile) -> Dict[str, Any]:
        """Extract text from various document formats"""
        result = {
            'success': False,
            'text': '',
            'metadata': {},
            'page_count': 0,
            'method': None,
            'error': None
        }
        
        if not DOCUMENT_AVAILABLE:
            result['error'] = 'Document processing libraries not available'
            return result
        
        file_extension = Path(file.name).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                result = self._extract_from_pdf(file)
            elif file_extension in ['.docx', '.doc']:
                result = self._extract_from_word(file)
            elif file_extension == '.txt':
                result = self._extract_from_text(file)
            elif file_extension in ['.eml', '.msg']:
                result = self._extract_from_email(file)
            else:
                result['error'] = f'Unsupported document format: {file_extension}'
                
        except Exception as e:
            result['error'] = f'Document processing error: {str(e)}'
            logger.error(f"Document text extraction error: {e}")
        
        return result
    
    def _extract_from_pdf(self, file: UploadedFile) -> Dict[str, Any]:
        """Extract text from PDF file"""
        result = {'success': False, 'text': '', 'metadata': {}, 'page_count': 0, 'method': 'PyPDF2'}
        
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            result['page_count'] = len(pdf_reader.pages)
            
            # Extract metadata
            if pdf_reader.metadata:
                result['metadata'] = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', '')
                }
            
            # Extract text from all pages
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            
            result['text'] = '\n'.join(text_parts)
            result['success'] = bool(result['text'].strip())
            
        except Exception as e:
            result['error'] = f'PDF extraction error: {str(e)}'
        
        return result
    
    def _extract_from_word(self, file: UploadedFile) -> Dict[str, Any]:
        """Extract text from Word document"""
        result = {'success': False, 'text': '', 'metadata': {}, 'page_count': 1, 'method': 'python-docx'}
        
        try:
            doc = Document(file)
            
            # Extract metadata
            result['metadata'] = {
                'title': doc.core_properties.title or '',
                'author': doc.core_properties.author or '',
                'subject': doc.core_properties.subject or '',
                'created': str(doc.core_properties.created) if doc.core_properties.created else ''
            }
            
            # Extract text from paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)
            
            result['text'] = '\n'.join(text_parts)
            result['success'] = bool(result['text'].strip())
            
        except Exception as e:
            result['error'] = f'Word document extraction error: {str(e)}'
        
        return result
    
    def _extract_from_text(self, file: UploadedFile) -> Dict[str, Any]:
        """Extract text from plain text file"""
        result = {'success': False, 'text': '', 'metadata': {}, 'page_count': 1, 'method': 'plain text'}
        
        try:
            text = file.read().decode('utf-8')
            result['text'] = text
            result['success'] = True
            result['metadata'] = {'encoding': 'utf-8', 'size': len(text)}
            
        except UnicodeDecodeError:
            try:
                file.seek(0)
                text = file.read().decode('latin-1')
                result['text'] = text
                result['success'] = True
                result['metadata'] = {'encoding': 'latin-1', 'size': len(text)}
            except Exception as e:
                result['error'] = f'Text file encoding error: {str(e)}'
        except Exception as e:
            result['error'] = f'Text file extraction error: {str(e)}'
        
        return result
    
    def _extract_from_email(self, file: UploadedFile) -> Dict[str, Any]:
        """Extract text from email files (.eml, .msg)"""
        result = {'success': False, 'text': '', 'metadata': {}, 'page_count': 1, 'method': 'email parser'}
        
        try:
            import email
            from email import policy
            
            file_content = file.read()
            
            if file.name.endswith('.eml'):
                msg = email.message_from_bytes(file_content, policy=policy.default)
            else:
                # For .msg files, try to parse as best as possible
                msg = email.message_from_bytes(file_content, policy=policy.default)
            
            # Extract metadata
            result['metadata'] = {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'message_id': msg.get('Message-ID', '')
            }
            
            # Extract text content
            text_parts = []
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        text_parts.append(part.get_content())
                    elif part.get_content_type() == 'text/html':
                        # Basic HTML stripping
                        html_content = part.get_content()
                        # Remove HTML tags (basic)
                        import re
                        text_content = re.sub(r'<[^>]+>', '', html_content)
                        text_parts.append(text_content)
            else:
                text_parts.append(msg.get_content())
            
            result['text'] = '\n'.join(text_parts)
            result['success'] = bool(result['text'].strip())
            
        except Exception as e:
            result['error'] = f'Email extraction error: {str(e)}'
        
        return result
    
    def capture_url_screenshot(self, url: str) -> Dict[str, Any]:
        """Capture screenshot of URL for analysis"""
        result = {
            'success': False,
            'screenshot_path': None,
            'page_title': '',
            'error': None
        }
        
        if not WEB_AUTOMATION_AVAILABLE:
            result['error'] = 'Web automation libraries not available'
            return result
        
        driver = None
        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
            # Initialize driver
            driver = webdriver.Chrome(
                service=webdriver.ChromeService(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # Navigate to URL with timeout
            driver.set_page_load_timeout(10)
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page title
            result['page_title'] = driver.title
            
            # Take screenshot
            screenshot_filename = f"url_screenshot_{hashlib.md5(url.encode()).hexdigest()}.png"
            screenshot_path = self.temp_dir / screenshot_filename
            
            driver.save_screenshot(str(screenshot_path))
            result['screenshot_path'] = str(screenshot_path)
            result['success'] = True
            
        except Exception as e:
            result['error'] = f'Screenshot capture error: {str(e)}'
            logger.error(f"URL screenshot error: {e}")
        finally:
            if driver:
                driver.quit()
        
        return result
    
    def process_multimodal_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to process multi-modal input"""
        result = {
            'success': False,
            'extracted_text': '',
            'input_type': 'unknown',
            'processing_steps': [],
            'metadata': {},
            'error': None
        }
        
        try:
            if 'file' in input_data and input_data['file']:
                file = input_data['file']
                validation = self.validate_file(file)
                
                if not validation['valid']:
                    result['error'] = f"File validation failed: {'; '.join(validation['errors'])}"
                    return result
                
                file_type = validation['file_type']
                result['input_type'] = file_type
                result['processing_steps'].append(f"File validated: {file_type}")
                
                # Process based on file type
                if file_type in self.SUPPORTED_IMAGE_TYPES:
                    ocr_result = self.extract_text_from_image(file)
                    if ocr_result['success']:
                        result['extracted_text'] = ocr_result['text']
                        result['metadata']['ocr_confidence'] = ocr_result['confidence']
                        result['metadata']['ocr_method'] = ocr_result['method']
                        result['metadata']['qr_codes'] = ocr_result['qr_codes']
                        result['success'] = True
                        result['processing_steps'].append(f"OCR completed: {ocr_result['method']}")
                    else:
                        result['error'] = ocr_result['error']
                
                elif file_type in self.SUPPORTED_AUDIO_TYPES:
                    speech_result = self.extract_text_from_audio(file)
                    if speech_result['success']:
                        result['extracted_text'] = speech_result['text']
                        result['metadata']['speech_confidence'] = speech_result['confidence']
                        result['metadata']['speech_method'] = speech_result['method']
                        result['metadata']['duration'] = speech_result['duration']
                        result['success'] = True
                        result['processing_steps'].append(f"Speech-to-text completed: {speech_result['method']}")
                    else:
                        result['error'] = speech_result['error']
                
                elif file_type in self.SUPPORTED_DOCUMENT_TYPES:
                    doc_result = self.extract_text_from_document(file)
                    if doc_result['success']:
                        result['extracted_text'] = doc_result['text']
                        result['metadata']['document_metadata'] = doc_result['metadata']
                        result['metadata']['page_count'] = doc_result['page_count']
                        result['metadata']['extraction_method'] = doc_result['method']
                        result['success'] = True
                        result['processing_steps'].append(f"Document processed: {doc_result['method']}")
                    else:
                        result['error'] = doc_result['error']
            
            elif 'url' in input_data and input_data['url']:
                # Enhanced URL processing with screenshot
                url = input_data['url']
                result['input_type'] = 'url'
                
                if input_data.get('capture_screenshot', False):
                    screenshot_result = self.capture_url_screenshot(url)
                    if screenshot_result['success']:
                        result['metadata']['screenshot_path'] = screenshot_result['screenshot_path']
                        result['metadata']['page_title'] = screenshot_result['page_title']
                        result['processing_steps'].append("Screenshot captured")
                        
                        # Optionally run OCR on screenshot
                        if input_data.get('ocr_screenshot', False):
                            with open(screenshot_result['screenshot_path'], 'rb') as img_file:
                                from django.core.files.uploadedfile import SimpleUploadedFile
                                screenshot_file = SimpleUploadedFile(
                                    "screenshot.png",
                                    img_file.read(),
                                    content_type="image/png"
                                )
                                ocr_result = self.extract_text_from_image(screenshot_file)
                                if ocr_result['success']:
                                    result['extracted_text'] = ocr_result['text']
                                    result['metadata']['screenshot_ocr'] = ocr_result
                                    result['processing_steps'].append("Screenshot OCR completed")
                
                result['extracted_text'] = result.get('extracted_text', url)
                result['success'] = True
            
            elif 'text' in input_data and input_data['text']:
                # Direct text input
                result['extracted_text'] = input_data['text']
                result['input_type'] = 'text'
                result['success'] = True
                result['processing_steps'].append("Direct text input")
            
            else:
                result['error'] = 'No valid input provided'
            
            # Add processing timestamp
            result['metadata']['processed_at'] = timezone.now().isoformat()
            result['metadata']['capabilities'] = self.get_capabilities()
            
        except Exception as e:
            result['error'] = f'Processing error: {str(e)}'
            logger.error(f"Multi-modal processing error: {e}")
        
        return result

# Global instance
multimodal_processor = MultiModalProcessor()

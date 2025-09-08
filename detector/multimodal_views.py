"""
Enhanced Views with Multi-Modal Input Support
Extends existing views to handle images, audio, documents, and enhanced URL 
analysis
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.core.files.uploadedfile import UploadedFile
import json
import logging

from .models import (
    PredictionHistory, EnhancedPredictionHistory,
    ThreatCategory, RiskLevel, BatchProcessingJob
)

# Import existing services
try:
    from .enhanced_deep_learning_service import enhanced_service
    SERVICE_AVAILABLE = enhanced_service is not None
except ImportError:
    SERVICE_AVAILABLE = False
    enhanced_service = None

# Import multi-modal processor
try:
    from .multimodal_service import multimodal_processor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    multimodal_processor = None

logger = logging.getLogger(__name__)


def home(request):
    """Multi-modal interface home page"""
    return render(request, 'detector/multimodal.html', {
        'multimodal_available': MULTIMODAL_AVAILABLE,
        'service_available': SERVICE_AVAILABLE,
    })


def get_confidence_level(confidence_score):
    """Convert confidence score to human-readable level."""
    if confidence_score >= 90:
        return 'Very High'
    elif confidence_score >= 75:
        return 'High'
    elif confidence_score >= 60:
        return 'Medium'
    elif confidence_score >= 40:
        return 'Low'
    else:
        return 'Very Low'


@csrf_exempt
@require_http_methods(["POST"])
def predict_multimodal(request):
    """
    Enhanced endpoint for multi-modal input prediction
    Supports: text, images, audio, documents, URLs with screenshots
    """
    try:
        # Check if multi-modal processing is available
        if not MULTIMODAL_AVAILABLE:
            return JsonResponse({
                'success': False,
                'error': 'Multi-modal processing not available. Please install required dependencies.'
            })

        # Parse input data
        input_data = {}
        
        # Handle file uploads
        if request.FILES:
            file = list(request.FILES.values())[0]  # Get first uploaded file
            input_data['file'] = file
        
        # Handle form data or JSON data
        if request.POST:
            # Form data (multipart/form-data)
            input_data.update(dict(request.POST))
            
            # Convert string values
            for key, value in input_data.items():
                if isinstance(value, list) and len(value) == 1:
                    input_data[key] = value[0]
                    
        elif request.body:
            # JSON data
            try:
                json_data = json.loads(request.body)
                input_data.update(json_data)
            except json.JSONDecodeError:
                pass
        
        # Validate input
        if not any(key in input_data for key in ['file', 'text', 'url']):
            return JsonResponse({
                'success': False,
                'error': 'No valid input provided. Please upload a file, enter text, or provide a URL.'
            })
        
        # Process multi-modal input
        processing_result = multimodal_processor.process_multimodal_input(input_data)
        
        if not processing_result['success']:
            return JsonResponse({
                'success': False,
                'error': processing_result['error'],
                'processing_info': {
                    'input_type': processing_result['input_type'],
                    'capabilities': processing_result['metadata'].get('capabilities', {})
                }
            })
        
        # Extract text for analysis
        extracted_text = processing_result['extracted_text']
        input_type = processing_result['input_type']
        
        if not extracted_text or len(extracted_text.strip()) < 5:
            return JsonResponse({
                'success': False,
                'error': 'Insufficient text extracted for analysis. Please ensure your input contains readable text.',
                'processing_info': {
                    'input_type': input_type,
                    'extracted_text_length': len(extracted_text),
                    'processing_steps': processing_result['processing_steps']
                }
            })
        
        # Determine prediction type based on content and input type
        prediction_type = _determine_prediction_type(extracted_text, input_type, input_data)
        
        # Run enhanced ML analysis
        if not SERVICE_AVAILABLE:
            return JsonResponse({
                'success': False,
                'error': 'Enhanced Deep Learning service not available'
            })
        
        # Call appropriate prediction method based on type
        if prediction_type == 'email':
            ml_result = enhanced_service.predict_email(extracted_text)
        elif prediction_type == 'sms':
            ml_result = enhanced_service.predict_sms(extracted_text)
        elif prediction_type == 'url':
            # For URL analysis, use the original URL if available
            analysis_text = input_data.get('url', extracted_text)
            ml_result = enhanced_service.predict_url(analysis_text)
        else:
            # Default to email analysis for unknown content
            ml_result = enhanced_service.predict_email(extracted_text)
        
        if 'error' in ml_result:
            return JsonResponse({
                'success': False,
                'error': ml_result['error'],
                'processing_info': {
                    'input_type': input_type,
                    'prediction_type': prediction_type,
                    'extracted_text_preview': extracted_text[:200] + '...' if len(extracted_text) > 200 else extracted_text
                }
            })
        
        # Extract analysis results
        if 'dl_analysis' in ml_result:
            basic_result = ml_result['dl_analysis']
            enhanced_data = {
                'final_recommendation': ml_result.get('final_recommendation', 'SAFE'),
                'threat_detected': ml_result.get('threat_detected', False),
                'llm_analysis': ml_result.get('llm_analysis'),
                'analysis_method': ml_result.get('analysis_method', 'deep_learning_only')
            }
        else:
            basic_result = ml_result
            enhanced_data = {
                'final_recommendation': 'SAFE',
                'threat_detected': False,
                'llm_analysis': None,
                'analysis_method': 'fallback'
            }
        
        # Save prediction to database
        predicted_label = basic_result.get('result', 'UNKNOWN')
        confidence_score = basic_result.get('confidence_score', 85.0)
        
        # Create enhanced prediction record
        prediction_record = _create_prediction_record(
            prediction_type=prediction_type,
            input_text=extracted_text,
            original_input_type=input_type,
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            processing_metadata=processing_result['metadata'],
            ml_result=ml_result
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'result': {
                'prediction': predicted_label,
                'confidence_score': confidence_score,
                'confidence_level': get_confidence_level(confidence_score),
                'prediction_type': prediction_type
            },
            'input_processing': {
                'input_type': input_type,
                'extracted_text_length': len(extracted_text),
                'processing_steps': processing_result['processing_steps'],
                'metadata': processing_result['metadata']
            },
            'enhanced_analysis': {
                'final_recommendation': enhanced_data['final_recommendation'],
                'threat_detected': enhanced_data['threat_detected'],
                'analysis_method': enhanced_data['analysis_method'],
                'record_id': str(prediction_record.id)
            }
        }
        
        # Add LLM analysis if available
        if enhanced_data.get('llm_analysis'):
            llm = enhanced_data['llm_analysis']
            response_data['enhanced_analysis']['llm_summary'] = {
                'threat_level': llm.get('threat_level', 'UNKNOWN'),
                'confidence': llm.get('confidence', 0),
                'analysis_summary': llm.get('analysis_summary', '')
            }
        
        # Add extracted text preview for user verification
        response_data['extracted_text_preview'] = (
            extracted_text[:500] + '...' if len(extracted_text) > 500 
            else extracted_text
        )
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Multi-modal prediction error: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def upload_and_analyze(request):
    """
    Dedicated endpoint for file upload and analysis
    Supports drag-and-drop and traditional file upload
    """
    try:
        if not request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file uploaded'
            })
        
        file = list(request.FILES.values())[0]
        
        # Get additional options from request
        options = {
            'capture_screenshot': request.POST.get('capture_screenshot', 'false').lower() == 'true',
            'ocr_screenshot': request.POST.get('ocr_screenshot', 'false').lower() == 'true',
            'prediction_type': request.POST.get('prediction_type', 'auto')
        }
        
        # Prepare input data
        input_data = {'file': file}
        input_data.update(options)
        
        # Use the multi-modal prediction endpoint
        request.POST = request.POST.copy()
        for key, value in options.items():
            request.POST[key] = str(value)
        
        return predict_multimodal(request)
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return JsonResponse({
            'success': False,
            'error': f'File upload error: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def batch_analyze(request):
    """
    Endpoint for batch analysis of multiple files or URLs
    """
    try:
        if not MULTIMODAL_AVAILABLE:
            return JsonResponse({
                'success': False,
                'error': 'Multi-modal processing not available'
            })
        
        # Parse batch input
        batch_data = json.loads(request.body) if request.body else {}
        files = list(request.FILES.values()) if request.FILES else []
        urls = batch_data.get('urls', [])
        
        if not files and not urls:
            return JsonResponse({
                'success': False,
                'error': 'No files or URLs provided for batch analysis'
            })
        
        results = []
        
        # Process files
        for i, file in enumerate(files):
            try:
                input_data = {'file': file}
                processing_result = multimodal_processor.process_multimodal_input(input_data)
                
                if processing_result['success']:
                    # Quick analysis using first few lines
                    preview_text = processing_result['extracted_text'][:1000]
                    prediction_type = _determine_prediction_type(
                        preview_text, processing_result['input_type'], input_data
                    )
                    
                    results.append({
                        'index': i,
                        'filename': file.name,
                        'input_type': processing_result['input_type'],
                        'prediction_type': prediction_type,
                        'text_length': len(processing_result['extracted_text']),
                        'success': True,
                        'preview': preview_text
                    })
                else:
                    results.append({
                        'index': i,
                        'filename': file.name,
                        'success': False,
                        'error': processing_result['error']
                    })
                    
            except Exception as e:
                results.append({
                    'index': i,
                    'filename': file.name if hasattr(file, 'name') else f'file_{i}',
                    'success': False,
                    'error': str(e)
                })
        
        # Process URLs
        for i, url in enumerate(urls):
            try:
                input_data = {
                    'url': url,
                    'capture_screenshot': batch_data.get('capture_screenshots', False)
                }
                processing_result = multimodal_processor.process_multimodal_input(input_data)
                
                results.append({
                    'index': len(files) + i,
                    'url': url,
                    'input_type': 'url',
                    'prediction_type': 'url',
                    'success': processing_result['success'],
                    'error': processing_result.get('error'),
                    'metadata': processing_result.get('metadata', {})
                })
                
            except Exception as e:
                results.append({
                    'index': len(files) + i,
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
        
        return JsonResponse({
            'success': True,
            'batch_results': results,
            'total_items': len(results),
            'successful_items': sum(1 for r in results if r['success']),
            'failed_items': sum(1 for r in results if not r['success'])
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Batch analysis error: {str(e)}'
        })


def capabilities(request):
    """Return available multi-modal processing capabilities"""
    if MULTIMODAL_AVAILABLE:
        capabilities = multimodal_processor.get_capabilities()
    else:
        capabilities = {
            'image_processing': False,
            'ocr_processing': False,
            'speech_processing': False,
            'document_processing': False,
            'qr_processing': False,
            'web_automation': False,
            'file_validation': False
        }
    
    return JsonResponse({
        'multimodal_available': MULTIMODAL_AVAILABLE,
        'enhanced_service_available': SERVICE_AVAILABLE,
        'capabilities': capabilities,
        'supported_formats': {
            'images': list(multimodal_processor.SUPPORTED_IMAGE_TYPES.keys()) if MULTIMODAL_AVAILABLE else [],
            'audio': list(multimodal_processor.SUPPORTED_AUDIO_TYPES.keys()) if MULTIMODAL_AVAILABLE else [],
            'documents': list(multimodal_processor.SUPPORTED_DOCUMENT_TYPES.keys()) if MULTIMODAL_AVAILABLE else []
        } if MULTIMODAL_AVAILABLE else {}
    })


def _determine_prediction_type(text: str, input_type: str, input_data: dict) -> str:
    """
    Determine the most appropriate prediction type based on content and context
    """
    # Explicit type from user
    if input_data.get('prediction_type') and input_data['prediction_type'] != 'auto':
        return input_data['prediction_type']
    
    # URL input
    if input_type == 'url' or 'url' in input_data:
        return 'url'
    
    # Content-based detection
    text_lower = text.lower()
    
    # SMS indicators
    sms_indicators = [
        'text me', 'txt me', 'reply stop', 'msg&data rates',
        'free msg', 'standard rates apply', 'opt out'
    ]
    
    # Email indicators  
    email_indicators = [
        'dear', 'sincerely', 'best regards', 'subject:', 'from:',
        'to:', 'unsubscribe', 'click here', 'verify your account'
    ]
    
    # URL indicators in text
    url_indicators = ['http://', 'https://', 'www.', '.com', '.org', '.net']
    
    # Count indicators
    sms_score = sum(1 for indicator in sms_indicators if indicator in text_lower)
    email_score = sum(1 for indicator in email_indicators if indicator in text_lower)
    url_score = sum(1 for indicator in url_indicators if indicator in text_lower)
    
    # Determine type based on highest score
    if url_score > max(sms_score, email_score):
        return 'url'
    elif sms_score > email_score:
        return 'sms'
    else:
        return 'email'  # Default to email


def _create_prediction_record(prediction_type: str, input_text: str, 
                             original_input_type: str, predicted_label: str,
                             confidence_score: float, processing_metadata: dict,
                             ml_result: dict) -> EnhancedPredictionHistory:
    """Create enhanced prediction record with multi-modal metadata"""
    
    # Determine threat category and risk level
    threat_category = None
    risk_level = None
    
    if predicted_label in ['PHISHING', 'MALICIOUS']:
        threat_category = ThreatCategory.objects.filter(
            name='Financial Fraud'
        ).first()
        risk_level = RiskLevel.objects.filter(name='High').first()
    elif predicted_label == 'SPAM':
        threat_category = ThreatCategory.objects.filter(
            name='Suspicious Activity'
        ).first()
        risk_level = RiskLevel.objects.filter(name='Medium').first()
    else:
        risk_level = RiskLevel.objects.filter(name='Low').first()
    
    # Extract domain for URL predictions
    domain = ''
    if prediction_type == 'url':
        try:
            from urllib.parse import urlparse
            import tldextract
            
            url = processing_metadata.get('original_url', input_text)
            parsed_url = urlparse(url)
            domain = parsed_url.netloc or tldextract.extract(url).registered_domain
        except:
            domain = ''
    
    # Create record
    record = EnhancedPredictionHistory.objects.create(
        prediction_type=prediction_type,
        input_text=input_text[:200] + '...' if len(input_text) > 200 else input_text,
        predicted_label=predicted_label,
        confidence_score=confidence_score,
        threat_category=threat_category,
        risk_level=risk_level,
        domain=domain,
        risk_score=confidence_score,
        # Add multi-modal specific fields if model supports them
        # original_input_type=original_input_type,
        # processing_metadata=processing_metadata
    )
    
    return record

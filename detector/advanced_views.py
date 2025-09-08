"""
Advanced Django Views with Deep Learning Integration
Supports the new multi-modal deep learning spam detection system
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.conf import settings
import json
import asyncio
import logging
from typing import Dict, Any

# Import both old and new ML services for backward compatibility
try:
    from .advanced_ml_service import advanced_ml_service
    ADVANCED_ML_AVAILABLE = advanced_ml_service is not None
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    advanced_ml_service = None

from .ml_service import ml_service as traditional_ml_service
from .models import (
    PredictionHistory, EnhancedPredictionHistory, ThreatCategory, 
    RiskLevel, BatchProcessingJob, AuditLog
)

logger = logging.getLogger(__name__)


def home(request):
    """Enhanced main page with deep learning analytics"""
    try:
        # Get recent prediction history
        enhanced_email_history = EnhancedPredictionHistory.objects.filter(
            prediction_type='email'
        ).order_by('-timestamp')[:10]
        
        enhanced_sms_history = EnhancedPredictionHistory.objects.filter(
            prediction_type='sms'
        ).order_by('-timestamp')[:10]
        
        enhanced_url_history = EnhancedPredictionHistory.objects.filter(
            prediction_type='url'
        ).order_by('-timestamp')[:10]
        
        # Get threat analytics
        threat_categories = ThreatCategory.objects.all()
        risk_levels = RiskLevel.objects.all().order_by('level')
        
        # Enhanced statistics
        total_predictions = EnhancedPredictionHistory.objects.count()
        malicious_predictions = EnhancedPredictionHistory.objects.filter(
            predicted_label__in=['PHISHING', 'SPAM', 'MALICIOUS']
        ).count()
        
        # Risk level distribution
        risk_distribution = {}
        for risk_level in risk_levels:
            count = EnhancedPredictionHistory.objects.filter(
                risk_level=risk_level
            ).count()
            risk_distribution[risk_level.name] = count
        
        # Recent batch jobs
        recent_batch_jobs = BatchProcessingJob.objects.order_by('-created_at')[:5]
        
        # Performance metrics (last 24 hours)
        from datetime import timedelta
        yesterday = timezone.now() - timedelta(days=1)
        recent_predictions = EnhancedPredictionHistory.objects.filter(
            timestamp__gte=yesterday
        )
        
        avg_confidence = 0
        if recent_predictions.exists():
            avg_confidence = sum(
                p.confidence_score for p in recent_predictions
            ) / recent_predictions.count()
        
        context = {
            'email_history': enhanced_email_history,
            'sms_history': enhanced_sms_history,
            'url_history': enhanced_url_history,
            'threat_categories': threat_categories,
            'risk_levels': risk_levels,
            'total_predictions': total_predictions,
            'malicious_predictions': malicious_predictions,
            'safe_predictions': total_predictions - malicious_predictions,
            'risk_distribution': risk_distribution,
            'recent_batch_jobs': recent_batch_jobs,
            'avg_confidence': round(avg_confidence, 1),
            'advanced_ml_available': ADVANCED_ML_AVAILABLE,
            'ml_model_status': 'Advanced Deep Learning' if ADVANCED_ML_AVAILABLE else 'Traditional ML'
        }
        
        return render(request, 'detector/enhanced_home.html', context)
        
    except Exception as e:
        logger.error(f"Home view error: {e}")
        # Fallback to traditional view
        return render(request, 'detector/home.html', {'error': str(e)})


@csrf_exempt
@require_http_methods(["POST"])
def predict_url_advanced(request):
    """Advanced URL prediction with deep learning ensemble"""
    try:
        data = json.loads(request.body)
        url = data.get('url', '').strip()
        
        if not url:
            return JsonResponse({'error': 'URL is required'}, status=400)
        
        # Log audit trail
        AuditLog.objects.create(
            action='prediction',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'input_type': 'url', 'input_length': len(url)},
            success=True
        )
        
        # Use advanced ML service if available
        if ADVANCED_ML_AVAILABLE:
            result = advanced_ml_service.predict_url(url)
            
            # Enhanced result processing
            enhanced_result = process_advanced_result(
                result, url, 'url', request
            )
            
            # Generate comprehensive report
            if 'detailed_analysis' in result:
                enhanced_result['security_report'] = advanced_ml_service.generate_security_report(
                    url, result
                )
            
            return JsonResponse(enhanced_result)
        
        else:
            # Fallback to traditional ML
            result = traditional_ml_service.predict_url(url)
            
            # Save to history
            save_prediction_history(url, result, 'url', request)
            
            return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"URL prediction error: {e}")
        
        # Log failed attempt
        AuditLog.objects.create(
            action='prediction',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'input_type': 'url', 'error': str(e)},
            success=False,
            error_message=str(e)
        )
        
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_sms_advanced(request):
    """Advanced SMS prediction with deep learning ensemble"""
    try:
        data = json.loads(request.body)
        sms_text = data.get('sms_text', '').strip()
        
        if not sms_text:
            return JsonResponse({'error': 'SMS text is required'}, status=400)
        
        # Log audit trail
        AuditLog.objects.create(
            action='prediction',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'input_type': 'sms', 'input_length': len(sms_text)},
            success=True
        )
        
        # Use advanced ML service if available
        if ADVANCED_ML_AVAILABLE:
            result = advanced_ml_service.predict_sms(sms_text)
            
            # Enhanced result processing
            enhanced_result = process_advanced_result(
                result, sms_text, 'sms', request
            )
            
            # Generate comprehensive report
            if 'detailed_analysis' in result:
                enhanced_result['security_report'] = advanced_ml_service.generate_security_report(
                    sms_text, result
                )
            
            return JsonResponse(enhanced_result)
        
        else:
            # Fallback to traditional ML
            result = traditional_ml_service.predict_sms(sms_text)
            
            # Save to history
            save_prediction_history(sms_text, result, 'sms', request)
            
            return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"SMS prediction error: {e}")
        
        # Log failed attempt
        AuditLog.objects.create(
            action='prediction',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'input_type': 'sms', 'error': str(e)},
            success=False,
            error_message=str(e)
        )
        
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_email_advanced(request):
    """Advanced email prediction with deep learning ensemble"""
    try:
        data = json.loads(request.body)
        email_text = data.get('email_text', '').strip()
        
        if not email_text:
            return JsonResponse({'error': 'Email text is required'}, status=400)
        
        # Log audit trail
        AuditLog.objects.create(
            action='prediction',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'input_type': 'email', 'input_length': len(email_text)},
            success=True
        )
        
        # Use advanced ML service if available
        if ADVANCED_ML_AVAILABLE:
            result = advanced_ml_service.predict_email(email_text)
            
            # Enhanced result processing
            enhanced_result = process_advanced_result(
                result, email_text, 'email', request
            )
            
            # Generate comprehensive report
            if 'detailed_analysis' in result:
                enhanced_result['security_report'] = advanced_ml_service.generate_security_report(
                    email_text, result
                )
            
            return JsonResponse(enhanced_result)
        
        else:
            # Fallback to traditional ML
            result = traditional_ml_service.predict_email(email_text)
            
            # Save to history
            save_prediction_history(email_text, result, 'email', request)
            
            return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Email prediction error: {e}")
        
        # Log failed attempt
        AuditLog.objects.create(
            action='prediction',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'input_type': 'email', 'error': str(e)},
            success=False,
            error_message=str(e)
        )
        
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def batch_predict_advanced(request):
    """Advanced batch prediction endpoint"""
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        content_types = data.get('content_types', [])
        
        if not texts or len(texts) != len(content_types):
            return JsonResponse({
                'error': 'Invalid input: texts and content_types must have same length'
            }, status=400)
        
        # Limit batch size
        max_batch_size = getattr(settings, 'MAX_BATCH_PREDICTION_SIZE', 100)
        if len(texts) > max_batch_size:
            return JsonResponse({
                'error': f'Batch size too large. Maximum: {max_batch_size}'
            }, status=400)
        
        # Log batch processing start
        AuditLog.objects.create(
            action='batch_upload',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'batch_size': len(texts), 'content_types': content_types},
            success=True
        )
        
        # Use advanced ML service if available
        if ADVANCED_ML_AVAILABLE:
            # Run async batch prediction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            batch_results = loop.run_until_complete(
                advanced_ml_service.batch_predict_async(texts, content_types)
            )
            loop.close()
            
            # Process and save results
            processed_results = []
            for i, result in enumerate(batch_results):
                enhanced_result = process_advanced_result(
                    result, texts[i], content_types[i], request, batch_index=i
                )
                processed_results.append(enhanced_result)
            
            return JsonResponse({
                'results': processed_results,
                'batch_size': len(texts),
                'processing_time': 'N/A',  # Could add timing
                'advanced_processing': True
            })
        
        else:
            # Fallback to traditional processing
            batch_results = []
            for i, (text, content_type) in enumerate(zip(texts, content_types)):
                if content_type == 'url':
                    result = traditional_ml_service.predict_url(text)
                elif content_type == 'sms':
                    result = traditional_ml_service.predict_sms(text)
                elif content_type == 'email':
                    result = traditional_ml_service.predict_email(text)
                else:
                    result = {'error': f'Unsupported content type: {content_type}'}
                
                batch_results.append(result)
            
            return JsonResponse({
                'results': batch_results,
                'batch_size': len(texts),
                'advanced_processing': False
            })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def dashboard_advanced(request):
    """Enhanced dashboard with deep learning analytics"""
    try:
        # Get enhanced analytics
        total_predictions = EnhancedPredictionHistory.objects.count()
        
        # Recent predictions by type
        email_count = EnhancedPredictionHistory.objects.filter(prediction_type='email').count()
        sms_count = EnhancedPredictionHistory.objects.filter(prediction_type='sms').count()
        url_count = EnhancedPredictionHistory.objects.filter(prediction_type='url').count()
        
        # Threat distribution
        threat_distribution = {}
        for category in ThreatCategory.objects.all():
            count = EnhancedPredictionHistory.objects.filter(threat_category=category).count()
            threat_distribution[category.name] = count
        
        # Risk level distribution
        risk_distribution = {}
        for risk_level in RiskLevel.objects.all():
            count = EnhancedPredictionHistory.objects.filter(risk_level=risk_level).count()
            risk_distribution[risk_level.name] = count
        
        # Performance metrics
        from datetime import timedelta
        yesterday = timezone.now() - timedelta(days=1)
        recent_predictions = EnhancedPredictionHistory.objects.filter(timestamp__gte=yesterday)
        
        avg_confidence = 0
        avg_processing_time = 0
        if recent_predictions.exists():
            avg_confidence = sum(p.confidence_score for p in recent_predictions) / recent_predictions.count()
            avg_processing_time = sum(p.processing_time for p in recent_predictions) / recent_predictions.count()
        
        # Geographical distribution
        country_distribution = {}
        countries = EnhancedPredictionHistory.objects.values_list('country', flat=True).distinct()
        for country in countries:
            if country:
                count = EnhancedPredictionHistory.objects.filter(country=country).count()
                country_distribution[country] = count
        
        context = {
            'total_predictions': total_predictions,
            'email_count': email_count,
            'sms_count': sms_count,
            'url_count': url_count,
            'threat_distribution': threat_distribution,
            'risk_distribution': risk_distribution,
            'avg_confidence': round(avg_confidence, 1),
            'avg_processing_time': round(avg_processing_time, 3),
            'country_distribution': dict(list(country_distribution.items())[:10]),  # Top 10
            'advanced_ml_available': ADVANCED_ML_AVAILABLE,
            'recent_predictions': recent_predictions[:20]
        }
        
        return render(request, 'detector/enhanced_dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render(request, 'detector/dashboard.html', {'error': str(e)})


# Backward compatibility endpoints
@csrf_exempt
@require_http_methods(["POST"])
def predict_url(request):
    """Backward compatible URL prediction"""
    return predict_url_advanced(request)


@csrf_exempt
@require_http_methods(["POST"])
def predict_sms(request):
    """Backward compatible SMS prediction"""
    return predict_sms_advanced(request)


@csrf_exempt
@require_http_methods(["POST"])
def predict_email(request):
    """Backward compatible email prediction"""
    return predict_email_advanced(request)


def dashboard(request):
    """Backward compatible dashboard"""
    return dashboard_advanced(request)


# Helper functions
def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def process_advanced_result(result: Dict, input_text: str, prediction_type: str, 
                          request, batch_index: int = None) -> Dict:
    """Process and save advanced ML results"""
    try:
        # Extract prediction details
        if prediction_type == 'url':
            is_malicious = result.get('is_malicious', False)
            predicted_label = 'MALICIOUS' if is_malicious else 'SAFE'
        elif prediction_type == 'sms':
            is_spam = result.get('is_spam', False)
            predicted_label = 'SPAM' if is_spam else 'NOT SPAM'
        elif prediction_type == 'email':
            is_phishing = result.get('is_phishing', False)
            predicted_label = 'PHISHING' if is_phishing else 'NOT PHISHING'
        else:
            predicted_label = result.get('result', 'UNKNOWN')
        
        confidence_score = result.get('confidence_score', 0.0)
        
        # Determine risk level based on confidence and analysis
        detailed_analysis = result.get('detailed_analysis', {})
        risk_level_name = detailed_analysis.get('risk_level', 'Low')
        
        try:
            risk_level = RiskLevel.objects.get(name=risk_level_name)
        except RiskLevel.DoesNotExist:
            risk_level = RiskLevel.objects.filter(level=1).first()  # Default to Low
        
        # Save to enhanced prediction history
        prediction_record = EnhancedPredictionHistory.objects.create(
            prediction_type=prediction_type,
            input_text=input_text[:1000],  # Truncate if too long
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            risk_level=risk_level,
            risk_score=confidence_score,  # Use confidence as risk score for now
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            source_ip=get_client_ip(request),
            processing_time=0.0,  # Could measure actual processing time
            batch_position=batch_index or 0
        )
        
        # Add database record ID to result
        result['prediction_id'] = str(prediction_record.id)
        result['timestamp'] = prediction_record.timestamp.isoformat()
        result['risk_level'] = risk_level_name
        
        # Enhance result with additional metadata
        result['processing_metadata'] = {
            'advanced_ml_used': True,
            'model_type': 'Deep Learning Ensemble',
            'processing_time': '< 1s',  # Placeholder
            'confidence_band': get_confidence_band(confidence_score)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing advanced result: {e}")
        return result


def save_prediction_history(input_text: str, result: Dict, 
                          prediction_type: str, request):
    """Save prediction to traditional history (fallback)"""
    try:
        confidence_score = result.get('confidence_score', 0.0)
        predicted_label = result.get('result', 'UNKNOWN')
        
        PredictionHistory.objects.create(
            prediction_type=prediction_type,
            input_text=input_text[:1000],  # Truncate if too long
            predicted_label=predicted_label,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Error saving prediction history: {e}")


def get_confidence_band(confidence: float) -> str:
    """Get confidence band description"""
    if confidence >= 90:
        return "Very High"
    elif confidence >= 80:
        return "High"
    elif confidence >= 70:
        return "Medium"
    elif confidence >= 60:
        return "Low"
    else:
        return "Very Low"


@csrf_exempt
@require_http_methods(["POST"])
def clear_history(request):
    """Clear prediction history"""
    try:
        # Clear both old and new history
        PredictionHistory.objects.all().delete()
        EnhancedPredictionHistory.objects.all().delete()
        
        # Log the action
        AuditLog.objects.create(
            action='history_clear',
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            details={'action': 'clear_all_history'},
            success=True
        )
        
        return JsonResponse({'message': 'History cleared successfully'})
        
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return JsonResponse({'error': str(e)}, status=500)

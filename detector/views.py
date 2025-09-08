from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import json
from .models import (
    PredictionHistory, EnhancedPredictionHistory,
    ThreatCategory, RiskLevel, BatchProcessingJob
)


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


# Import Enhanced Deep Learning + LLM service
try:
    from .enhanced_deep_learning_service import enhanced_service
    SERVICE_AVAILABLE = enhanced_service is not None
    if SERVICE_AVAILABLE:
        print("🧠 Enhanced DL + LLM Service loaded successfully!")
        status = enhanced_service.get_service_status()
        print(f"📊 Service Status: {status}")
    else:
        print("❌ Enhanced DL + LLM Service failed to load")
except ImportError as e:
    SERVICE_AVAILABLE = False
    enhanced_service = None
    print(f"❌ Failed to import Enhanced DL + LLM Service: {e}")

def home(request):
    """Main page with the three-tab interface."""
    # Get recent prediction history for each type (using both old and new models)
    email_history = PredictionHistory.objects.filter(prediction_type='email')[:5]
    sms_history = PredictionHistory.objects.filter(prediction_type='sms')[:5]
    url_history = PredictionHistory.objects.filter(prediction_type='url')[:5]
    
    # Get enhanced prediction history
    enhanced_email_history = EnhancedPredictionHistory.objects.filter(prediction_type='email')[:5]
    enhanced_sms_history = EnhancedPredictionHistory.objects.filter(prediction_type='sms')[:5]
    enhanced_url_history = EnhancedPredictionHistory.objects.filter(prediction_type='url')[:5]
    
    # Get threat categories and risk levels for display
    threat_categories = ThreatCategory.objects.all()
    risk_levels = RiskLevel.objects.all().order_by('level')
    
    # Get some basic statistics
    total_predictions = EnhancedPredictionHistory.objects.count()
    malicious_predictions = EnhancedPredictionHistory.objects.filter(
        predicted_label__in=['PHISHING', 'SPAM', 'MALICIOUS']
    ).count()
    
    context = {
        'email_history': enhanced_email_history,  # Use enhanced history for display
        'sms_history': enhanced_sms_history,      # Use enhanced history for display
        'url_history': enhanced_url_history,      # Use enhanced history for display
        'enhanced_email_history': enhanced_email_history,
        'enhanced_sms_history': enhanced_sms_history,
        'enhanced_url_history': enhanced_url_history,
        'threat_categories': threat_categories,
        'risk_levels': risk_levels,
        'total_predictions': total_predictions,
        'malicious_predictions': malicious_predictions,
        'safe_predictions': total_predictions - malicious_predictions,
    }
    return render(request, 'detector/home.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def predict_url(request):
    """AJAX endpoint for URL prediction."""
    try:
        data = json.loads(request.body)
        url = data.get('url', '').strip()
        
        if not url:
            return JsonResponse({
                'success': False,
                'error': 'URL is required'
            })
        
        if SERVICE_AVAILABLE and enhanced_service:
            result = enhanced_service.predict_url(url)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Enhanced Deep Learning service not available'
            })
        
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
        
        # Extract basic result from enhanced response
        if 'dl_analysis' in result:
            # Enhanced response format
            basic_result = result['dl_analysis']
            enhanced_data = {
                'final_recommendation': result.get('final_recommendation', 'SAFE'),
                'threat_detected': result.get('threat_detected', False),
                'llm_analysis': result.get('llm_analysis'),
                'analysis_method': result.get('analysis_method', 'deep_learning_only')
            }
        else:
            # Fallback for basic response format
            basic_result = result
            enhanced_data = {
                'final_recommendation': 'SAFE',
                'threat_detected': False,
                'llm_analysis': None,
                'analysis_method': 'fallback'
            }
        
        # Save to both old and new models for compatibility
        predicted_label = basic_result.get('result', 'UNKNOWN')
        confidence_score = basic_result.get('confidence_score', 85.0)
        
        PredictionHistory.objects.create(
            prediction_type='url',
            input_text=(
                url[:200] + '...' if len(url) > 200 else url
            ),
            predicted_label=predicted_label,
            confidence_score=confidence_score
        )
        
        # Save to enhanced model with threat intelligence
        from urllib.parse import urlparse
        import tldextract
        
        # Extract domain for analysis
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc or tldextract.extract(url).registered_domain
        except:
            domain = ''
        
        # Determine threat category and risk level based on prediction
        threat_category = None
        risk_level = None
        
        if predicted_label == 'MALICIOUS':
            threat_category = ThreatCategory.objects.filter(
                name='Malware Distribution'
            ).first()
            risk_level = RiskLevel.objects.filter(name='High').first()
        else:
            risk_level = RiskLevel.objects.filter(name='Low').first()
        
        EnhancedPredictionHistory.objects.create(
            prediction_type='url',
            input_text=url[:200] + '...' if len(url) > 200 else url,
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            threat_category=threat_category,
            risk_level=risk_level,
            domain=domain,
            risk_score=confidence_score
        )
        
        # Create enhanced response
        response_data = {
            'success': True,
            'result': basic_result,
            'enhanced_analysis': {
                'final_recommendation': enhanced_data['final_recommendation'],
                'threat_detected': enhanced_data['threat_detected'],
                'analysis_method': enhanced_data['analysis_method'],
                'llm_available': enhanced_data['llm_analysis'] is not None
            }
        }
        
        # Add LLM analysis summary if available
        if enhanced_data['llm_analysis']:
            llm = enhanced_data['llm_analysis']
            response_data['enhanced_analysis']['llm_summary'] = {
                'threat_level': llm.get('threat_level', 'UNKNOWN'),
                'confidence': llm.get('confidence', 0),
                'analysis_summary': llm.get('analysis_summary', '')
            }
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

def dashboard(request):
    """Enhanced dashboard with analytics and threat intelligence."""
    # Get comprehensive statistics
    total_predictions = EnhancedPredictionHistory.objects.count()
    malicious_predictions = EnhancedPredictionHistory.objects.filter(
        predicted_label__in=['PHISHING', 'SPAM', 'MALICIOUS']
    ).count()
    
    # Get threat category distribution
    threat_categories = ThreatCategory.objects.all()
    category_stats = []
    for category in threat_categories:
        count = EnhancedPredictionHistory.objects.filter(threat_category=category).count()
        category_stats.append({
            'name': category.name,
            'count': count,
            'color': category.color,
            'icon': category.icon
        })
    
    # Get risk level distribution
    risk_levels = RiskLevel.objects.all().order_by('level')
    risk_stats = []
    for risk in risk_levels:
        count = EnhancedPredictionHistory.objects.filter(risk_level=risk).count()
        risk_stats.append({
            'name': risk.name,
            'count': count,
            'color': risk.color,
            'level': risk.level
        })
    
    # Get recent enhanced predictions
    recent_predictions = EnhancedPredictionHistory.objects.all().order_by('-timestamp')[:10]
    
    # Get batch processing jobs (if any exist)
    try:
        batch_jobs = BatchProcessingJob.objects.all().order_by('-created_at')[:5]
    except:
        batch_jobs = []
    
    context = {
        'total_predictions': total_predictions,
        'malicious_predictions': malicious_predictions,
        'safe_predictions': total_predictions - malicious_predictions,
        'category_stats': category_stats,
        'risk_stats': risk_stats,
        'recent_predictions': recent_predictions,
        'batch_jobs': batch_jobs,
        'threat_categories': threat_categories,
        'risk_levels': risk_levels,
    }
    return render(request, 'detector/dashboard.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def predict_sms(request):
    """AJAX endpoint for SMS prediction."""
    try:
        data = json.loads(request.body)
        sms_text = data.get('sms_text', '').strip()
        
        if not sms_text:
            return JsonResponse({
                'success': False,
                'error': 'SMS text is required'
            })
        
        if SERVICE_AVAILABLE and enhanced_service:
            result = enhanced_service.predict_sms(sms_text)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Enhanced Deep Learning service not available'
            })
        
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
        
        # Extract basic result from enhanced response
        if 'dl_analysis' in result:
            # Enhanced response format
            basic_result = result['dl_analysis']
            enhanced_data = {
                'final_recommendation': result.get('final_recommendation', 'SAFE'),
                'threat_detected': result.get('threat_detected', False),
                'llm_analysis': result.get('llm_analysis'),
                'analysis_method': result.get('analysis_method', 'deep_learning_only')
            }
        else:
            # Fallback for basic response format
            basic_result = result
            enhanced_data = {
                'final_recommendation': 'SAFE',
                'threat_detected': False,
                'llm_analysis': None,
                'analysis_method': 'fallback'
            }
        
        # Extract values for database storage
        predicted_label = basic_result.get('result', 'UNKNOWN')
        confidence_score = basic_result.get('confidence_score', 85.0)
        
        # Save to both old and new models for compatibility
        PredictionHistory.objects.create(
            prediction_type='sms',
            input_text=(
                sms_text[:200] + '...' if len(sms_text) > 200 else sms_text
            ),
            predicted_label=predicted_label,
            confidence_score=confidence_score
        )
        
        # Save to enhanced model with threat intelligence
        # Determine threat category and risk level based on prediction
        threat_category = None
        risk_level = None
        
        if predicted_label == 'SPAM':
            threat_category = ThreatCategory.objects.filter(
                name='Suspicious Activity'
            ).first()
            risk_level = RiskLevel.objects.filter(name='Medium').first()
        else:
            risk_level = RiskLevel.objects.filter(name='Low').first()
        
        EnhancedPredictionHistory.objects.create(
            prediction_type='sms',
            input_text=(
                sms_text[:200] + '...' if len(sms_text) > 200 else sms_text
            ),
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            threat_category=threat_category,
            risk_level=risk_level,
            risk_score=confidence_score
        )
        
        # Create enhanced response
        response_data = {
            'success': True,
            'result': basic_result,
            'enhanced_analysis': {
                'final_recommendation': enhanced_data['final_recommendation'],
                'threat_detected': enhanced_data['threat_detected'],
                'analysis_method': enhanced_data['analysis_method'],
                'llm_available': enhanced_data['llm_analysis'] is not None
            }
        }
        
        # Add LLM analysis summary if available
        if enhanced_data['llm_analysis']:
            llm = enhanced_data['llm_analysis']
            response_data['enhanced_analysis']['llm_summary'] = {
                'threat_level': llm.get('threat_level', 'UNKNOWN'),
                'confidence': llm.get('confidence', 0),
                'analysis_summary': llm.get('analysis_summary', '')
            }
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@require_http_methods(["POST"])
def clear_history(request):
    """Clear prediction history from both old and enhanced models."""
    try:
        data = json.loads(request.body)
        history_type = data.get('type', 'all')  # 'all', 'email', 'sms', 'url'
        
        deleted_count_old = 0
        deleted_count_enhanced = 0
        
        if history_type == 'all':
            # Delete from both models
            deleted_count_old = PredictionHistory.objects.all().delete()[0]
            deleted_count_enhanced = EnhancedPredictionHistory.objects.all().delete()[0]
        else:
            # Delete specific type from both models
            deleted_count_old = PredictionHistory.objects.filter(prediction_type=history_type).delete()[0]
            deleted_count_enhanced = EnhancedPredictionHistory.objects.filter(prediction_type=history_type).delete()[0]
        
        total_deleted = deleted_count_old + deleted_count_enhanced
        
        return JsonResponse({
            'success': True,
            'message': f'Successfully deleted {total_deleted} prediction records (Old: {deleted_count_old}, Enhanced: {deleted_count_enhanced})',
            'deleted_count': total_deleted
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@require_http_methods(["POST"])
def predict_email(request):
    """AJAX endpoint for email prediction."""
    try:
        data = json.loads(request.body)
        email_text = data.get('email_text', '').strip()
        
        if not email_text:
            return JsonResponse({
                'success': False,
                'error': 'Email text is required'
            })
        
        if SERVICE_AVAILABLE and enhanced_service:
            result = enhanced_service.predict_email(email_text)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Enhanced Deep Learning service not available'
            })
        
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
        
        # Extract basic result from enhanced response
        if 'dl_analysis' in result:
            # Enhanced response format
            basic_result = result['dl_analysis']
            enhanced_data = {
                'final_recommendation': result.get('final_recommendation', 'SAFE'),
                'threat_detected': result.get('threat_detected', False),
                'llm_analysis': result.get('llm_analysis'),
                'analysis_method': result.get('analysis_method', 'deep_learning_only')
            }
        else:
            # Fallback for basic response format
            basic_result = result
            enhanced_data = {
                'final_recommendation': 'SAFE',
                'threat_detected': False,
                'llm_analysis': None,
                'analysis_method': 'fallback'
            }
        
        # Extract values for database storage
        predicted_label = basic_result.get('result', 'UNKNOWN')
        confidence_score = basic_result.get('confidence_score', 85.0)
        
        # Save to both old and new models for compatibility
        PredictionHistory.objects.create(
            prediction_type='email',
            input_text=(
                email_text[:200] + '...' if len(email_text) > 200 else email_text
            ),
            predicted_label=predicted_label,
            confidence_score=confidence_score
        )
        
        # Save to enhanced model with threat intelligence
        # Determine threat category and risk level based on prediction
        threat_category = None
        risk_level = None
        
        if predicted_label == 'PHISHING':
            threat_category = ThreatCategory.objects.filter(
                name='Financial Fraud'
            ).first()
            risk_level = RiskLevel.objects.filter(name='High').first()
        else:
            risk_level = RiskLevel.objects.filter(name='Low').first()
        
        EnhancedPredictionHistory.objects.create(
            prediction_type='email',
            input_text=(
                email_text[:200] + '...'
                if len(email_text) > 200 else email_text
            ),
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            threat_category=threat_category,
            risk_level=risk_level,
            risk_score=confidence_score
        )
        
        # Prepare comprehensive response data
        response_data = {
            'success': True,
            'result': predicted_label,
            'confidence_score': confidence_score,
            'confidence_level': get_confidence_level(confidence_score),
            'analysis_timestamp': timezone.now().isoformat(),
            'model_version': '2.1.0',
            'enhanced_analysis': {
                'final_recommendation': enhanced_data['final_recommendation'],
                'threat_detected': enhanced_data['threat_detected'],
                'analysis_method': enhanced_data['analysis_method']
            }
        }
        
        # Add LLM analysis if available
        if enhanced_data.get('llm_analysis'):
            response_data['llm_analysis'] = enhanced_data['llm_analysis']
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@require_http_methods(["POST"])
def delete_prediction(request, prediction_id):
    """Delete a specific prediction from both models."""
    try:
        # Try to delete from enhanced model first
        try:
            enhanced_prediction = EnhancedPredictionHistory.objects.get(id=prediction_id)
            enhanced_prediction.delete()
        except EnhancedPredictionHistory.DoesNotExist:
            pass
        
        return JsonResponse({
            'success': True,
            'message': 'Prediction deleted successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@require_http_methods(["POST"])
def delete_selected_history(request):
    """Bulk delete selected predictions from EnhancedPredictionHistory by IDs."""
    try:
        data = json.loads(request.body)
        ids = data.get('ids', [])
        history_type = data.get('type', None)
        if not ids or not history_type:
            return JsonResponse({'success': False, 'error': 'Missing IDs or type.'})
        # Delete from EnhancedPredictionHistory
        deleted, _ = EnhancedPredictionHistory.objects.filter(id__in=ids, prediction_type=history_type).delete()
        return JsonResponse({'success': True, 'message': f'Successfully deleted {deleted} selected {history_type} prediction(s).', 'deleted_count': deleted})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

"""
Enhanced views for email predictions dashboard with cards, pagination, and filtering.
Includes action views to run fetch/analyze scripts from the web UI.
"""

from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Avg
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
try:
    from weasyprint import HTML
except ImportError:
    HTML = None
import json
import os
import subprocess
import threading
import uuid as uuid_module
import re
import sys
from datetime import timedelta
from pathlib import Path
from .models import EnhancedPredictionHistory, ThreatCategory, RiskLevel

# ─────────────────────────────────────────────────────────────────────────────
# Background process registry  {process_id: {proc, output_lines, status}}
# ─────────────────────────────────────────────────────────────────────────────
_PROCESS_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()

PROJECT_DIR = Path(__file__).parent.parent


def _stream_process(process_id, proc):
    """Thread target: stream stdout+stderr into registry."""
    with _REGISTRY_LOCK:
        _PROCESS_REGISTRY[process_id]['status'] = 'running'
    lines = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip('\n')
            lines.append(line)
            with _REGISTRY_LOCK:
                _PROCESS_REGISTRY[process_id]['output_lines'] = lines[:]
        proc.wait()
        status = 'done' if proc.returncode == 0 else f'error (exit {proc.returncode})'
    except Exception as e:
        status = f'error ({e})'
    with _REGISTRY_LOCK:
        _PROCESS_REGISTRY[process_id]['status'] = status


def _launch_script(script_name, extra_args=None):
    """Launch a Python script as a background process; return process_id."""
    process_id = str(uuid_module.uuid4())
    cmd = [sys.executable, str(PROJECT_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,   # prevent interactive prompts
            text=True,
            bufsize=1,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        with _REGISTRY_LOCK:
            _PROCESS_REGISTRY[process_id] = {
                'proc': proc,
                'output_lines': [],
                'status': 'starting'
            }
        t = threading.Thread(target=_stream_process, args=(process_id, proc), daemon=True)
        t.start()
    except Exception as e:
        with _REGISTRY_LOCK:
            _PROCESS_REGISTRY[process_id] = {
                'proc': None,
                'output_lines': [f'Failed to launch {script_name}: {e}'],
                'status': 'error'
            }
    return process_id


def _parse_email_from_input_text(input_text):
    """
    Try to parse structured email fields from input_text.
    Supports JSON format or plain Subject: / From: headers.
    """
    data = {'subject': '', 'from': '', 'date': '', 'body': input_text}
    if not input_text:
        return data
    # Try JSON
    try:
        parsed = json.loads(input_text)
        if isinstance(parsed, dict):
            data['subject'] = parsed.get('subject', parsed.get('Subject', ''))
            data['from'] = parsed.get('from', parsed.get('From', parsed.get('sender', '')))
            data['date'] = parsed.get('date', parsed.get('Date', ''))
            data['body'] = parsed.get('body', parsed.get('content', parsed.get('text', input_text)))
            return data
    except Exception:
        pass
    # Try header parsing
    lines = input_text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith('subject:'):
            data['subject'] = line[8:].strip()
        elif low.startswith('from:'):
            data['from'] = line[5:].strip()
        elif low.startswith('date:'):
            data['date'] = line[5:].strip()
        elif line.strip() == '' and i > 0:
            body_start = i + 1
            break
    if body_start:
        data['body'] = '\n'.join(lines[body_start:])
    else:
        data['body'] = input_text
    return data


def _compute_risk_score(prediction):
    """
    Compute a real risk score (0-100) based on ML prediction label and confidence.
    """
    label = prediction.predicted_label.upper()
    conf = prediction.confidence_score  # 0–100
    if label == 'PHISHING':
        # Risk = confidence directly (high confidence phishing = high risk)
        return round(max(conf, 50), 1)
    elif label in ('SPAM', 'MALICIOUS'):
        return round(max(conf * 0.85, 40), 1)
    elif label == 'SUSPICIOUS':
        return round(max(conf * 0.6, 30), 1)
    else:
        # NOT PHISHING / SAFE → risk is the inverse (low confidence safe = higher risk)
        return round(max(100 - conf, 5), 1)


def _extract_risk_indicators(prediction, parsed_email):
    """
    Produce a list of risk indicator dicts with {name, detected, description}.
    Based purely on real ML prediction data and text heuristics.
    """
    text = (parsed_email.get('body', '') + ' ' + parsed_email.get('subject', '')).lower()
    subject = parsed_email.get('subject', '').lower()
    label = prediction.predicted_label.upper()
    is_threat = label in ('PHISHING', 'SPAM', 'MALICIOUS', 'SUSPICIOUS')

    indicators = [
        {
            'name': 'Urgent Language',
            'detected': any(w in text for w in ['urgent', 'immediately', 'action required', 'verify now', 'account suspended', 'limited time']),
            'description': 'Email uses urgency to pressure the recipient into acting quickly.'
        },
        {
            'name': 'Suspicious Links',
            'detected': bool(re.search(r'https?://\S+', text) and is_threat),
            'description': 'Email contains links that may be malicious based on ML analysis.'
        },
        {
            'name': 'Credential Phishing',
            'detected': any(w in text for w in ['password', 'login', 'account', 'verify', 'confirm', 'credential', 'username']),
            'description': 'Email requests credentials or personal authentication info.'
        },
        {
            'name': 'Financial Lure',
            'detected': any(w in text for w in ['money', 'bank', '$', 'payment', 'transfer', 'winner', 'prize', 'million', 'bitcoin', 'crypto', 'refund']),
            'description': 'Email contains financial incentives or payment requests.'
        },
        {
            'name': 'Impersonation',
            'detected': any(d in text for d in ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'irs', 'netflix', 'fedex', 'dhl']),
            'description': 'Email may be impersonating a well-known organization.'
        },
        {
            'name': 'ML Threat Detected',
            'detected': is_threat,
            'description': f'Deep learning model classified this email as {label} with {prediction.confidence_score:.1f}% confidence.'
        },
    ]
    return indicators


def enhanced_predictions_dashboard(request):
    """
    Enhanced dashboard with card-based predictions, pagination, and filtering
    """
    # Get filter parameters
    prediction_type = request.GET.get('type', 'all')
    threat_category = request.GET.get('category', 'all')
    risk_level = request.GET.get('risk', 'all')
    search_query = request.GET.get('search', '')
    page_number = request.GET.get('page', 1)
    
    # Start with all enhanced predictions
    predictions = EnhancedPredictionHistory.objects.all().order_by('-timestamp')
    
    # Apply filters
    if prediction_type != 'all':
        predictions = predictions.filter(prediction_type=prediction_type)
    
    if threat_category != 'all':
        predictions = predictions.filter(threat_category__name=threat_category)
    
    if risk_level != 'all':
        predictions = predictions.filter(risk_level__name=risk_level)
    
    if search_query:
        predictions = predictions.filter(
            Q(input_text__icontains=search_query) |
            Q(predicted_label__icontains=search_query) |
            Q(domain__icontains=search_query)
        )
    
    # Get statistics
    total_predictions = EnhancedPredictionHistory.objects.count()
    malicious_predictions = EnhancedPredictionHistory.objects.filter(
        predicted_label__in=['PHISHING', 'SPAM', 'MALICIOUS']
    ).count()
    safe_predictions = total_predictions - malicious_predictions
    
    # Get unique values for filters
    prediction_types = EnhancedPredictionHistory.objects.values_list(
        'prediction_type', flat=True
    ).distinct()
    
    threat_categories = ThreatCategory.objects.all()
    risk_levels = RiskLevel.objects.all().order_by('level')
    
    # Paginate results (20 per page)
    paginator = Paginator(predictions, 20)
    page_obj = paginator.get_page(page_number)
    
    # Enhance predictions with report data
    enhanced_predictions = []
    for prediction in page_obj:
        prediction_data = {
            'id': prediction.id,
            'prediction_type': prediction.prediction_type,
            'input_text': prediction.input_text,
            'predicted_label': prediction.predicted_label,
            'confidence_score': prediction.confidence_score,
            'confidence_percentage': f"{prediction.confidence_score:.1f}%",
            'timestamp': prediction.timestamp,
            'threat_category': prediction.threat_category,
            'risk_level': prediction.risk_level,
            'domain': prediction.domain,
            'risk_score': prediction.risk_score,
            'has_report': False,
            'report_files': None,
            'email_content': None
        }
        
        # Try to find associated report files
        if prediction.prediction_type == 'email':
            # Look for report files in email_reports directory
            report_files = find_email_report_files(prediction.timestamp, prediction.input_text)
            if report_files:
                prediction_data['has_report'] = True
                prediction_data['report_files'] = report_files
                
                # Try to extract email content from report
                email_content = extract_email_content_from_report(report_files)
                if email_content:
                    prediction_data['email_content'] = email_content
        
        enhanced_predictions.append(prediction_data)
    
    context = {
        'predictions': enhanced_predictions,
        'page_obj': page_obj,
        'total_predictions': total_predictions,
        'malicious_predictions': malicious_predictions,
        'safe_predictions': safe_predictions,
        'prediction_types': prediction_types,
        'threat_categories': threat_categories,
        'risk_levels': risk_levels,
        'current_type': prediction_type,
        'current_category': threat_category,
        'current_risk': risk_level,
        'search_query': search_query,
    }
    
    return render(request, 'detector/enhanced_predictions_dashboard.html', context)


def prediction_detail(request, prediction_id):
    """
    Detailed view for a single prediction with email content and report data
    """
    prediction = get_object_or_404(EnhancedPredictionHistory, id=prediction_id)

    # Always parse from input_text first for headers
    parsed_email = _parse_email_from_input_text(prediction.input_text)

    # Get report files if available
    report_files = None
    html_report_content = None
    email_content = {'subject': '', 'from': '', 'date': '', 'body': '', 'content_preview': ''}
    deepseek_analysis = None

    if prediction.prediction_type == 'email':
        report_files = find_email_report_files(prediction.timestamp, prediction.input_text, str(prediction_id))
        if report_files:
            email_content = extract_email_content_from_report(report_files)
            deepseek_analysis = extract_deepseek_analysis_from_report(report_files)
            
            # Read HTML report content if available
            if report_files.get('html'):
                try:
                    with open(report_files['html'], 'r', encoding='utf-8') as f:
                        html_report_content = f.read()
                except Exception as e:
                    print(f"Error reading HTML report: {e}")

            # Enrich parsed_email from report if missing
            if isinstance(email_content, dict):
                for key in ('subject', 'from', 'date', 'body'):
                    if not parsed_email.get(key) and email_content.get(key):
                        parsed_email[key] = email_content[key]

    # Get related predictions
    related_predictions = EnhancedPredictionHistory.objects.filter(
        prediction_type=prediction.prediction_type
    ).exclude(id=prediction_id).order_by('-timestamp')[:5]

    context = {
        'prediction': prediction,
        'parsed_email': parsed_email,
        'report_files': report_files,
        'html_report_content': html_report_content,
        'email_content': email_content,
        'deepseek_analysis': deepseek_analysis,
        'related_predictions': related_predictions,
    }

    return render(request, 'detector/prediction_detail.html', context)


def manual_analysis_page(request, prediction_id):
    """
    Manual analysis page for detailed inspection with real parsed email data.
    """
    prediction = get_object_or_404(EnhancedPredictionHistory, id=prediction_id)

    parsed_email = _parse_email_from_input_text(prediction.input_text)

    # Also try to enrich from report files
    email_content = {'subject': '', 'from': '', 'date': '', 'body': '', 'content_preview': ''}
    if prediction.prediction_type == 'email':
        report_files = find_email_report_files(prediction.timestamp, prediction.input_text)
        if report_files:
            email_content = extract_email_content_from_report(report_files)
            # Enrich parsed_email from report dict if fields are missing
            if isinstance(email_content, dict):
                if not parsed_email['subject'] and email_content.get('subject'):
                    parsed_email['subject'] = email_content['subject']
                if not parsed_email['from'] and email_content.get('from'):
                    parsed_email['from'] = email_content['from']
                if not parsed_email['date'] and email_content.get('date'):
                    parsed_email['date'] = email_content['date']
                if not parsed_email['body'] and email_content.get('body'):
                    parsed_email['body'] = email_content['body']
                elif not parsed_email['body'] and email_content.get('content_preview'):
                    parsed_email['body'] = email_content['content_preview']

    real_risk_score = _compute_risk_score(prediction)
    risk_indicators = _extract_risk_indicators(prediction, parsed_email)

    # Confidence color
    label = prediction.predicted_label.upper()
    if label in ('PHISHING', 'MALICIOUS'):
        confidence_color = 'red'
    elif label in ('SPAM', 'SUSPICIOUS'):
        confidence_color = 'orange'
    else:
        confidence_color = 'green'

    context = {
        'prediction': prediction,
        'email_content': email_content,
        'parsed_email': parsed_email,
        'real_risk_score': real_risk_score,
        'risk_indicators': risk_indicators,
        'confidence_color': confidence_color,
        'analysis_timestamp': timezone.now(),
    }

    return render(request, 'detector/manual_analysis.html', context)


def deepseek_analysis_page(request, prediction_id):
    """
    DeepSeek AI analysis page
    """
    prediction = get_object_or_404(EnhancedPredictionHistory, id=prediction_id)

    # Parse email fields from stored input_text (always available)
    parsed_email = _parse_email_from_input_text(prediction.input_text)

    # Get DeepSeek analysis from report
    deepseek_analysis = None
    email_content = {'subject': '', 'from': '', 'date': '', 'body': '', 'content_preview': ''}
    if prediction.prediction_type == 'email':
        report_files = find_email_report_files(prediction.timestamp, prediction.input_text)
        if report_files:
            deepseek_analysis = extract_deepseek_analysis_from_report(report_files)
            ec = extract_email_content_from_report(report_files)
            if isinstance(ec, dict):
                email_content = ec
                # Enrich parsed_email body from report if still empty
                if not parsed_email['body']:
                    parsed_email['body'] = ec.get('body') or ec.get('content_preview') or ''

    context = {
        'prediction': prediction,
        'deepseek_analysis': deepseek_analysis,
        'parsed_email': parsed_email,
        'email_content': email_content,
    }

    return render(request, 'detector/deepseek_analysis.html', context)


def report_pdf(request, prediction_id):
    """
    Generate and return a PDF version of the security report.
    Supports both UUID (new) and string/integer IDs (legacy).
    """
    prediction = None
    report_files = None
    
    # Try UUID lookup first
    try:
        from uuid import UUID
        val = UUID(str(prediction_id), version=4)
        prediction = EnhancedPredictionHistory.objects.filter(id=prediction_id).first()
    except ValueError:
        # Not a valid UUID, might be a legacy integer ID
        pass

    if prediction:
        # Normal path: found the prediction in DB
        report_files = find_email_report_files(prediction.timestamp, prediction.input_text, str(prediction_id))
        subject_source = prediction.input_text
        timestamp_source = prediction.timestamp
    else:
        # Legacy/Fallback path: Try to find report by ID alone
        logger.info(f"PDF requested for ID {prediction_id} not in DB as UUID. Searching files...")
        report_files = find_email_report_files(None, None, str(prediction_id))
        subject_source = None
        timestamp_source = timezone.now()

    if not report_files or not report_files.get('html'):
        return HttpResponse("Report PDF not found.", status=404)
        
    if HTML is None:
        return HttpResponse("PDF generation library (WeasyPrint) not installed.", status=500)
    
    try:
        # Load HTML content
        with open(report_files['html'], 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Generate PDF
        pdf_file = HTML(string=html_content, base_url=request.build_absolute_uri('/')).write_pdf()
        
        # Prepare filename
        parsed = _parse_email_from_input_text(subject_source) if subject_source else {}
        subject = parsed.get('subject')
        if subject:
            filename = f"{slugify(subject)[:100]}.pdf"
        else:
            filename = f"security_report_{timestamp_source.strftime('%Y%m%d_%H%M%S')}.pdf"
        
        response = HttpResponse(pdf_file, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        return HttpResponse(f"Error generating PDF: {str(e)}", status=500)


def find_email_report_files(timestamp, email_text, prediction_id=None):
    """
    Find report files associated with an email prediction.
    Searches both email_reports/ and enhanced_email_reports/.
    Priority given to ID-based matching.
    """
    search_dirs = [
        Path(__file__).parent.parent / 'email_reports',
        Path(__file__).parent.parent / 'enhanced_email_reports',
    ]

    # Build search tokens from email_text
    search_tokens = [t.lower().strip() for t in re.split(r'[\s,;]+', email_text or '') if len(t) > 3]

    for reports_dir in search_dirs:
        if not reports_dir.exists():
            continue

        json_files = sorted(reports_dir.glob('*.json'), key=lambda f: f.stat().st_mtime, reverse=True)

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)

                matched = False

                def _text_match(candidate):
                    if not candidate:
                        return False
                    cand_low = candidate.lower()
                    # Direct substring check
                    if email_text and (email_text in candidate or candidate in email_text):
                        return True
                    # Token overlap
                    return any(tok in cand_low for tok in search_tokens if len(tok) > 3)

                # Priority 1: ID-based match (Exact)
                if prediction_id:
                    # Check Format 1: email_metadata
                    if 'email_metadata' in report_data:
                        m_id = str(report_data['email_metadata'].get('prediction_id'))
                        if m_id == prediction_id:
                            matched = True
                    
                    # Check Format 2: root key
                    if not matched and str(report_data.get('prediction_id')) == prediction_id:
                        matched = True

                if not matched:
                    pass

                # Format 1: email_metadata (from email_phishing_checker)
                if 'email_metadata' in report_data:
                    meta = report_data['email_metadata']
                    if _text_match(meta.get('subject', '')) or _text_match(meta.get('from', '')):
                        matched = True

                # Format 2: enhanced analysis (from email_checker_with_deepseek)
                if not matched and 'email_results' in report_data:
                    for res in report_data.get('email_results', []):
                        if _text_match(res.get('subject', '')) or _text_match(res.get('from', '')):
                            matched = True
                            break

                # Format 3: flat keys
                if not matched:
                    if _text_match(report_data.get('subject', '')) or _text_match(report_data.get('from', '')):
                        matched = True

                if matched:
                    stem = json_file.stem
                    html_file = reports_dir / f"{stem}.html"
                    txt_file = reports_dir / f"{stem}.txt"
                    return {
                        'json': str(json_file),
                        'html': str(html_file) if html_file.exists() else None,
                        'txt': str(txt_file) if txt_file.exists() else None,
                        'base_name': stem
                    }
            except Exception:
                continue

    return None


def extract_email_content_from_report(report_files):
    """
    Extract email content from report files.
    Returns a structured dict: {subject, from, date, body, content_preview}
    Body = actual email message text (content_preview) NOT the analysis report.
    """
    result = {'subject': '', 'from': '', 'date': '', 'body': '', 'content_preview': ''}
    try:
        if report_files.get('json'):
            with open(report_files['json'], 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            # Format 1: email_metadata (from email_phishing_checker report)
            if 'email_metadata' in report_data:
                meta = report_data['email_metadata']
                result['subject'] = meta.get('subject', '')
                result['from'] = meta.get('sender', meta.get('from', ''))
                result['date'] = meta.get('date', '')
                # content_preview is the actual email text snippet
                preview = meta.get('content_preview', '')
                result['content_preview'] = preview
                # Use content_preview as body (it IS the email text)
                result['body'] = preview

            # Format 2: email_results list (from enhanced checker)
            elif 'email_results' in report_data:
                first = report_data['email_results'][0] if report_data['email_results'] else {}
                result['subject'] = first.get('subject', '')
                result['from'] = first.get('from', '')
                result['body'] = first.get('content_preview', first.get('body', ''))

            # Fallback: direct keys
            else:
                result['subject'] = report_data.get('subject', '')
                result['from'] = report_data.get('from', report_data.get('sender', ''))
                result['date'] = report_data.get('date', '')
                result['body'] = report_data.get('body', report_data.get('content', ''))

        # NOTE: We intentionally do NOT read the TXT file as body —
        # the TXT file contains the analysis report, not the email message.

        return result
    except Exception as e:
        print(f"Error extracting email content: {e}")
        return result


def extract_deepseek_analysis_from_report(report_files):
    """
    Extract DeepSeek analysis from report files.
    Handles multiple JSON formats produced by different checker scripts.
    """
    try:
        if not report_files or not report_files.get('json'):
            return None
        with open(report_files['json'], 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        # Format 1: top-level deepseek_analysis
        if 'deepseek_analysis' in report_data:
            return report_data['deepseek_analysis']

        # Format 2: top-level analysis
        if 'analysis' in report_data:
            return report_data['analysis']

        # Format 3: enhanced_analysis results list — return first entry's deepseek_analysis
        if 'email_results' in report_data:
            for res in report_data['email_results']:
                if res.get('deepseek_verdict') and res['deepseek_verdict'] != 'UNKNOWN':
                    # Build a normalized analysis dict
                    return {
                        'verdict': res.get('deepseek_verdict', 'UNKNOWN'),
                        'traditional_result': res.get('traditional_result', ''),
                        'combined_verdict': res.get('combined_verdict', ''),
                        'subject': res.get('subject', ''),
                        'sender': res.get('from', ''),
                        'confidence': res.get('traditional_confidence', 0),
                        'risk_score': res.get('traditional_confidence', 0),
                        'threat_level': res.get('combined_verdict', 'UNKNOWN'),
                        'analysis_summary': (
                            f"Combined verdict: {res.get('combined_verdict', 'UNKNOWN')}. "
                            f"Traditional: {res.get('traditional_result', 'N/A')}. "
                            f"DeepSeek: {res.get('deepseek_verdict', 'N/A')}."
                        ),
                        'report_files': res.get('report_files'),
                    }

        return None
    except Exception as e:
        print(f"Error extracting DeepSeek analysis: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Email Dashboard — email-only focused view
# ─────────────────────────────────────────────────────────────────────────────

def email_dashboard(request):
    """Email-only security dashboard with fetch/analyze action buttons."""
    search_query = request.GET.get('search', '')
    page_number = request.GET.get('page', 1)
    verdict_filter = request.GET.get('verdict', 'all')

    predictions_qs = EnhancedPredictionHistory.objects.filter(
        prediction_type='email'
    ).order_by('-timestamp')

    if verdict_filter != 'all':
        predictions_qs = predictions_qs.filter(predicted_label__icontains=verdict_filter)

    if search_query:
        predictions_qs = predictions_qs.filter(
            Q(input_text__icontains=search_query) |
            Q(predicted_label__icontains=search_query) |
            Q(domain__icontains=search_query)
        )

    total_emails = EnhancedPredictionHistory.objects.filter(prediction_type='email').count()
    phishing_count = EnhancedPredictionHistory.objects.filter(
        prediction_type='email', predicted_label__in=['PHISHING', 'SPAM', 'MALICIOUS']
    ).count()
    safe_count = total_emails - phishing_count
    avg_conf = EnhancedPredictionHistory.objects.filter(
        prediction_type='email'
    ).aggregate(avg=Avg('confidence_score'))['avg'] or 0

    detection_rate = round((phishing_count / total_emails * 100), 1) if total_emails else 0

    paginator = Paginator(predictions_qs, 12)
    page_obj = paginator.get_page(page_number)

    # Build enriched card data
    email_cards = []
    for p in page_obj:
        parsed = _parse_email_from_input_text(p.input_text)
        label = p.predicted_label.upper()
        if label == 'PHISHING':
            bar_color = 'bg-red-500'
        elif label in ('SPAM', 'MALICIOUS'):
            bar_color = 'bg-orange-500'
        elif label == 'SUSPICIOUS':
            bar_color = 'bg-yellow-500'
        else:
            bar_color = 'bg-green-500'

        email_cards.append({
            'id': p.id,
            'predicted_label': p.predicted_label,
            'confidence_score': p.confidence_score,
            'confidence_percentage': f"{p.confidence_score:.1f}%",
            'risk_score': _compute_risk_score(p),
            'bar_color': bar_color,
            'timestamp': p.timestamp,
            'subject': parsed['subject'] or p.input_text[:60],
            'sender': parsed['from'],
            'date': parsed['date'],
            'body_preview': (parsed['body'] or '')[:200],
            'full_body': parsed['body'] or p.input_text,
            'threat_category': p.threat_category,
            'risk_level': p.risk_level,
            'is_malicious': label in ('PHISHING', 'SPAM', 'MALICIOUS', 'SUSPICIOUS'),
        })

    context = {
        'email_cards': email_cards,
        'page_obj': page_obj,
        'total_emails': total_emails,
        'phishing_count': phishing_count,
        'safe_count': safe_count,
        'avg_confidence': round(avg_conf, 1),
        'detection_rate': detection_rate,
        'search_query': search_query,
        'verdict_filter': verdict_filter,
    }
    return render(request, 'detector/email_dashboard.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# Script launcher views
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(['POST'])
def fetch_emails_view(request):
    """Launch run_email_checker.py always using Gmail account (my_gmail) in background."""
    process_id = _launch_script('run_email_checker.py', extra_args=['--account', 'my_gmail'])
    return JsonResponse({'success': True, 'process_id': process_id})


@csrf_exempt
@require_http_methods(['POST'])
def analyze_emails_deepseek_view(request):
    """Launch email_checker_with_deepseek_env.py in background."""
    process_id = _launch_script('email_checker_with_deepseek_env.py')
    return JsonResponse({'success': True, 'process_id': process_id})


def get_process_status(request, process_id):
    """Poll status + output of a background script process."""
    with _REGISTRY_LOCK:
        entry = _PROCESS_REGISTRY.get(process_id)
    if not entry:
        return JsonResponse({'success': False, 'error': 'Unknown process ID'})
    return JsonResponse({
        'success': True,
        'status': entry['status'],
        'output_lines': entry['output_lines'],
    })


@csrf_exempt
@require_http_methods(['POST'])
def reanalyze_email_api(request, prediction_id):
    """
    Re-run DeepSeek analysis on an existing email prediction.
    Calls analyze_email_with_deepseek directly and returns result JSON.
    """
    prediction = get_object_or_404(EnhancedPredictionHistory, id=prediction_id)

    # Load DeepSeek API key from env
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / '.env')
        deepseek_key = os.environ.get('DEEPSEEK_API_KEY', '')
        if not deepseek_key:
            return JsonResponse({
                'success': False,
                'error': 'DEEPSEEK_API_KEY not set in .env'
            })
    except ImportError:
        return JsonResponse({'success': False, 'error': 'python-dotenv not installed'})

    # Build email_data dict from prediction
    parsed = _parse_email_from_input_text(prediction.input_text)
    email_data = {
        'subject': parsed['subject'] or prediction.input_text[:80],
        'from': parsed['from'],
        'date': parsed['date'],
        'body': parsed['body'],
        'full_text': f"Subject: {parsed['subject']}\nFrom: {parsed['from']}\nDate: {parsed['date']}\n\n{parsed['body']}",
        'id': str(prediction.id),
    }

    try:
        from detector.deepseek_llm_service import analyze_email_with_deepseek, initialize_deepseek_service
        initialize_deepseek_service(deepseek_key)
        result = analyze_email_with_deepseek(email_data, deepseek_key)
        return JsonResponse({'success': True, 'analysis': result})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def get_prediction_stats(request):
    """
    API endpoint to get prediction statistics
    """
    try:
        total = EnhancedPredictionHistory.objects.count()
        malicious = EnhancedPredictionHistory.objects.filter(
            predicted_label__in=['PHISHING', 'SPAM', 'MALICIOUS']
        ).count()
        safe = total - malicious
        
        # Get counts by type
        email_count = EnhancedPredictionHistory.objects.filter(prediction_type='email').count()
        sms_count = EnhancedPredictionHistory.objects.filter(prediction_type='sms').count()
        url_count = EnhancedPredictionHistory.objects.filter(prediction_type='url').count()
        
        # Get counts by threat category
        category_stats = []
        for category in ThreatCategory.objects.all():
            count = EnhancedPredictionHistory.objects.filter(threat_category=category).count()
            if count > 0:
                category_stats.append({
                    'name': category.name,
                    'count': count,
                    'color': category.color
                })
        
        # Get counts by risk level
        risk_stats = []
        for risk in RiskLevel.objects.all():
            count = EnhancedPredictionHistory.objects.filter(risk_level=risk).count()
            if count > 0:
                risk_stats.append({
                    'name': risk.name,
                    'count': count,
                    'color': risk.color,
                    'level': risk.level
                })
        
        return JsonResponse({
            'success': True,
            'stats': {
                'total': total,
                'malicious': malicious,
                'safe': safe,
                'by_type': {
                    'email': email_count,
                    'sms': sms_count,
                    'url': url_count
                },
                'by_category': category_stats,
                'by_risk': risk_stats
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def delete_prediction_api(request, prediction_id):
    """
    API endpoint to delete a prediction
    """
    try:
        prediction = get_object_or_404(EnhancedPredictionHistory, id=prediction_id)
        prediction.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Prediction deleted successfully'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def bulk_delete_predictions(request):
    """
    API endpoint for bulk deletion of predictions
    """
    try:
        data = json.loads(request.body)
        prediction_ids = data.get('ids', [])
        
        if not prediction_ids:
            return JsonResponse({
                'success': False,
                'error': 'No prediction IDs provided'
            })
        
        deleted_count = EnhancedPredictionHistory.objects.filter(
            id__in=prediction_ids
        ).delete()[0]
        
        return JsonResponse({
            'success': True,
            'message': f'Successfully deleted {deleted_count} predictions',
            'deleted_count': deleted_count
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


def serve_report_file(request):
    """
    Serve a report file (JSON/HTML/TXT) directly from the filesystem.
    Query param: ?path=/absolute/path/to/file
    Only permits reading files inside the project directory.
    """
    from django.http import FileResponse, HttpResponseForbidden, HttpResponseNotFound
    import mimetypes

    path_param = request.GET.get('path', '')
    if not path_param:
        return HttpResponseNotFound('No path specified')

    # Safety: only allow files within the project directory
    try:
        target = Path(path_param).resolve()
        project_root = PROJECT_DIR.resolve()
        if not str(target).startswith(str(project_root)):
            return HttpResponseForbidden('Access denied')
        if not target.exists() or not target.is_file():
            return HttpResponseNotFound(f'File not found: {target.name}')
    except Exception:
        return HttpResponseNotFound('Invalid path')

    ext = target.suffix.lower()
    content_types = {
        '.json': 'application/json',
        '.html': 'text/html',
        '.txt': 'text/plain',
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    return FileResponse(open(target, 'rb'), content_type=content_type)
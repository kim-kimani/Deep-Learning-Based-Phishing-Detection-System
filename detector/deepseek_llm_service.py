"""
DeepSeek LLM Service Integration for Enhanced Email Analysis and Report Generation
Uses DeepSeek API for detailed email analysis and comprehensive report generation
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DeepSeekLLMService:
    """
    DeepSeek LLM service for enhanced email analysis and detailed report generation
    Can be used for comprehensive email analysis with detailed reports
    """
    
    def __init__(self, api_key: str):
        """
        Initialize DeepSeek LLM Service
        
        Args:
            api_key (str): DeepSeek API key for authentication
        """
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"  # DeepSeek's chat model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Report storage directory
        self.reports_dir = Path("email_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if DeepSeek API is accessible"""
        try:
            test_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": "Test connection. Reply with 'OK'."
                    }
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("[SUCCESS] DeepSeek LLM service connected successfully")
                return True
            else:
                logger.error(
                    f"[ERROR] DeepSeek API connection failed: {response.status_code}"
                )
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] DeepSeek connection test failed: {e}")
            return False
    
    def analyze_email_with_report(self, email_data: Dict) -> Dict:
        """
        Analyze email content and generate comprehensive report
        
        Args:
            email_data (Dict): Email data including subject, from, body, etc.
            
        Returns:
            Dict: Complete analysis with detailed report
        """
        email_text = email_data.get('full_text') or email_data.get('body') or 'No content body provided.'
        subject = email_data.get('subject', 'No Subject')
        sender = email_data.get('from', 'Unknown Sender')
        date = email_data.get('date', 'Unknown Date')
        
        # Ensure we don't send empty text
        if not email_text or email_text.strip() == '':
            email_text = "No content body provided."
        
        prompt = f"""
        You are an expert cybersecurity analyst and email security specialist. 
        Analyze this email comprehensively and generate a detailed security report.

        EMAIL METADATA:
        - Subject: {subject}
        - From: {sender}
        - Date: {date}
        
        EMAIL CONTENT:
        {email_text}

        Please provide a COMPREHENSIVE SECURITY ANALYSIS covering:

        1. THREAT ASSESSMENT:
           - Overall threat level (LOW/MEDIUM/HIGH/CRITICAL)
           - Confidence score (0-100%)
           - Primary threat category (Phishing, Scam, Malware, Social Engineering, etc.)

        2. TECHNICAL ANALYSIS:
           - Sender reputation indicators
           - Domain analysis
           - URL/Link analysis (if present)
           - Attachment analysis (if mentioned)
           - Email header anomalies

        3. CONTENT ANALYSIS:
           - Social engineering tactics used
           - Psychological manipulation techniques
           - Urgency/Scarcity indicators
           - Authority impersonation attempts
           - Emotional triggers exploited

        4. LINGUISTIC ANALYSIS:
           - Grammar and spelling anomalies
           - Tone and style analysis
           - Cultural/language inconsistencies
           - Professionalism assessment

        5. BEHAVIORAL INDICATORS:
           - Call-to-action analysis
           - Information harvesting attempts
           - Financial transaction requests
           - Credential phishing indicators

        6. RISK SCORING (0-100):
           - Technical risk score
           - Content risk score
           - Behavioral risk score
           - Overall risk score

        7. RECOMMENDATIONS:
           - Immediate actions
           - User education points
           - Technical controls needed
           - Reporting requirements

        Return your analysis as a JSON object with this exact structure:
        {{
            "report_id": "generated-unique-id",
            "analysis_timestamp": "ISO timestamp",
            "email_metadata": {{
                "subject": "{subject}",
                "sender": "{sender}",
                "date": "{date}",
                "content_length": {len(email_text)}
            }},
            "threat_assessment": {{
                "overall_threat_level": "LOW|MEDIUM|HIGH|CRITICAL",
                "confidence_score": 0-100,
                "primary_threat_category": "string",
                "secondary_categories": ["list", "of", "categories"]
            }},
            "technical_analysis": {{
                "sender_reputation": "GOOD|SUSPICIOUS|MALICIOUS",
                "domain_analysis": "detailed domain analysis",
                "url_analysis": ["list of URL findings"],
                "header_anomalies": ["list of header issues"],
                "technical_risk_score": 0-100
            }},
            "content_analysis": {{
                "social_engineering_tactics": ["list of tactics"],
                "psychological_indicators": ["list of indicators"],
                "urgency_indicators": ["list of urgency signs"],
                "authority_indicators": ["list of authority signs"],
                "content_risk_score": 0-100
            }},
            "linguistic_analysis": {{
                "grammar_issues": ["list of grammar problems"],
                "tone_analysis": "analysis of writing tone",
                "consistency_issues": ["list of inconsistencies"],
                "professionalism_score": 0-100
            }},
            "behavioral_analysis": {{
                "call_to_action_analysis": "analysis of requested actions",
                "information_harvesting": ["list of data collection attempts"],
                "financial_indicators": ["list of financial red flags"],
                "behavioral_risk_score": 0-100
            }},
            "risk_scoring": {{
                "technical_risk": 0-100,
                "content_risk": 0-100,
                "behavioral_risk": 0-100,
                "overall_risk_score": 0-100,
                "risk_level": "LOW|MEDIUM|HIGH|CRITICAL"
            }},
            "recommendations": {{
                "immediate_actions": ["list of immediate steps"],
                "user_education": ["list of education points"],
                "technical_controls": ["list of technical measures"],
                "reporting_actions": ["list of reporting steps"]
            }},
            "executive_summary": "Brief 3-4 sentence summary for executives",
            "detailed_findings": "Comprehensive detailed analysis paragraph",
            "verdict": "SAFE|SUSPICIOUS|MALICIOUS|CRITICAL"
        }}
        """
        
        return self._call_deepseek_api(prompt, "email_comprehensive", email_data)
    
    def generate_detailed_email_report(self, email_data: Dict, analysis_result: Dict) -> Dict:
        """
        Generate a detailed HTML/PDF ready report from analysis results
        
        Args:
            email_data (Dict): Original email data
            analysis_result (Dict): Analysis results from DeepSeek
            
        Returns:
            Dict: Report generation results including file paths
        """
        try:
            # Generate report ID
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(email_data.get('subject', '')) % 10000:04d}"
            
            # Create HTML report
            prediction_id = email_data.get('id')
            html_report = self._generate_html_report(email_data, analysis_result, report_id, prediction_id)
            
            # Create JSON report
            json_report = self._generate_json_report(email_data, analysis_result, report_id)
            
            # Create text summary
            text_report = self._generate_text_report(email_data, analysis_result, report_id)
            
            # Save reports to files
            report_files = self._save_reports(report_id, html_report, json_report, text_report)
            
            return {
                "success": True,
                "report_id": report_id,
                "report_files": report_files,
                "html_content": html_report,
                "analysis_summary": analysis_result.get("executive_summary", ""),
                "verdict": analysis_result.get("verdict", "UNKNOWN")
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "report_id": None
            }
    
    def _generate_html_report(self, email_data: Dict, analysis: Dict, report_id: str, prediction_id: str = None) -> str:
        """Generate a modern, beautiful HTML formatted report"""
        threat_assessment = analysis.get("threat_assessment", {})
        threat_level = threat_assessment.get("overall_threat_level", "UNKNOWN")
        risk_scoring = analysis.get("risk_scoring", {})
        risk_score = risk_scoring.get("overall_risk_score", 0)
        verdict = analysis.get("verdict", "UNKNOWN")
        
        # Color & Icon mapping based on threat level
        theme = {
            "LOW": {"color": "#10b981", "bg": "rgba(16, 185, 129, 0.1)", "icon": "🛡️", "label": "SAFE"},
            "MEDIUM": {"color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.1)", "icon": "⚠️", "label": "SUSPICIOUS"},
            "HIGH": {"color": "#ef4444", "bg": "rgba(239, 68, 68, 0.1)", "icon": "🚫", "label": "MALICIOUS"},
            "CRITICAL": {"color": "#7f1d1d", "bg": "rgba(127, 29, 29, 0.1)", "icon": "☢️", "label": "CRITICAL"},
            "UNKNOWN": {"color": "#6b7280", "bg": "rgba(107, 114, 128, 0.1)", "icon": "❓", "label": "UNKNOWN"}
        }
        
        current_theme = theme.get(threat_level, theme["UNKNOWN"])
        accent = current_theme["color"]
        
        # Build recommendation cards
        rec_html = ""
        recs = analysis.get("recommendations", {})
        for category, items in [
            ("Immediate Actions", recs.get("immediate_actions", [])),
            ("User Education", recs.get("user_education", []))
        ]:
            if items:
                rec_html += f'<div class="rec-group"><h3>{category}</h3><ul>'
                for item in items:
                    rec_html += f'<li>{item}</li>'
                rec_html += '</ul></div>'

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Security Analysis - {report_id}</title>
            <style>
                :root {{
                    --primary: #0f172a;
                    --secondary: #1e293b;
                    --accent: {accent};
                    --text: #f8fafc;
                    --text-muted: #94a3b8;
                    --card-bg: #1e293b;
                    --status-bg: {current_theme['bg']};
                }}
                
                @media print {{
                    .download-btn {{ display: none !important; }}
                    body {{ background: white !important; color: black !important; }}
                    .container {{ border: none !important; box-shadow: none !important; width: 100% !important; max-width: none !important; margin: 0 !important; padding: 20px !important; }}
                    .card {{ border: 1px solid #ddd !important; background: white !important; color: black !important; box-shadow: none !important; }}
                    .verdict-card {{ background: white !important; color: black !important; border: 2px solid var(--accent) !important; }}
                    .status-badge {{ background: #f8fafc !important; border: 1px solid #ddd !important; }}
                    .header-brand strong {{ color: black !important; }}
                }}
                
                body {{
                    font-family: 'Inter', -apple-system, sans-serif;
                    background-color: var(--primary);
                    color: var(--text);
                    margin: 0;
                    line-height: 1.6;
                    padding-bottom: 50px;
                }}
                
                .container {{
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 0 20px;
                }}
                
                header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 40px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid var(--secondary);
                }}
                
                .header-brand {{ display: flex; align-items: center; gap: 12px; }}
                .logo {{ font-size: 24px; }}
                .report-id {{ color: var(--text-muted); font-size: 14px; font-family: monospace; }}
                
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 30px; }}
                
                .card {{
                    background: var(--card-bg);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid rgba(255,255,255,0.05);
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                }}
                
                .verdict-card {{
                    grid-column: span 2;
                    display: flex;
                    align-items: center;
                    gap: 40px;
                    background: linear-gradient(135deg, var(--card-bg) 0%, #2a3a5a 100%);
                    border-left: 6px solid var(--accent);
                }}
                
                .status-badge {{
                    background: var(--status-bg);
                    color: var(--accent);
                    padding: 20px;
                    border-radius: 20px;
                    text-align: center;
                    min-width: 140px;
                }}
                
                .status-icon {{ font-size: 40px; margin-bottom: 8px; display: block; }}
                .status-label {{ font-weight: 800; font-size: 18px; letter-spacing: 1px; }}
                
                .verdict-text h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
                .verdict-summary {{ color: var(--text-muted); font-size: 16px; }}
                
                .download-btn {{
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    background: #dc2626;
                    color: white;
                    padding: 10px 18px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-size: 14px;
                    font-weight: 700;
                    border: none;
                    cursor: pointer;
                    transition: all 0.2s;
                    box-shadow: 0 4px 10px rgba(220, 38, 38, 0.3);
                }}
                
                .download-btn:hover {{
                    background: #b91c1c;
                    transform: translateY(-2px);
                    box-shadow: 0 6px 15px rgba(220, 38, 38, 0.4);
                }}
                
                .section-title {{
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    color: var(--accent);
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                
                .data-row {{ display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 14px; }}
                .data-label {{ color: var(--text-muted); }}
                .data-value {{ font-weight: 600; text-align: right; }}
                
                .risk-meter {{
                    height: 12px;
                    background: #334155;
                    border-radius: 6px;
                    overflow: hidden;
                    margin: 15px 0;
                }}
                
                .risk-fill {{
                    height: 100%;
                    background: var(--accent);
                    width: {risk_score}%;
                    transition: width 1s ease;
                }}

                .meta-box {{ display: flex; flex-direction: column; gap: 8px; }}
                .meta-item {{ padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; }}
                .meta-val {{ display: block; font-weight: 500; font-size: 14px; word-break: break-all; }}
                
                .pill {{
                    display: inline-block;
                    padding: 4px 12px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 20px;
                    font-size: 12px;
                    margin: 2px;
                }}
                
                .rec-group h3 {{ font-size: 16px; margin: 20px 0 10px 0; color: var(--accent); }}
                .rec-group ul {{ padding-left: 20px; margin: 0; }}
                .rec-group li {{ margin-bottom: 8px; font-size: 14px; }}
                
                footer {{ text-align: center; margin-top: 60px; color: var(--text-muted); font-size: 12px; }}
                
                @media (max-width: 768px) {{
                    .grid {{ grid-template-columns: 1fr; }}
                    .verdict-card {{ flex-direction: column; gap: 20px; text-align: center; }}
                    .container {{ margin: 20px auto; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div class="header-brand">
                        <span class="logo">🛡️</span>
                        <div>
                            <strong style="display: block;">DEEPSEEK AI</strong>
                            <span style="font-size: 12px; color: var(--text-muted);">SECURITY ANALYZER</span>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <div class="report-id">REF: {report_id}</div>
                    </div>
                </header>

                <div class="grid">
                    <div class="card verdict-card">
                        <div class="status-badge">
                            <span class="status-icon">{current_theme['icon']}</span>
                            <span class="status-label">{current_theme['label']}</span>
                        </div>
                        <div class="verdict-text">
                            <h1>Analysis Verdict: {verdict}</h1>
                            <p class="verdict-summary">{analysis.get("executive_summary", "No summary available")}</p>
                        </div>
                    </div>

                    <div class="card">
                        <div class="section-title">📊 Threat Score</div>
                        <div style="font-size: 32px; font-weight: 800; margin-bottom: 5px;">{risk_score}<small style="font-size: 14px; color: var(--text-muted); font-weight: 400;"> / 100</small></div>
                        <div class="risk-meter"><div class="risk-fill"></div></div>
                        <div class="data-row"><span class="data-label">Technical Risk</span><span class="data-value">{risk_scoring.get('technical_risk', 0)}%</span></div>
                        <div class="data-row"><span class="data-label">Content Risk</span><span class="data-value">{risk_scoring.get('content_risk', 0)}%</span></div>
                        <div class="data-row"><span class="data-label">Behavioral Risk</span><span class="data-value">{risk_scoring.get('behavioral_risk', 0)}%</span></div>
                    </div>

                    <div class="card">
                        <div class="section-title">📧 Email Metadata</div>
                        <div class="meta-box">
                            <div class="meta-item">
                                <span class="data-label">Subject</span>
                                <span class="meta-val">{email_data.get('subject', 'N/A')}</span>
                            </div>
                            <div class="meta-item">
                                <span class="data-label">Sender</span>
                                <span class="meta-val">{email_data.get('from', 'N/A')}</span>
                            </div>
                            <div class="data-row" style="margin-top: 10px;">
                                <span class="data-label">Timestamp</span>
                                <span class="data-value">{email_data.get('date', 'N/A')}</span>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="section-title">🛡️ Analysis Details</div>
                        <div class="data-row"><span class="data-label">Category</span><span class="data-value">{threat_assessment.get('primary_threat_category', 'N/A')}</span></div>
                        <div class="data-row"><span class="data-label">Confidence</span><span class="data-value">{threat_assessment.get('confidence_score', 0)}%</span></div>
                        <div class="data-row"><span class="data-label">Reputation</span><span class="data-value">{analysis.get('technical_analysis', {}).get('sender_reputation', 'N/A')}</span></div>
                        <div style="margin-top: 15px;">
                            <span class="data-label" style="display:block; margin-bottom:8px;">Tactics Identified:</span>
                            {" ".join([f'<span class="pill">{t}</span>' for t in analysis.get('content_analysis', {}).get('social_engineering_tactics', [])])}
                        </div>
                    </div>

                    <div class="card">
                        <div class="section-title">🎯 Recommendations</div>
                        {rec_html}
                    </div>

                    <div class="card" style="grid-column: span 2;">
                        <div class="section-title">📝 Detailed Findings</div>
                        <p style="font-size: 14px; color: var(--text-muted); white-space: pre-wrap;">{analysis.get("detailed_findings", "No detailed findings available")}</p>
                    </div>
                </div>

                <footer>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &bull; DeepSeek AI v1.0</p>
                    <p>&copy; Deep-Learning Spam Detection System</p>
                    
                    {f'''<div style="margin-top: 30px; display: flex; justify-content: center;">
                        <a href="/prediction/{prediction_id}/pdf/" class="download-btn" target="_blank" style="padding: 12px 24px; font-size: 15px;">
                            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="margin-right: 10px;"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                            DOWNLOAD COMPLETE PDF REPORT
                        </a>
                    </div>''' if prediction_id else ''}
                </footer>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_json_report(self, email_data: Dict, analysis: Dict, report_id: str) -> Dict:
        """Generate JSON formatted report"""
        return {
            "report_id": report_id,
            "generated_at": datetime.now().isoformat(),
            "email_metadata": {
                "prediction_id": email_data.get('id'),
                "subject": email_data.get('subject'),
                "sender": email_data.get('from'),
                "date": email_data.get('date'),
                "content_preview": email_data.get('full_text', '')[:500] + "..." if len(email_data.get('full_text', '')) > 500 else email_data.get('full_text', '')
            },
            "analysis_results": analysis,
            "report_summary": {
                "threat_level": analysis.get("threat_assessment", {}).get("overall_threat_level", "UNKNOWN"),
                "risk_score": analysis.get("risk_scoring", {}).get("overall_risk_score", 0),
                "verdict": analysis.get("verdict", "UNKNOWN"),
                "executive_summary": analysis.get("executive_summary", "")
            }
        }
    
    def _generate_text_report(self, email_data: Dict, analysis: Dict, report_id: str) -> str:
        """Generate plain text report"""
        threat_level = analysis.get("threat_assessment", {}).get("overall_threat_level", "UNKNOWN")
        risk_score = analysis.get("risk_scoring", {}).get("overall_risk_score", 0)
        verdict = analysis.get("verdict", "UNKNOWN")
        
        text = f"""
        ============================================================
        EMAIL SECURITY ANALYSIS REPORT
        ============================================================
        Report ID: {report_id}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        EMAIL DETAILS:
        - Subject: {email_data.get('subject', 'No Subject')}
        - From: {email_data.get('from', 'Unknown Sender')}
        - Date: {email_data.get('date', 'Unknown Date')}
        
        EXECUTIVE SUMMARY:
        {analysis.get("executive_summary", "No summary available")}
        
        VERDICT: {verdict}
        Threat Level: {threat_level}
        Overall Risk Score: {risk_score}/100
        
        THREAT ASSESSMENT:
        - Primary Category: {analysis.get("threat_assessment", {}).get("primary_threat_category", "Unknown")}
        - Confidence: {analysis.get("threat_assessment", {}).get("confidence_score", 0)}%
        
        RISK SCORING:
        - Technical Risk: {analysis.get("risk_scoring", {}).get("technical_risk", 0)}/100
        - Content Risk: {analysis.get("risk_scoring", {}).get("content_risk", 0)}/100
        - Behavioral Risk: {analysis.get("risk_scoring", {}).get("behavioral_risk", 0)}/100
        
        RECOMMENDED ACTIONS:
        """
        
        # Add immediate actions
        for action in analysis.get("recommendations", {}).get("immediate_actions", []):
            text += f"  • {action}\n"
        
        text += f"""
        
        DETAILED FINDINGS:
        {analysis.get("detailed_findings", "No detailed findings available")}
        
        ============================================================
        Report generated by DeepSeek AI Security Analyzer
        Model: {self.model}
        ============================================================
        """
        
        return text
    
    def _save_reports(self, report_id: str, html_report: str, json_report: Dict, text_report: str) -> Dict:
        """Save all report formats to files"""
        try:
            # Save HTML report
            html_file = self.reports_dir / f"{report_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Save JSON report
            json_file = self.reports_dir / f"{report_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            
            # Save text report
            text_file = self.reports_dir / f"{report_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            return {
                "html": str(html_file),
                "json": str(json_file),
                "text": str(text_file),
                "all_files": [str(html_file), str(json_file), str(text_file)]
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save reports: {e}")
            return {}
    
    def _call_deepseek_api(self, prompt: str, content_type: str, email_data: Dict) -> Dict:
        """
        Make API call to DeepSeek LLM
        
        Args:
            prompt (str): Analysis prompt
            content_type (str): Type of content being analyzed
            email_data (Dict): Original email data
            
        Returns:
            Dict: API response or error information
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a cybersecurity expert specializing in email security analysis. "
                            "Always respond with valid JSON format only. No additional text."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60  # 60 second timeout for comprehensive analysis
            )
            
            if response.status_code == 200:
                response_data = response.json()
                llm_response = response_data['choices'][0]['message']['content']
                
                # Try to parse the JSON response
                try:
                    analysis_result = json.loads(llm_response)
                    analysis_result['analysis_timestamp'] = datetime.now().isoformat()
                    analysis_result['model_used'] = self.model
                    analysis_result['content_type'] = content_type
                    analysis_result['api_call_success'] = True
                    
                    logger.info(
                        f"[SUCCESS] DeepSeek analysis completed for {content_type}"
                    )
                    return analysis_result
                    
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[ERROR] Failed to parse DeepSeek JSON response: {e}"
                    )
                    return self._create_error_response(
                        "LLM response parsing failed", content_type, email_data
                    )
            else:
                logger.error(
                    f"[ERROR] DeepSeek API call failed: {response.status_code} - "
                    f"{response.text}"
                )
                return self._create_error_response(
                    f"API call failed: {response.status_code}", content_type, email_data
                )
                
        except requests.exceptions.Timeout:
            logger.error("[ERROR] DeepSeek API call timed out")
            return self._create_error_response("API timeout", content_type, email_data)
            
        except Exception as e:
            logger.error(f"[ERROR] DeepSeek API call failed: {e}")
            return self._create_error_response(str(e), content_type, email_data)
    
    def _create_error_response(self, error_msg: str, content_type: str, email_data: Dict) -> Dict:
        """
        Create standardized error response
        
        Args:
            error_msg (str): Error message
            content_type (str): Type of content being analyzed
            email_data (Dict): Original email data
            
        Returns:
            Dict: Error response in standard format
        """
        return {
            "report_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "analysis_timestamp": datetime.now().isoformat(),
            "email_metadata": {
                "subject": email_data.get('subject', 'Unknown'),
                "sender": email_data.get('from', 'Unknown'),
                "date": email_data.get('date', 'Unknown')
            },
            "threat_assessment": {
                "overall_threat_level": "UNKNOWN",
                "confidence_score": 0,
                "primary_threat_category": "Analysis Failed",
                "secondary_categories": []
            },
            "technical_analysis": {
                "sender_reputation": "UNKNOWN",
                "domain_analysis": "Analysis failed",
                "url_analysis": [],
                "header_anomalies": [],
                "technical_risk_score": 0
            },
            "content_analysis": {
                "social_engineering_tactics": [],
                "psychological_indicators": [],
                "urgency_indicators": [],
                "authority_indicators": [],
                "content_risk_score": 0
            },
            "risk_scoring": {
                "technical_risk": 0,
                "content_risk": 0,
                "behavioral_risk": 0,
                "overall_risk_score": 0,
                "risk_level": "UNKNOWN"
            },
            "recommendations": {
                "immediate_actions": ["Check analysis logs for errors"],
                "user_education": [],
                "technical_controls": [],
                "reporting_actions": []
            },
            "executive_summary": f"Analysis failed: {error_msg}",
            "detailed_findings": f"Unable to perform comprehensive analysis due to: {error_msg}",
            "verdict": "UNKNOWN",
            "error": True,
            "error_message": error_msg,
            "model_used": self.model,
            "content_type": content_type,
            "api_call_success": False
        }


# Global DeepSeek service instance
deepseek_service = None

def initialize_deepseek_service(api_key: str) -> bool:
    """
    Initialize global DeepSeek service instance
    
    Args:
        api_key (str): DeepSeek API key
        
    Returns:
        bool: True if initialization successful
    """
    global deepseek_service
    try:
        deepseek_service = DeepSeekLLMService(api_key)
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize DeepSeek service: {e}")
        deepseek_service = None
        return False


def get_deepseek_service() -> Optional[DeepSeekLLMService]:
    """
    Get the global DeepSeek service instance
    
    Returns:
        Optional[DeepSeekLLMService]: Service instance or None if not initialized
    """
    return deepseek_service


def analyze_email_with_deepseek(email_data: Dict, api_key: str = None) -> Dict:
    """
    Convenience function to analyze email with DeepSeek
    
    Args:
        email_data (Dict): Email data to analyze
        api_key (str): Optional API key (uses global service if None)
        
    Returns:
        Dict: Complete analysis with report
    """
    global deepseek_service
    
    # Initialize service if needed
    if deepseek_service is None and api_key:
        initialize_deepseek_service(api_key)
    
    if deepseek_service is None:
        return {
            "error": True,
            "error_message": "DeepSeek service not initialized. Provide API key.",
            "analysis": None
        }
    
    try:
        # Perform analysis
        analysis_result = deepseek_service.analyze_email_with_report(email_data)
        
        # Generate reports
        if not analysis_result.get("error", False):
            report_result = deepseek_service.generate_detailed_email_report(email_data, analysis_result)
            analysis_result["report_generation"] = report_result
        
        return {
            "error": False,
            "analysis": analysis_result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Email analysis failed: {e}")
        return {
            "error": True,
            "error_message": str(e),
            "analysis": None,
            "success": False
        }

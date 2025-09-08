"""
Groq LLM Service Integration for Enhanced Content Analysis
Uses Groq API for detailed threat analysis when DL models 
report non-malicious content
"""

import json
import logging
from typing import Dict, Optional
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class GroqLLMService:
    """
    Groq LLM service for enhanced content analysis
    Only called when deep learning models report content as non-malicious
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Groq LLM Service
        
        Args:
            api_key (str): Groq API key for authentication
        """
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"  # Current supported Groq model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if Groq API is accessible"""
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
                logger.info("[SUCCESS] Groq LLM service connected successfully")
                return True
            else:
                logger.error(
                    f"[ERROR] Groq API connection failed: {response.status_code}"
                )
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Groq connection test failed: {e}")
            return False
    
    def analyze_email_content(self, email_text: str) -> Dict:
        """
        Analyze email content for potential threats missed by DL models
        
        Args:
            email_text (str): Email content to analyze
            
        Returns:
            Dict: Analysis results with threat assessment
        """
        prompt = f"""
        You are an expert cybersecurity analyst. Analyze this email content for potential threats that automated systems might miss.

        Email Content:
        {email_text}

        Provide a detailed security analysis focusing on:
        1. Social engineering tactics
        2. Suspicious language patterns
        3. Urgency manipulation
        4. Authority impersonation
        5. Information harvesting attempts
        6. Psychological manipulation techniques

        Return your analysis as a JSON object with this exact structure:
        {{
            "threat_level": "LOW|MEDIUM|HIGH",
            "confidence": 0-100,
            "threats_detected": ["list of specific threats found"],
            "social_engineering_indicators": ["list of social engineering tactics"],
            "recommended_action": "SAFE|CAUTION|BLOCK",
            "analysis_summary": "brief summary of findings",
            "detailed_report": "detailed explanation of the analysis"
        }}
        """
        
        return self._call_groq_api(prompt, "email")
    
    def analyze_sms_content(self, sms_text: str) -> Dict:
        """
        Analyze SMS content for potential threats missed by DL models
        
        Args:
            sms_text (str): SMS content to analyze
            
        Returns:
            Dict: Analysis results with threat assessment
        """
        prompt = f"""
        You are an expert cybersecurity analyst. Analyze this SMS/text message for potential threats that automated systems might miss.

        SMS Content:
        {sms_text}

        Provide a detailed security analysis focusing on:
        1. Smishing (SMS phishing) indicators
        2. Fraudulent schemes
        3. Urgent action requests
        4. Suspicious links or numbers
        5. Identity theft attempts
        6. Financial scam patterns

        Return your analysis as a JSON object with this exact structure:
        {{
            "threat_level": "LOW|MEDIUM|HIGH",
            "confidence": 0-100,
            "threats_detected": ["list of specific threats found"],
            "scam_indicators": ["list of scam indicators"],
            "recommended_action": "SAFE|CAUTION|BLOCK",
            "analysis_summary": "brief summary of findings",
            "detailed_report": "detailed explanation of the analysis"
        }}
        """
        
        return self._call_groq_api(prompt, "sms")
    
    def analyze_url_content(self, url: str) -> Dict:
        """
        Analyze URL for potential threats missed by DL models
        
        Args:
            url (str): URL to analyze
            
        Returns:
            Dict: Analysis results with threat assessment
        """
        prompt = f"""
        You are an expert cybersecurity analyst. Analyze this URL for potential threats that automated systems might miss.

        URL:
        {url}

        Provide a detailed security analysis focusing on:
        1. Domain reputation indicators
        2. URL structure anomalies
        3. Typosquatting attempts
        4. Suspicious subdomains
        5. Known malicious patterns
        6. Homograph attacks

        Return your analysis as a JSON object with this exact structure:
        {{
            "threat_level": "LOW|MEDIUM|HIGH",
            "confidence": 0-100,
            "threats_detected": ["list of specific threats found"],
            "url_indicators": ["list of suspicious URL indicators"],
            "recommended_action": "SAFE|CAUTION|BLOCK",
            "analysis_summary": "brief summary of findings",
            "detailed_report": "detailed explanation of the analysis"
        }}
        """
        
        return self._call_groq_api(prompt, "url")
    
    def _call_groq_api(self, prompt: str, content_type: str) -> Dict:
        """
        Make API call to Groq LLM
        
        Args:
            prompt (str): Analysis prompt
            content_type (str): Type of content being analyzed
            
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
                            "You are a cybersecurity expert. Always respond "
                            "with valid JSON format only. No additional text."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30  # 30 second timeout for analysis
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
                    
                    logger.info(
                        f"[SUCCESS] Groq analysis completed for {content_type}"
                    )
                    return analysis_result
                    
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[ERROR] Failed to parse Groq JSON response: {e}"
                    )
                    return self._create_error_response(
                        "LLM response parsing failed", content_type
                    )
            else:
                logger.error(
                    f"[ERROR] Groq API call failed: {response.status_code} - "
                    f"{response.text}"
                )
                return self._create_error_response(
                    f"API call failed: {response.status_code}", content_type
                )
                
        except requests.exceptions.Timeout:
            logger.error("[ERROR] Groq API call timed out")
            return self._create_error_response("API timeout", content_type)
            
        except Exception as e:
            logger.error(f"[ERROR] Groq API call failed: {e}")
            return self._create_error_response(str(e), content_type)
    
    def _create_error_response(self, error_msg: str, content_type: str) -> Dict:
        """
        Create standardized error response
        
        Args:
            error_msg (str): Error message
            content_type (str): Type of content being analyzed
            
        Returns:
            Dict: Error response in standard format
        """
        return {
            "threat_level": "UNKNOWN",
            "confidence": 0,
            "threats_detected": [],
            "recommended_action": "CAUTION",
            "analysis_summary": f"LLM analysis failed: {error_msg}",
            "detailed_report": (
                f"Unable to perform enhanced analysis due to: {error_msg}. "
                "Relying on deep learning model results only."
            ),
            "error": True,
            "error_message": error_msg,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": self.model,
            "content_type": content_type
        }


# Global Groq service instance
groq_service = None

def initialize_groq_service(api_key: str) -> bool:
    """
    Initialize global Groq service instance
    
    Args:
        api_key (str): Groq API key
        
    Returns:
        bool: True if initialization successful
    """
    global groq_service
    try:
        groq_service = GroqLLMService(api_key)
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize Groq service: {e}")
        groq_service = None
        return False


def get_groq_service() -> Optional[GroqLLMService]:
    """
    Get the global Groq service instance
    
    Returns:
        Optional[GroqLLMService]: Service instance or None if not initialized
    """
    return groq_service

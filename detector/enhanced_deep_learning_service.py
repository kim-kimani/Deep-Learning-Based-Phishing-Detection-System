"""
Enhanced Deep Learning Service with Groq LLM Integration
Combines DL models with LLM analysis for comprehensive threat detection
"""

import logging
import os
from typing import Dict, Optional
from pathlib import Path

from .deep_learning_service import deep_learning_service
from .groq_llm_service import get_groq_service, initialize_groq_service

logger = logging.getLogger(__name__)

# Groq API Key - Retrieved from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class EnhancedDeepLearningService:
    """
    Enhanced service combining Deep Learning models with Groq LLM analysis
    
    Workflow:
    1. DL model analyzes input
    2. If DL reports malicious -> return immediately
    3. If DL reports non-malicious -> pass to LLM for detailed analysis
    """
    
    def __init__(self):
        """Initialize the enhanced service"""
        self.dl_service = deep_learning_service
        self.groq_service = None
        self.groq_available = False
        
        # Initialize Groq service
        self._initialize_groq()
    
    def _initialize_groq(self):
        """Initialize Groq LLM service"""
        try:
            # Try to get API key from environment first
            api_key = os.getenv('GROQ_API_KEY', GROQ_API_KEY)
            
            if initialize_groq_service(api_key):
                self.groq_service = get_groq_service()
                self.groq_available = True
                logger.info("[SUCCESS] Enhanced service with Groq LLM initialized")
            else:
                logger.warning(
                    "[WARNING] Groq LLM service unavailable, "
                    "using DL models only"
                )
                self.groq_available = False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Groq service: {e}")
            self.groq_available = False
    
    def predict_email(self, email_text: str) -> Dict:
        """
        Enhanced email prediction with LLM analysis
        
        Args:
            email_text (str): Email content to analyze
            
        Returns:
            Dict: Combined analysis results
        """
        # Step 1: Get DL model prediction
        dl_result = self.dl_service.predict_email(email_text)
        
        # Step 2: Check if DL model found it malicious
        is_malicious_dl = dl_result.get('is_malicious', False)
        
        if is_malicious_dl:
            # DL found it malicious - return immediately
            return {
                "dl_result": "malicious",
                "dl_confidence": dl_result.get('confidence_score', 0),
                "dl_analysis": dl_result,
                "llm_analysis": None,
                "final_recommendation": "BLOCK",
                "analysis_method": "deep_learning_only",
                "threat_detected": True
            }
        
        # Step 3: DL says non-malicious, analyze with LLM if available
        llm_analysis = None
        if self.groq_available and self.groq_service:
            try:
                llm_analysis = self.groq_service.analyze_email_content(email_text)
            except Exception as e:
                logger.error(f"[ERROR] LLM analysis failed: {e}")
                llm_analysis = None
        
        # Step 4: Combine results
        return self._combine_results(
            dl_result, llm_analysis, "email", email_text
        )
    
    def predict_sms(self, sms_text: str) -> Dict:
        """
        Enhanced SMS prediction with LLM analysis
        
        Args:
            sms_text (str): SMS content to analyze
            
        Returns:
            Dict: Combined analysis results
        """
        # Step 1: Get DL model prediction
        dl_result = self.dl_service.predict_sms(sms_text)
        
        # Step 2: Check if DL model found it malicious
        is_malicious_dl = dl_result.get('is_malicious', False)
        
        if is_malicious_dl:
            # DL found it malicious - return immediately
            return {
                "dl_result": "malicious",
                "dl_confidence": dl_result.get('confidence_score', 0),
                "dl_analysis": dl_result,
                "llm_analysis": None,
                "final_recommendation": "BLOCK",
                "analysis_method": "deep_learning_only",
                "threat_detected": True
            }
        
        # Step 3: DL says non-malicious, analyze with LLM if available
        llm_analysis = None
        if self.groq_available and self.groq_service:
            try:
                llm_analysis = self.groq_service.analyze_sms_content(sms_text)
            except Exception as e:
                logger.error(f"[ERROR] LLM analysis failed: {e}")
                llm_analysis = None
        
        # Step 4: Combine results
        return self._combine_results(
            dl_result, llm_analysis, "sms", sms_text
        )
    
    def predict_url(self, url: str) -> Dict:
        """
        Enhanced URL prediction with LLM analysis
        
        Args:
            url (str): URL to analyze
            
        Returns:
            Dict: Combined analysis results
        """
        # Step 1: Get DL model prediction
        dl_result = self.dl_service.predict_url(url)
        
        # Step 2: Check if DL model found it malicious
        is_malicious_dl = dl_result.get('is_malicious', False)
        
        if is_malicious_dl:
            # DL found it malicious - return immediately
            return {
                "dl_result": "malicious",
                "dl_confidence": dl_result.get('confidence_score', 0),
                "dl_analysis": dl_result,
                "llm_analysis": None,
                "final_recommendation": "BLOCK",
                "analysis_method": "deep_learning_only",
                "threat_detected": True
            }
        
        # Step 3: DL says non-malicious, analyze with LLM if available
        llm_analysis = None
        if self.groq_available and self.groq_service:
            try:
                llm_analysis = self.groq_service.analyze_url_content(url)
            except Exception as e:
                logger.error(f"[ERROR] LLM analysis failed: {e}")
                llm_analysis = None
        
        # Step 4: Combine results
        return self._combine_results(
            dl_result, llm_analysis, "url", url
        )
    
    def _combine_results(
        self, 
        dl_result: Dict, 
        llm_analysis: Optional[Dict], 
        content_type: str,
        content: str
    ) -> Dict:
        """
        Combine DL and LLM analysis results
        
        Args:
            dl_result (Dict): Deep learning model results
            llm_analysis (Optional[Dict]): LLM analysis results
            content_type (str): Type of content analyzed
            content (str): Original content
            
        Returns:
            Dict: Combined analysis results
        """
        base_result = {
            "dl_result": "non-malicious",
            "dl_confidence": dl_result.get('confidence_score', 0),
            "dl_analysis": dl_result,
            "llm_analysis": llm_analysis,
            "analysis_method": "deep_learning_with_llm" if llm_analysis else "deep_learning_only",
            "threat_detected": False
        }
        
        if llm_analysis and not llm_analysis.get('error', False):
            # LLM analysis available - use it to enhance results
            llm_threat_level = llm_analysis.get('threat_level', 'LOW')
            llm_confidence = llm_analysis.get('confidence', 0)
            llm_action = llm_analysis.get('recommended_action', 'SAFE')
            
            # Determine final recommendation based on LLM analysis
            if llm_threat_level in ['HIGH'] or llm_action == 'BLOCK':
                base_result.update({
                    "final_recommendation": "BLOCK",
                    "threat_detected": True,
                    "enhanced_analysis": "LLM detected high threat level"
                })
            elif llm_threat_level == 'MEDIUM' or llm_action == 'CAUTION':
                base_result.update({
                    "final_recommendation": "CAUTION",
                    "threat_detected": False,
                    "enhanced_analysis": "LLM detected moderate risk"
                })
            else:
                base_result.update({
                    "final_recommendation": "SAFE",
                    "threat_detected": False,
                    "enhanced_analysis": "Both DL and LLM indicate content is safe"
                })
                
            # Add LLM confidence to overall assessment
            base_result["combined_confidence"] = (
                dl_result.get('confidence_score', 0) * 0.7 + 
                llm_confidence * 0.3
            )
            
        else:
            # No LLM analysis - rely on DL only
            base_result.update({
                "final_recommendation": "SAFE",
                "enhanced_analysis": "DL analysis only - content appears safe",
                "combined_confidence": dl_result.get('confidence_score', 0)
            })
        
        return base_result
    
    def get_service_status(self) -> Dict:
        """Get status of the enhanced service"""
        dl_status = self.dl_service.get_service_status() if self.dl_service else {}
        
        return {
            "service_type": "Enhanced Deep Learning + LLM",
            "dl_service_available": self.dl_service is not None,
            "llm_service_available": self.groq_available,
            "dl_status": dl_status,
            "groq_model": "llama-3.1-70b-versatile" if self.groq_available else None,
            "analysis_modes": [
                "deep_learning_only",
                "deep_learning_with_llm" if self.groq_available else None
            ]
        }


# Global enhanced service instance
enhanced_service = None

try:
    enhanced_service = EnhancedDeepLearningService()
    logger.info(
        "[SUCCESS] Enhanced Deep Learning + LLM Service "
        "initialized successfully"
    )
except Exception as e:
    logger.error(
        f"[ERROR] Failed to initialize Enhanced Service: {e}"
    )
    enhanced_service = None

"""
Production Deep Learning ML Service for Django Integration
Completely replaces traditional ML with advanced deep learning models
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np

# Add the ml_models directory to the path
current_dir = Path(__file__).resolve().parent
ml_models_dir = current_dir.parent / 'ml_models'
sys.path.append(str(ml_models_dir))

logger = logging.getLogger(__name__)

class DeepLearningSpamDetector:
    """
    Production Deep Learning Spam Detection Service
    Handles all three types: Email, SMS, URL
    """
    
    def __init__(self):
        self.ensemble_detector = None
        self.models_loaded = False
        self.fallback_active = False
        
        # Initialize the service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the deep learning service"""
        try:
            logger.info("Initializing Deep Learning Spam Detection Service...")
            
            # Try to import and load the ensemble model
            try:
                from ensemble_model import EnsembleSpamDetector, EnsembleConfig
                
                # Create ensemble configuration
                ensemble_config = EnsembleConfig(
                    transformer_weight=0.5,
                    cnn_lstm_weight=0.3,
                    gnn_weight=0.2,
                    confidence_threshold=0.7,
                    use_dynamic_weighting=True
                )
                
                # Initialize ensemble detector
                self.ensemble_detector = EnsembleSpamDetector(ensemble_config)
                
                # Try to load pre-trained models
                models_dir = Path("deep_learning_models")
                if models_dir.exists():
                    try:
                        self.ensemble_detector.load_ensemble(str(models_dir))
                        self.models_loaded = True
                        logger.info(
                            "[SUCCESS] Pre-trained deep learning models "
                            "loaded successfully!"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not load pre-trained models: {e}"
                        )
                        self.models_loaded = False
                else:
                    logger.warning(
                        "No pre-trained models found. Using untrained models."
                    )
                    self.models_loaded = False
                
                logger.info(
                    "[BRAIN] Deep Learning Service initialized successfully"
                )
                
            except ImportError as e:
                logger.error(f"Failed to import ensemble model: {e}")
                self._initialize_fallback()
                
        except Exception as e:
            logger.error(f"Failed to initialize Deep Learning Service: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize a simple fallback detection system"""
        logger.info("Initializing fallback detection system...")
        self.fallback_active = True
        self.models_loaded = False
    
    def _fallback_prediction(self, text: str, prediction_type: str) -> Dict:
        """Simple rule-based fallback prediction"""
        # Simple keyword-based detection
        spam_keywords = [
            'free', 'win', 'prize', 'urgent', 'click', 'offer', 'deal',
            'limited', 'act now', 'congratulations', 'selected', 'bonus',
            'money', 'cash', 'credit', 'loan', 'debt', 'bank', 'account'
        ]
        
        suspicious_patterns = [
            'http://',  # Non-HTTPS URLs
            'bit.ly', 'tinyurl.com', 'goo.gl',  # URL shorteners
            '!!!', '$$', 'URGENT', 'FREE', 'WIN'  # Aggressive patterns
        ]
        
        text_lower = text.lower()
        
        # Count suspicious indicators
        keyword_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
        pattern_count = sum(1 for pattern in suspicious_patterns if pattern in text_lower)
        
        # Simple scoring
        suspicion_score = (keyword_count * 15) + (pattern_count * 25)
        
        # Determine result
        is_malicious = suspicion_score > 30
        confidence = min(85, max(55, suspicion_score + 40))
        
        if prediction_type == "email":
            result = "PHISHING" if is_malicious else "NOT PHISHING"
        elif prediction_type == "sms":
            result = "SPAM" if is_malicious else "NOT SPAM"
        else:  # URL
            result = "MALICIOUS" if is_malicious else "SAFE"
        
        return {
            'result': result,
            'confidence_score': confidence,
            'is_malicious': is_malicious,
            'fallback_used': True,
            'detailed_analysis': {
                'keyword_count': keyword_count,
                'pattern_count': pattern_count,
                'suspicion_score': suspicion_score
            }
        }
    
    def predict_email(self, email_text: str) -> Dict:
        """Predict if an email is phishing using deep learning"""
        try:
            if self.fallback_active or not self.models_loaded:
                return self._fallback_prediction(email_text, "email")
            
            # Use ensemble prediction
            results = self.ensemble_detector.ensemble_predict([email_text])
            
            prediction = results['predictions'][0]
            confidence = results['confidence_scores'][0]
            
            return {
                'is_phishing': bool(prediction),
                'confidence_score': confidence,
                'result': 'PHISHING' if prediction == 1 else 'NOT PHISHING',
                'model_predictions': {
                    'transformer': results.get('transformer_predictions', [None])[0],
                    'cnn_lstm': results.get('cnn_lstm_predictions', [None])[0],
                    'gnn': results.get('gnn_predictions', [None])[0]
                },
                'uncertainty': results.get('uncertainty_metrics', [0])[0]
            }
            
        except Exception as e:
            logger.error(f"Email prediction failed: {e}")
            return self._fallback_prediction(email_text, "email")
    
    def predict_sms(self, sms_text: str) -> Dict:
        """Predict if an SMS is spam using deep learning"""
        try:
            if self.fallback_active or not self.models_loaded:
                return self._fallback_prediction(sms_text, "sms")
            
            # Use ensemble prediction
            results = self.ensemble_detector.ensemble_predict([sms_text])
            
            prediction = results['predictions'][0]
            confidence = results['confidence_scores'][0]
            
            return {
                'is_spam': bool(prediction),
                'confidence_score': confidence,
                'result': 'SPAM' if prediction == 1 else 'NOT SPAM',
                'model_predictions': {
                    'transformer': results.get('transformer_predictions', [None])[0],
                    'cnn_lstm': results.get('cnn_lstm_predictions', [None])[0],
                    'gnn': results.get('gnn_predictions', [None])[0]
                },
                'uncertainty': results.get('uncertainty_metrics', [0])[0]
            }
            
        except Exception as e:
            logger.error(f"SMS prediction failed: {e}")
            return self._fallback_prediction(sms_text, "sms")
    
    def predict_url(self, url: str) -> Dict:
        """Predict if a URL is malicious using deep learning"""
        try:
            if self.fallback_active or not self.models_loaded:
                return self._fallback_prediction(url, "url")
            
            # Use ensemble prediction
            results = self.ensemble_detector.ensemble_predict([url])
            
            prediction = results['predictions'][0]
            confidence = results['confidence_scores'][0]
            
            return {
                'is_malicious': bool(prediction),
                'confidence_score': confidence,
                'result': 'MALICIOUS' if prediction == 1 else 'SAFE',
                'model_predictions': {
                    'transformer': results.get('transformer_predictions', [None])[0],
                    'cnn_lstm': results.get('cnn_lstm_predictions', [None])[0],
                    'gnn': results.get('gnn_predictions', [None])[0]
                },
                'uncertainty': results.get('uncertainty_metrics', [0])[0]
            }
            
        except Exception as e:
            logger.error(f"URL prediction failed: {e}")
            return self._fallback_prediction(url, "url")
    
    def get_service_status(self) -> Dict:
        """Get the current status of the deep learning service"""
        return {
            'service_type': 'Deep Learning',
            'models_loaded': self.models_loaded,
            'fallback_active': self.fallback_active,
            'available_models': ['Transformer', 'CNN-LSTM', 'GNN'] if self.models_loaded else ['Fallback'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def retrain_models(self):
        """Trigger model retraining (placeholder for future implementation)"""
        logger.info("Model retraining requested - this would trigger the training pipeline")
        return {"status": "Training request queued"}


# Global instance for Django integration
try:
    deep_learning_service = DeepLearningSpamDetector()
    logger.info(
        "[SUCCESS] Deep Learning Spam Detection Service instance "
        "created successfully"
    )
except Exception as e:
    logger.error(
        "[ERROR] Failed to create Deep Learning Service instance: %s", e
    )
    deep_learning_service = None

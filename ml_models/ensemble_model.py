"""
Ensemble Model for Advanced Multi-Modal Spam Detection
Combines Transformer, CNN-LSTM, and GNN models with confidence weighting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from .transformer_model import TransformerSpamDetector, TransformerConfig
    from .cnn_lstm_model import CNNLSTMSpamDetector, CNNLSTMConfig
    from .gnn_model import GNNSpamDetector, GNNConfig
except ImportError:
    try:
        from transformer_model import TransformerSpamDetector, TransformerConfig
        from cnn_lstm_model import CNNLSTMSpamDetector, CNNLSTMConfig
        from gnn_model import GNNSpamDetector, GNNConfig
    except ImportError:
        # Create dummy classes for fallback
        class TransformerSpamDetector:
            def __init__(self, config): pass
            def predict(self, texts): return {'predictions': [0]*len(texts), 'probabilities': [[0.5, 0.5]]*len(texts)}
        class CNNLSTMSpamDetector:
            def __init__(self, config): pass
            def predict(self, texts): return {'predictions': [0]*len(texts), 'probabilities': [[0.5, 0.5]]*len(texts)}
        class GNNSpamDetector:
            def __init__(self, config): pass
            def predict(self, texts): return {'predictions': [0]*len(texts), 'probabilities': [[0.5, 0.5]]*len(texts)}
        class TransformerConfig: pass
        class CNNLSTMConfig: pass
        class GNNConfig: pass

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model"""
    transformer_weight: float = 0.5
    cnn_lstm_weight: float = 0.3
    gnn_weight: float = 0.2
    confidence_threshold: float = 0.7
    uncertainty_samples: int = 100
    use_dynamic_weighting: bool = True
    performance_history_size: int = 1000


class UncertaintyQuantifier:
    """Advanced uncertainty quantification for ensemble predictions"""
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
    
    def monte_carlo_dropout(self, model, inputs: Dict, num_samples: int = None):
        """Perform Monte Carlo dropout for uncertainty estimation"""
        if num_samples is None:
            num_samples = self.num_samples
        
        model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                if hasattr(model, 'predict'):
                    pred = model.predict(inputs)
                    predictions.append(pred['probabilities'])
                else:
                    # For direct model inference
                    pred = model(**inputs)
                    probs = torch.softmax(pred, dim=-1)
                    predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=-1)
        
        return {
            'mean_prediction': mean_pred,
            'uncertainty': std_pred,
            'entropy': entropy,
            'confidence': 1 - entropy / np.log(mean_pred.shape[-1])
        }
    
    def ensemble_uncertainty(self, predictions: List[np.ndarray]) -> Dict:
        """Calculate uncertainty across ensemble predictions"""
        predictions = np.array(predictions)
        
        # Aleatoric uncertainty (within-model uncertainty)
        aleatoric = np.mean([np.var(pred, axis=0) for pred in predictions], axis=0)
        
        # Epistemic uncertainty (between-model uncertainty)
        mean_preds = np.mean(predictions, axis=0)
        epistemic = np.var([np.mean(pred, axis=0) for pred in predictions], axis=0)
        
        # Total uncertainty
        total = aleatoric + epistemic
        
        return {
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'total_uncertainty': total,
            'confidence': 1 - (total / (total + 1))
        }


class PerformanceTracker:
    """Track and adapt model performance for dynamic weighting"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.transformer_performance = []
        self.cnn_lstm_performance = []
        self.gnn_performance = []
        
    def update_performance(self, predictions: Dict, ground_truth: List[int]):
        """Update performance metrics for each model"""
        transformer_preds = predictions.get('transformer', {}).get('predictions', [])
        cnn_lstm_preds = predictions.get('cnn_lstm', {}).get('predictions', [])
        gnn_preds = predictions.get('gnn', {}).get('predictions', [])
        
        if len(transformer_preds) == len(ground_truth):
            accuracy = np.mean(np.array(transformer_preds) == np.array(ground_truth))
            self.transformer_performance.append(accuracy)
            
        if len(cnn_lstm_preds) == len(ground_truth):
            accuracy = np.mean(np.array(cnn_lstm_preds) == np.array(ground_truth))
            self.cnn_lstm_performance.append(accuracy)
            
        if len(gnn_preds) == len(ground_truth):
            accuracy = np.mean(np.array(gnn_preds) == np.array(ground_truth))
            self.gnn_performance.append(accuracy)
        
        # Keep only recent history
        self.transformer_performance = self.transformer_performance[-self.history_size:]
        self.cnn_lstm_performance = self.cnn_lstm_performance[-self.history_size:]
        self.gnn_performance = self.gnn_performance[-self.history_size:]
    
    def get_dynamic_weights(self) -> Tuple[float, float, float]:
        """Calculate dynamic weights based on recent performance"""
        if not all([self.transformer_performance, 
                   self.cnn_lstm_performance, 
                   self.gnn_performance]):
            return 0.5, 0.3, 0.2  # Default weights
        
        # Calculate recent performance
        transformer_perf = np.mean(self.transformer_performance[-100:])
        cnn_lstm_perf = np.mean(self.cnn_lstm_performance[-100:])
        gnn_perf = np.mean(self.gnn_performance[-100:])
        
        # Normalize to sum to 1
        total_perf = transformer_perf + cnn_lstm_perf + gnn_perf
        
        if total_perf > 0:
            return (transformer_perf / total_perf, 
                   cnn_lstm_perf / total_perf, 
                   gnn_perf / total_perf)
        else:
            return 0.5, 0.3, 0.2


class EnsembleSpamDetector:
    """
    Advanced ensemble spam detector combining three deep learning models
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize individual models
        self.transformer_model = TransformerSpamDetector(TransformerConfig())
        self.cnn_lstm_model = CNNLSTMSpamDetector(CNNLSTMConfig())
        self.gnn_model = GNNSpamDetector(GNNConfig())
        
        # Uncertainty quantification
        self.uncertainty_quantifier = UncertaintyQuantifier(config.uncertainty_samples)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(config.performance_history_size)
        
        # Model states
        self.models_trained = {
            'transformer': False,
            'cnn_lstm': False,
            'gnn': False
        }
        
        logger.info("Ensemble spam detector initialized")
    
    def train_all_models(self, train_texts: List[str], train_labels: List[int],
                        val_texts: List[str], val_labels: List[int]):
        """Train all three models in the ensemble"""
        logger.info("Starting ensemble training...")
        
        # Train Transformer model
        logger.info("Training Transformer model...")
        try:
            self.transformer_model.train(train_texts, train_labels, val_texts, val_labels)
            self.models_trained['transformer'] = True
            logger.info("Transformer model training completed")
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
        
        # Train CNN-LSTM model
        logger.info("Training CNN-LSTM model...")
        try:
            self.cnn_lstm_model.train(train_texts, train_labels, val_texts, val_labels)
            self.models_trained['cnn_lstm'] = True
            logger.info("CNN-LSTM model training completed")
        except Exception as e:
            logger.error(f"CNN-LSTM training failed: {e}")
        
        # Train GNN model
        logger.info("Training GNN model...")
        try:
            self.gnn_model.train(train_texts, train_labels, val_texts, val_labels)
            self.models_trained['gnn'] = True
            logger.info("GNN model training completed")
        except Exception as e:
            logger.error(f"GNN training failed: {e}")
        
        logger.info("Ensemble training completed")
    
    def predict_individual_models(self, texts: List[str]) -> Dict:
        """Get predictions from all individual models"""
        predictions = {}
        
        # Transformer predictions
        if self.models_trained['transformer']:
            try:
                transformer_pred = self.transformer_model.predict(texts)
                predictions['transformer'] = transformer_pred
            except Exception as e:
                logger.error(f"Transformer prediction failed: {e}")
                predictions['transformer'] = None
        
        # CNN-LSTM predictions
        if self.models_trained['cnn_lstm']:
            try:
                cnn_lstm_pred = self.cnn_lstm_model.predict(texts)
                predictions['cnn_lstm'] = cnn_lstm_pred
            except Exception as e:
                logger.error(f"CNN-LSTM prediction failed: {e}")
                predictions['cnn_lstm'] = None
        
        # GNN predictions
        if self.models_trained['gnn']:
            try:
                gnn_pred = self.gnn_model.predict(texts)
                predictions['gnn'] = gnn_pred
            except Exception as e:
                logger.error(f"GNN prediction failed: {e}")
                predictions['gnn'] = None
        
        return predictions
    
    def ensemble_predict(self, texts: List[str]) -> Dict:
        """
        Make ensemble predictions with confidence scoring and uncertainty quantification
        """
        # Get individual model predictions
        individual_preds = self.predict_individual_models(texts)
        
        # Get dynamic weights if enabled
        if self.config.use_dynamic_weighting:
            weights = self.performance_tracker.get_dynamic_weights()
        else:
            weights = (self.config.transformer_weight, 
                      self.config.cnn_lstm_weight, 
                      self.config.gnn_weight)
        
        # Ensemble predictions
        ensemble_predictions = []
        ensemble_probabilities = []
        ensemble_confidence_scores = []
        uncertainty_metrics = []
        
        for i in range(len(texts)):
            text_predictions = []
            text_probabilities = []
            text_confidences = []
            
            # Collect predictions from available models
            if individual_preds.get('transformer') is not None:
                pred = individual_preds['transformer']
                text_predictions.append(pred['predictions'][i])
                text_probabilities.append(pred['probabilities'][i])
                text_confidences.append(pred['confidence_scores'][i])
            
            if individual_preds.get('cnn_lstm') is not None:
                pred = individual_preds['cnn_lstm']
                text_predictions.append(pred['predictions'][i])
                text_probabilities.append(pred['probabilities'][i])
                text_confidences.append(pred['confidence_scores'][i])
            
            if individual_preds.get('gnn') is not None:
                pred = individual_preds['gnn']
                text_predictions.append(pred['predictions'][i])
                text_probabilities.append(pred['probabilities'][i])
                text_confidences.append(pred['confidence_scores'][i])
            
            if not text_predictions:
                # No models available
                ensemble_predictions.append(0)
                ensemble_probabilities.append([0.5, 0.5])
                ensemble_confidence_scores.append(0.0)
                uncertainty_metrics.append({'total_uncertainty': 1.0})
                continue
            
            # Weighted ensemble
            weighted_probs = np.zeros(2)
            total_weight = 0
            
            for j, (prob, weight) in enumerate(zip(text_probabilities, weights[:len(text_probabilities)])):
                weighted_probs += np.array(prob) * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_probs /= total_weight
            
            # Final prediction
            final_prediction = np.argmax(weighted_probs)
            final_confidence = np.max(weighted_probs) * 100
            
            # Calculate uncertainty
            uncertainty = self.uncertainty_quantifier.ensemble_uncertainty(
                [np.array(text_probabilities)]
            )
            
            ensemble_predictions.append(final_prediction)
            ensemble_probabilities.append(weighted_probs.tolist())
            ensemble_confidence_scores.append(final_confidence)
            uncertainty_metrics.append(uncertainty)
        
        return {
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probabilities,
            'confidence_scores': ensemble_confidence_scores,
            'uncertainty_metrics': uncertainty_metrics,
            'individual_predictions': individual_preds,
            'ensemble_weights': weights,
            'labels': ['HAM', 'SPAM']
        }
    
    def generate_detailed_report(self, text: str, prediction_result: Dict, 
                               index: int = 0) -> Dict:
        """
        Generate a comprehensive analysis report for a single prediction
        """
        # Extract prediction details
        prediction = prediction_result['predictions'][index]
        probability = prediction_result['probabilities'][index]
        confidence = prediction_result['confidence_scores'][index]
        uncertainty = prediction_result['uncertainty_metrics'][index]
        
        # Analyze text patterns
        detected_patterns = self._analyze_text_patterns(text)
        
        # Risk assessment
        risk_level = self._assess_risk_level(confidence, uncertainty, detected_patterns)
        
        # Generate explanation
        explanation = self._generate_explanation(
            text, prediction, confidence, detected_patterns
        )
        
        # Recommended action
        action = self._recommend_action(prediction, confidence, risk_level)
        
        report = {
            'input_analysis': {
                'content_type': self._detect_content_type(text),
                'content_length': f"{len(text)} characters / {len(text.split())} words",
                'timestamp': self._get_timestamp()
            },
            'deep_learning_results': {
                'transformer_model': self._get_model_result(
                    prediction_result['individual_predictions'].get('transformer'), index
                ),
                'cnn_lstm_model': self._get_model_result(
                    prediction_result['individual_predictions'].get('cnn_lstm'), index
                ),
                'gnn_model': self._get_model_result(
                    prediction_result['individual_predictions'].get('gnn'), index
                )
            },
            'ensemble_prediction': {
                'result': prediction_result['labels'][prediction],
                'confidence': f"{confidence:.1f}%",
                'probability_distribution': {
                    'HAM': f"{probability[0]*100:.1f}%",
                    'SPAM': f"{probability[1]*100:.1f}%"
                }
            },
            'detected_patterns': detected_patterns,
            'uncertainty_analysis': {
                'epistemic_uncertainty': uncertainty.get('epistemic_uncertainty', 0),
                'aleatoric_uncertainty': uncertainty.get('aleatoric_uncertainty', 0),
                'total_uncertainty': uncertainty.get('total_uncertainty', 0),
                'model_confidence': uncertainty.get('confidence', 0)
            },
            'final_verdict': prediction_result['labels'][prediction],
            'risk_level': risk_level,
            'confidence_score': f"{confidence:.1f}%",
            'recommended_action': action,
            'explanation': explanation
        }
        
        return report
    
    def _analyze_text_patterns(self, text: str) -> Dict:
        """Analyze text for suspicious patterns"""
        patterns = {
            'suspicious_keywords': [],
            'malicious_urls': [],
            'suspicious_domains': [],
            'phone_numbers': [],
            'email_addresses': [],
            'financial_indicators': [],
            'urgency_indicators': []
        }
        
        import re
        from urllib.parse import urlparse
        
        # Suspicious keywords
        spam_keywords = [
            'free', 'win', 'prize', 'urgent', 'click', 'offer', 'deal',
            'limited time', 'act now', 'congratulations', 'selected',
            'bonus', 'cash', 'money', 'earn', 'guaranteed'
        ]
        
        for keyword in spam_keywords:
            if keyword.lower() in text.lower():
                patterns['suspicious_keywords'].append(keyword)
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        for url in urls:
            try:
                domain = urlparse(url).netloc
                # Check for suspicious domains
                if any(susp in domain.lower() for susp in ['secure', 'update', 'verify', 'account']):
                    patterns['malicious_urls'].append(url)
                    patterns['suspicious_domains'].append(domain)
            except:
                pass
        
        # Financial indicators
        financial_terms = ['$', '€', '£', 'money', 'cash', 'bank', 'account', 'credit', 'loan']
        for term in financial_terms:
            if term.lower() in text.lower():
                patterns['financial_indicators'].append(term)
        
        # Urgency indicators
        urgency_terms = ['urgent', 'immediate', 'act now', 'limited time', 'expires', 'deadline']
        for term in urgency_terms:
            if term.lower() in text.lower():
                patterns['urgency_indicators'].append(term)
        
        return patterns
    
    def _assess_risk_level(self, confidence: float, uncertainty: Dict, 
                          patterns: Dict) -> str:
        """Assess overall risk level"""
        risk_score = 0
        
        # Confidence contribution
        if confidence < 60:
            risk_score += 1
        elif confidence < 80:
            risk_score += 0.5
        
        # Uncertainty contribution
        total_uncertainty = uncertainty.get('total_uncertainty', 0)
        if total_uncertainty > 0.5:
            risk_score += 1
        elif total_uncertainty > 0.3:
            risk_score += 0.5
        
        # Pattern contribution
        pattern_count = sum(len(v) for v in patterns.values() if isinstance(v, list))
        if pattern_count > 5:
            risk_score += 2
        elif pattern_count > 2:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 3:
            return "High"
        elif risk_score >= 1.5:
            return "Medium"
        else:
            return "Low"
    
    def _generate_explanation(self, text: str, prediction: int, 
                             confidence: float, patterns: Dict) -> str:
        """Generate human-readable explanation"""
        label = "SPAM" if prediction == 1 else "HAM"
        
        explanation = f"The ensemble model classified this content as {label} "
        explanation += f"with {confidence:.1f}% confidence. "
        
        if patterns['suspicious_keywords']:
            explanation += f"Detected suspicious keywords: {', '.join(patterns['suspicious_keywords'][:3])}. "
        
        if patterns['malicious_urls']:
            explanation += f"Found {len(patterns['malicious_urls'])} potentially malicious URLs. "
        
        if patterns['financial_indicators']:
            explanation += "Content contains financial terminology which is common in spam. "
        
        if patterns['urgency_indicators']:
            explanation += "Urgency indicators suggest potential manipulation tactics. "
        
        return explanation
    
    def _recommend_action(self, prediction: int, confidence: float, 
                         risk_level: str) -> str:
        """Recommend action based on prediction and confidence"""
        if prediction == 1:  # SPAM
            if confidence > 90 and risk_level == "High":
                return "Block immediately"
            elif confidence > 70:
                return "Block and review"
            else:
                return "Flag for manual review"
        else:  # HAM
            if confidence > 90 and risk_level == "Low":
                return "Allow"
            elif confidence > 70:
                return "Allow with monitoring"
            else:
                return "Review before allowing"
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content"""
        if '@' in text and '.' in text:
            return "Email"
        elif 'http' in text:
            return "URL/Web Content"
        elif len(text) < 160:
            return "SMS"
        else:
            return "Text Content"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_model_result(self, model_pred: Optional[Dict], index: int) -> str:
        """Get formatted result for individual model"""
        if model_pred is None:
            return "Model not available"
        
        pred = model_pred['predictions'][index]
        conf = model_pred['confidence_scores'][index]
        label = model_pred['labels'][pred]
        
        return f"{label} (Confidence: {conf:.1f}%)"
    
    def save_ensemble(self, path: str):
        """Save the entire ensemble"""
        ensemble_data = {
            'config': self.config,
            'models_trained': self.models_trained,
            'performance_history': {
                'transformer': self.performance_tracker.transformer_performance,
                'cnn_lstm': self.performance_tracker.cnn_lstm_performance,
                'gnn': self.performance_tracker.gnn_performance
            }
        }
        
        # Save individual models
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        if self.models_trained['transformer']:
            self.transformer_model.save_model(str(base_path / "transformer.pth"))
        
        if self.models_trained['cnn_lstm']:
            self.cnn_lstm_model.save_model(str(base_path / "cnn_lstm.pth"))
        
        if self.models_trained['gnn']:
            self.gnn_model.save_model(str(base_path / "gnn.pth"))
        
        # Save ensemble configuration
        with open(base_path / "ensemble_config.json", 'w') as f:
            json.dump(ensemble_data, f, indent=2, default=str)
        
        logger.info(f"Ensemble saved to {path}")
    
    def load_ensemble(self, path: str):
        """Load the entire ensemble"""
        base_path = Path(path)
        
        # Load ensemble configuration
        with open(base_path / "ensemble_config.json", 'r') as f:
            ensemble_data = json.load(f)
        
        self.config = ensemble_data['config']
        self.models_trained = ensemble_data['models_trained']
        
        # Restore performance history
        perf_history = ensemble_data.get('performance_history', {})
        self.performance_tracker.transformer_performance = perf_history.get('transformer', [])
        self.performance_tracker.cnn_lstm_performance = perf_history.get('cnn_lstm', [])
        self.performance_tracker.gnn_performance = perf_history.get('gnn', [])
        
        # Load individual models
        if self.models_trained['transformer'] and (base_path / "transformer.pth").exists():
            self.transformer_model.load_model(str(base_path / "transformer.pth"))
        
        if self.models_trained['cnn_lstm'] and (base_path / "cnn_lstm.pth").exists():
            self.cnn_lstm_model.load_model(str(base_path / "cnn_lstm.pth"))
        
        if self.models_trained['gnn'] and (base_path / "gnn.pth").exists():
            self.gnn_model.load_model(str(base_path / "gnn.pth"))
        
        logger.info(f"Ensemble loaded from {path}")

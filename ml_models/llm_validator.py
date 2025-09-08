"""
LLM Validation Component for Dual-Stage Spam Detection
Provides secondary validation using Large Language Models
"""

import asyncio
import aiohttp
import openai
import anthropic
from typing import Dict, List, Optional, Any
import logging
import json
from dataclasses import dataclass
import time

try:
    from .ensemble_model import EnsembleSpamDetector
except ImportError:
    pass  # Will import when needed

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM validation"""
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    confidence_threshold: float = 0.6
    timeout: int = 30
    max_retries: int = 3


class LLMValidator:
    """
    Large Language Model validator for spam detection
    Provides contextual analysis and reasoning for secondary validation
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize clients based on provider
        if config.provider == "openai" and config.api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=config.api_key)
        elif config.provider == "anthropic" and config.api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=config.api_key)
        
        # Validation prompts
        self.system_prompt = self._get_system_prompt()
        self.validation_prompt_template = self._get_validation_prompt_template()
        
        logger.info(f"LLM Validator initialized with {config.provider}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM validation"""
        return """You are an expert cybersecurity analyst specializing in spam and phishing detection. 
Your role is to provide secondary validation of content that has been preliminarily classified as non-malicious by deep learning models.

Your task is to:
1. Analyze the content for subtle signs of spam, phishing, or malicious intent
2. Consider context, tone, and sophisticated manipulation techniques
3. Provide reasoning for your assessment
4. Give a confidence score (0-100) for your classification
5. Suggest specific indicators that support your conclusion

Be thorough but efficient. Focus on patterns that automated systems might miss, such as:
- Social engineering tactics
- Sophisticated phishing attempts
- Context-dependent threats
- Cultural or linguistic manipulation
- Advanced persuasion techniques"""
    
    def _get_validation_prompt_template(self) -> str:
        """Get the validation prompt template"""
        return """Please analyze the following content for spam/phishing/malicious intent:

CONTENT TYPE: {content_type}
CONTENT: {content}

PRELIMINARY ANALYSIS:
- Deep Learning Prediction: {dl_prediction}
- Confidence: {dl_confidence}%
- Detected Patterns: {detected_patterns}

Please provide your analysis in the following JSON format:
{
    "classification": "SPAM" or "HAM",
    "confidence": <0-100>,
    "reasoning": "<detailed explanation>",
    "risk_factors": ["<list of specific risk factors found>"],
    "manipulation_techniques": ["<list of manipulation techniques identified>"],
    "final_verdict": "CONFIRMED_HAM" or "FLAGGED_FOR_REVIEW" or "RECLASSIFIED_AS_SPAM",
    "recommended_action": "<specific action recommendation>"
}

Be thorough in your analysis and provide clear reasoning for your assessment."""
    
    async def validate_content(self, content: str, content_type: str, 
                             dl_prediction: str, dl_confidence: float,
                             detected_patterns: Dict) -> Dict:
        """
        Validate content using LLM
        
        Args:
            content: The content to validate
            content_type: Type of content (Email, SMS, URL, etc.)
            dl_prediction: Deep learning model prediction
            dl_confidence: Deep learning model confidence
            detected_patterns: Patterns detected by DL models
            
        Returns:
            Dictionary with LLM validation results
        """
        try:
            # Prepare the prompt
            prompt = self.validation_prompt_template.format(
                content_type=content_type,
                content=content,
                dl_prediction=dl_prediction,
                dl_confidence=dl_confidence,
                detected_patterns=json.dumps(detected_patterns, indent=2)
            )
            
            # Get LLM response
            if self.config.provider == "openai":
                response = await self._validate_with_openai(prompt)
            elif self.config.provider == "anthropic":
                response = await self._validate_with_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
            
            # Parse and validate response
            parsed_response = self._parse_llm_response(response)
            
            # Add metadata
            parsed_response['llm_provider'] = self.config.provider
            parsed_response['llm_model'] = self.config.model
            parsed_response['validation_timestamp'] = time.time()
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return self._get_fallback_response(dl_prediction, dl_confidence)
    
    async def _validate_with_openai(self, prompt: str) -> str:
        """Validate content using OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _validate_with_anthropic(self, prompt: str) -> str:
        """Validate content using Anthropic API"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            response = await self.anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = [
                    'classification', 'confidence', 'reasoning',
                    'final_verdict', 'recommended_action'
                ]
                
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")
                
                # Ensure confidence is in valid range
                parsed['confidence'] = max(0, min(100, parsed['confidence']))
                
                return parsed
            
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return a basic parsed response
            return {
                'classification': 'HAM',
                'confidence': 50,
                'reasoning': f"Failed to parse LLM response: {str(e)}",
                'risk_factors': [],
                'manipulation_techniques': [],
                'final_verdict': 'FLAGGED_FOR_REVIEW',
                'recommended_action': 'Manual review required due to parsing error'
            }
    
    def _get_fallback_response(self, dl_prediction: str, dl_confidence: float) -> Dict:
        """Get fallback response when LLM validation fails"""
        return {
            'classification': dl_prediction,
            'confidence': max(0, dl_confidence - 20),  # Reduce confidence due to failure
            'reasoning': 'LLM validation failed, using deep learning prediction only',
            'risk_factors': ['LLM validation unavailable'],
            'manipulation_techniques': [],
            'final_verdict': 'FLAGGED_FOR_REVIEW',
            'recommended_action': 'Manual review recommended due to LLM validation failure',
            'llm_provider': 'fallback',
            'llm_model': 'none',
            'validation_timestamp': time.time()
        }
    
    async def batch_validate(self, contents: List[str], content_types: List[str],
                           dl_predictions: List[str], dl_confidences: List[float],
                           detected_patterns_list: List[Dict]) -> List[Dict]:
        """
        Validate multiple contents in batch
        """
        tasks = []
        
        for i, content in enumerate(contents):
            task = self.validate_content(
                content,
                content_types[i],
                dl_predictions[i],
                dl_confidences[i],
                detected_patterns_list[i]
            )
            tasks.append(task)
        
        # Execute validation tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch validation failed for item {i}: {result}")
                fallback = self._get_fallback_response(
                    dl_predictions[i], dl_confidences[i]
                )
                processed_results.append(fallback)
            else:
                processed_results.append(result)
        
        return processed_results


class DualStageDetector:
    """
    Dual-stage spam detection system combining deep learning and LLM validation
    """
    
    def __init__(self, ensemble_detector, llm_validator: LLMValidator):
        self.ensemble_detector = ensemble_detector
        self.llm_validator = llm_validator
        self.validation_stats = {
            'total_predictions': 0,
            'llm_validations': 0,
            'reclassifications': 0,
            'confirmed_predictions': 0
        }
        
        logger.info("Dual-stage detector initialized")
    
    async def predict(self, texts: List[str]) -> Dict:
        """
        Perform dual-stage prediction with LLM validation
        
        Stage 1: Deep Learning Ensemble Prediction
        Stage 2: LLM Validation (for non-malicious content)
        """
        # Stage 1: Deep Learning Prediction
        logger.info("Stage 1: Deep Learning Ensemble Prediction")
        dl_results = self.ensemble_detector.ensemble_predict(texts)
        
        # Identify items that need LLM validation
        items_for_llm_validation = []
        validation_indices = []
        
        for i, (pred, conf) in enumerate(zip(dl_results['predictions'], 
                                           dl_results['confidence_scores'])):
            # Validate HAM predictions with low-medium confidence
            # Or any prediction with very low confidence
            if (pred == 0 and conf < 85) or conf < 60:
                items_for_llm_validation.append({
                    'index': i,
                    'text': texts[i],
                    'prediction': 'HAM' if pred == 0 else 'SPAM',
                    'confidence': conf
                })
                validation_indices.append(i)
        
        # Stage 2: LLM Validation
        llm_results = []
        if items_for_llm_validation:
            logger.info(f"Stage 2: LLM Validation for {len(items_for_llm_validation)} items")
            
            # Prepare data for LLM validation
            val_texts = [item['text'] for item in items_for_llm_validation]
            val_types = [self._detect_content_type(text) for text in val_texts]
            val_predictions = [item['prediction'] for item in items_for_llm_validation]
            val_confidences = [item['confidence'] for item in items_for_llm_validation]
            
            # Get detected patterns for each text
            val_patterns = []
            for i, item in enumerate(items_for_llm_validation):
                idx = item['index']
                patterns = self._extract_patterns_from_dl_result(dl_results, idx)
                val_patterns.append(patterns)
            
            # Perform LLM validation
            llm_results = await self.llm_validator.batch_validate(
                val_texts, val_types, val_predictions, val_confidences, val_patterns
            )
        
        # Combine results
        final_results = self._combine_stage_results(
            dl_results, llm_results, validation_indices
        )
        
        # Update statistics
        self._update_stats(dl_results, llm_results)
        
        return final_results
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type for LLM validation"""
        if '@' in text and '.' in text:
            return "Email"
        elif 'http' in text.lower():
            return "URL/Web Content"
        elif len(text) < 160:
            return "SMS"
        else:
            return "Text Content"
    
    def _extract_patterns_from_dl_result(self, dl_results: Dict, index: int) -> Dict:
        """Extract detected patterns from deep learning results"""
        # This would extract patterns from the ensemble prediction
        # For now, return a basic structure
        return {
            'confidence': dl_results['confidence_scores'][index],
            'uncertainty': dl_results['uncertainty_metrics'][index],
            'model_weights': dl_results['ensemble_weights']
        }
    
    def _combine_stage_results(self, dl_results: Dict, llm_results: List[Dict],
                              validation_indices: List[int]) -> Dict:
        """Combine results from both stages"""
        final_predictions = dl_results['predictions'].copy()
        final_confidences = dl_results['confidence_scores'].copy()
        final_probabilities = dl_results['probabilities'].copy()
        
        llm_validations = {}
        
        # Apply LLM validation results
        for i, (val_idx, llm_result) in enumerate(zip(validation_indices, llm_results)):
            llm_validations[val_idx] = llm_result
            
            # Check if LLM reclassified the result
            if llm_result['final_verdict'] == 'RECLASSIFIED_AS_SPAM':
                final_predictions[val_idx] = 1  # SPAM
                final_confidences[val_idx] = llm_result['confidence']
                final_probabilities[val_idx] = [
                    1 - llm_result['confidence']/100,
                    llm_result['confidence']/100
                ]
            elif llm_result['final_verdict'] == 'CONFIRMED_HAM':
                # Keep original prediction but update confidence
                final_confidences[val_idx] = min(
                    final_confidences[val_idx] + 10,  # Boost confidence
                    95
                )
        
        # Create comprehensive result
        result = {
            'predictions': final_predictions,
            'probabilities': final_probabilities,
            'confidence_scores': final_confidences,
            'uncertainty_metrics': dl_results['uncertainty_metrics'],
            'ensemble_weights': dl_results['ensemble_weights'],
            'labels': dl_results['labels'],
            'stage_1_results': dl_results,
            'stage_2_results': llm_validations,
            'dual_stage_metadata': {
                'total_items': len(final_predictions),
                'llm_validated_items': len(validation_indices),
                'reclassified_items': sum(
                    1 for result in llm_results 
                    if result['final_verdict'] == 'RECLASSIFIED_AS_SPAM'
                )
            }
        }
        
        return result
    
    def _update_stats(self, dl_results: Dict, llm_results: List[Dict]):
        """Update validation statistics"""
        self.validation_stats['total_predictions'] += len(dl_results['predictions'])
        self.validation_stats['llm_validations'] += len(llm_results)
        
        for result in llm_results:
            if result['final_verdict'] == 'RECLASSIFIED_AS_SPAM':
                self.validation_stats['reclassifications'] += 1
            elif result['final_verdict'] == 'CONFIRMED_HAM':
                self.validation_stats['confirmed_predictions'] += 1
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats['total_predictions'] > 0:
            stats['llm_validation_rate'] = (
                stats['llm_validations'] / stats['total_predictions'] * 100
            )
            stats['reclassification_rate'] = (
                stats['reclassifications'] / stats['llm_validations'] * 100
                if stats['llm_validations'] > 0 else 0
            )
        
        return stats
    
    async def generate_comprehensive_report(self, text: str, 
                                          prediction_result: Dict,
                                          index: int = 0) -> Dict:
        """
        Generate a comprehensive dual-stage analysis report
        """
        # Get ensemble report
        ensemble_report = self.ensemble_detector.generate_detailed_report(
            text, prediction_result['stage_1_results'], index
        )
        
        # Add LLM validation results if available
        llm_validation = prediction_result['stage_2_results'].get(index)
        
        if llm_validation:
            ensemble_report['llm_validation'] = {
                'secondary_analysis': llm_validation['classification'],
                'contextual_reasoning': llm_validation['reasoning'],
                'final_verification': llm_validation['final_verdict'],
                'risk_factors': llm_validation.get('risk_factors', []),
                'manipulation_techniques': llm_validation.get('manipulation_techniques', []),
                'llm_confidence': f"{llm_validation['confidence']:.1f}%"
            }
        else:
            ensemble_report['llm_validation'] = {
                'secondary_analysis': 'Not performed',
                'contextual_reasoning': 'LLM validation criteria not met',
                'final_verification': 'Deep learning prediction confirmed',
                'risk_factors': [],
                'manipulation_techniques': [],
                'llm_confidence': 'N/A'
            }
        
        # Update final verdict based on dual-stage analysis
        final_pred = prediction_result['predictions'][index]
        final_conf = prediction_result['confidence_scores'][index]
        
        ensemble_report['final_verdict'] = prediction_result['labels'][final_pred]
        ensemble_report['confidence_score'] = f"{final_conf:.1f}%"
        
        # Add dual-stage metadata
        ensemble_report['dual_stage_analysis'] = {
            'stage_1_prediction': prediction_result['stage_1_results']['labels'][
                prediction_result['stage_1_results']['predictions'][index]
            ],
            'stage_2_validation': 'Performed' if llm_validation else 'Not required',
            'final_decision': 'LLM Override' if llm_validation and 
                           llm_validation['final_verdict'] == 'RECLASSIFIED_AS_SPAM'
                           else 'Ensemble Confirmed'
        }
        
        return ensemble_report

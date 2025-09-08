"""
Advanced Data Processing and Training Pipeline
Handles large-scale dataset processing and model training coordination
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from pathlib import Path
import yaml
import asyncio
import os

# Import our deep learning models
from .ensemble_model import EnsembleSpamDetector, EnsembleConfig
from .llm_validator import LLMValidator, LLMConfig, DualStageDetector

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Advanced data processor for large-scale spam detection datasets
    """
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.processed_data = {}
        self.label_encoders = {}
        
    def load_email_dataset(self) -> Tuple[List[str], List[int]]:
        """Load and process the Enron email dataset"""
        logger.info("Loading email dataset...")
        
        try:
            # Read the large emails.csv file in chunks
            emails_path = self.data_dir / "emails.csv"
            
            if not emails_path.exists():
                logger.error(f"Email dataset not found at {emails_path}")
                return [], []
            
            # Read in chunks to handle large file
            chunk_size = 10000
            email_texts = []
            email_labels = []
            
            for chunk in pd.read_csv(emails_path, chunksize=chunk_size):
                # Assuming the CSV has columns like 'text' and 'label'
                # Adjust column names based on actual dataset structure
                if 'text' in chunk.columns and 'label' in chunk.columns:
                    texts = chunk['text'].astype(str).tolist()
                    labels = chunk['label'].tolist()
                elif len(chunk.columns) >= 2:
                    # Use first two columns if names are different
                    texts = chunk.iloc[:, 1].astype(str).tolist()  # Second column as text
                    labels = chunk.iloc[:, 0].tolist()  # First column as label
                else:
                    logger.warning("Unexpected CSV structure in email dataset")
                    continue
                
                email_texts.extend(texts)
                email_labels.extend(labels)
                
                logger.info(f"Processed {len(email_texts)} emails so far...")
                
                # Limit dataset size for training efficiency
                if len(email_texts) >= 100000:
                    break
            
            # Encode labels to binary (0: ham, 1: spam/phishing)
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(email_labels)
            self.label_encoders['email'] = label_encoder
            
            logger.info(f"Loaded {len(email_texts)} emails with {len(set(email_labels))} unique labels")
            
            return email_texts, encoded_labels.tolist()
            
        except Exception as e:
            logger.error(f"Failed to load email dataset: {e}")
            return [], []
    
    def load_sms_dataset(self) -> Tuple[List[str], List[int]]:
        """Load and process the SMS spam dataset"""
        logger.info("Loading SMS dataset...")
        
        try:
            sms_path = self.data_dir / "spam.csv"
            
            if not sms_path.exists():
                logger.error(f"SMS dataset not found at {sms_path}")
                return [], []
            
            df = pd.read_csv(sms_path, encoding='latin-1')
            
            # Clean the dataset
            df = df.dropna()
            
            # Extract text and labels
            # Assuming first column is label (ham/spam) and second is text
            if len(df.columns) >= 2:
                texts = df.iloc[:, 1].astype(str).tolist()
                labels = df.iloc[:, 0].tolist()
            else:
                logger.error("Unexpected SMS dataset structure")
                return [], []
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            self.label_encoders['sms'] = label_encoder
            
            logger.info(f"Loaded {len(texts)} SMS messages")
            
            return texts, encoded_labels.tolist()
            
        except Exception as e:
            logger.error(f"Failed to load SMS dataset: {e}")
            return [], []
    
    def load_url_dataset(self) -> Tuple[List[str], List[int]]:
        """Load and process the phishing URL dataset"""
        logger.info("Loading URL dataset...")
        
        try:
            url_dir = self.data_dir / "Phishing URL dataset"
            
            # Try to load from different possible files
            url_files = [
                url_dir / "Phishing URLs.csv",
                url_dir / "URL dataset.csv"
            ]
            
            all_urls = []
            all_labels = []
            
            for url_file in url_files:
                if url_file.exists():
                    logger.info(f"Loading {url_file}")
                    
                    # Read in chunks for large files
                    chunk_size = 5000
                    for chunk in pd.read_csv(url_file, chunksize=chunk_size):
                        if 'url' in chunk.columns:
                            urls = chunk['url'].astype(str).tolist()
                            
                            # Determine labels based on file name or column
                            if 'label' in chunk.columns:
                                labels = chunk['label'].tolist()
                            elif 'Phishing' in str(url_file):
                                labels = [1] * len(urls)  # Phishing URLs
                            else:
                                labels = [0] * len(urls)  # Legitimate URLs
                        
                        elif len(chunk.columns) >= 1:
                            urls = chunk.iloc[:, 0].astype(str).tolist()
                            # Default to phishing if from phishing file
                            if 'Phishing' in str(url_file):
                                labels = [1] * len(urls)
                            else:
                                labels = [0] * len(urls)
                        else:
                            continue
                        
                        all_urls.extend(urls)
                        all_labels.extend(labels)
                        
                        # Limit dataset size
                        if len(all_urls) >= 50000:
                            break
                    
                    if len(all_urls) >= 50000:
                        break
            
            if not all_urls:
                logger.error("No URLs loaded from dataset")
                return [], []
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(all_labels)
            self.label_encoders['url'] = label_encoder
            
            logger.info(f"Loaded {len(all_urls)} URLs")
            
            return all_urls, encoded_labels.tolist()
            
        except Exception as e:
            logger.error(f"Failed to load URL dataset: {e}")
            return [], []
    
    def combine_datasets(self) -> Tuple[List[str], List[int], List[str]]:
        """Combine all datasets into a unified training set"""
        logger.info("Combining datasets...")
        
        # Load individual datasets
        email_texts, email_labels = self.load_email_dataset()
        sms_texts, sms_labels = self.load_sms_dataset()
        url_texts, url_labels = self.load_url_dataset()
        
        # Combine all data
        all_texts = []
        all_labels = []
        all_types = []
        
        # Add emails
        all_texts.extend(email_texts)
        all_labels.extend(email_labels)
        all_types.extend(['email'] * len(email_texts))
        
        # Add SMS
        all_texts.extend(sms_texts)
        all_labels.extend(sms_labels)
        all_types.extend(['sms'] * len(sms_texts))
        
        # Add URLs
        all_texts.extend(url_texts)
        all_labels.extend(url_labels)
        all_types.extend(['url'] * len(url_texts))
        
        logger.info(f"Combined dataset: {len(all_texts)} samples")
        logger.info(f"Email samples: {len(email_texts)}")
        logger.info(f"SMS samples: {len(sms_texts)}")
        logger.info(f"URL samples: {len(url_texts)}")
        
        return all_texts, all_labels, all_types
    
    def create_train_val_test_split(self, texts: List[str], labels: List[int], 
                                   types: List[str], test_size: float = 0.2,
                                   val_size: float = 0.1) -> Dict:
        """Create stratified train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, types_temp, types_test = train_test_split(
            texts, labels, types, test_size=test_size, 
            stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, types_train, types_val = train_test_split(
            X_temp, y_temp, types_temp, test_size=val_ratio,
            stratify=y_temp, random_state=42
        )
        
        splits = {
            'train': {
                'texts': X_train,
                'labels': y_train,
                'types': types_train
            },
            'val': {
                'texts': X_val,
                'labels': y_val,
                'types': types_val
            },
            'test': {
                'texts': X_test,
                'labels': y_test,
                'types': types_test
            }
        }
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return splits


class AdvancedTrainer:
    """
    Advanced trainer for the deep learning ensemble
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.data_processor = DataProcessor()
        self.ensemble_detector = None
        self.llm_validator = None
        self.dual_stage_detector = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'model_config': {
                'transformer': {
                    'model_name': 'bert-base-uncased',
                    'max_length': 512,
                    'batch_size': 16,
                    'num_epochs': 3
                },
                'cnn_lstm': {
                    'vocab_size': 50000,
                    'embedding_dim': 300,
                    'batch_size': 32,
                    'num_epochs': 5
                },
                'gnn': {
                    'hidden_channels': 256,
                    'num_layers': 3,
                    'batch_size': 64,
                    'num_epochs': 10
                }
            },
            'ensemble': {
                'weights': {
                    'transformer': 0.5,
                    'cnn_lstm': 0.3,
                    'gnn': 0.2
                }
            }
        }
    
    def initialize_models(self):
        """Initialize ensemble and LLM validator"""
        logger.info("Initializing models...")
        
        # Initialize ensemble
        ensemble_config = EnsembleConfig(
            transformer_weight=self.config['ensemble']['weights']['transformer'],
            cnn_lstm_weight=self.config['ensemble']['weights']['cnn_lstm'],
            gnn_weight=self.config['ensemble']['weights']['gnn']
        )
        
        self.ensemble_detector = EnsembleSpamDetector(ensemble_config)
        
        # Initialize LLM validator
        llm_config = LLMConfig(
            provider=self.config.get('llm_validation', {}).get('provider', 'openai'),
            model=self.config.get('llm_validation', {}).get('model', 'gpt-4'),
            api_key=os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        )
        
        self.llm_validator = LLMValidator(llm_config)
        
        # Initialize dual-stage detector
        self.dual_stage_detector = DualStageDetector(
            self.ensemble_detector, self.llm_validator
        )
        
        logger.info("Models initialized successfully")
    
    def train_ensemble(self):
        """Train the complete ensemble"""
        logger.info("Starting ensemble training pipeline...")
        
        # Load and process data
        texts, labels, types = self.data_processor.combine_datasets()
        
        if not texts:
            logger.error("No data loaded for training")
            return
        
        # Create splits
        data_splits = self.data_processor.create_train_val_test_split(
            texts, labels, types
        )
        
        # Initialize models
        self.initialize_models()
        
        # Train ensemble
        logger.info("Training ensemble models...")
        self.ensemble_detector.train_all_models(
            data_splits['train']['texts'],
            data_splits['train']['labels'],
            data_splits['val']['texts'],
            data_splits['val']['labels']
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = self.ensemble_detector.ensemble_predict(
            data_splits['test']['texts']
        )
        
        # Calculate metrics
        test_accuracy = self._calculate_accuracy(
            test_results['predictions'],
            data_splits['test']['labels']
        )
        
        logger.info(f"Test accuracy: {test_accuracy:.2f}%")
        
        # Save ensemble
        self.ensemble_detector.save_ensemble("./models/ensemble")
        
        logger.info("Training completed successfully")
    
    async def test_dual_stage_system(self, test_texts: List[str] = None):
        """Test the dual-stage detection system"""
        logger.info("Testing dual-stage detection system...")
        
        if test_texts is None:
            # Use some sample texts for testing
            test_texts = [
                "Congratulations! You've won $1000! Click here to claim your prize now!",
                "Hi, are we still meeting for lunch tomorrow?",
                "Your account has been compromised. Please verify your credentials immediately.",
                "The meeting is scheduled for 3 PM in conference room B."
            ]
        
        # Perform dual-stage prediction
        results = await self.dual_stage_detector.predict(test_texts)
        
        # Generate reports
        for i, text in enumerate(test_texts):
            report = await self.dual_stage_detector.generate_comprehensive_report(
                text, results, i
            )
            
            logger.info(f"\n--- Report for Text {i+1} ---")
            logger.info(f"Text: {text[:100]}...")
            logger.info(f"Final Verdict: {report['final_verdict']}")
            logger.info(f"Confidence: {report['confidence_score']}")
            logger.info(f"Recommended Action: {report['recommended_action']}")
        
        # Print validation stats
        stats = self.dual_stage_detector.get_validation_stats()
        logger.info(f"\nValidation Statistics: {stats}")
    
    def _calculate_accuracy(self, predictions: List[int], 
                          true_labels: List[int]) -> float:
        """Calculate accuracy"""
        correct = sum(p == t for p, t in zip(predictions, true_labels))
        return (correct / len(predictions)) * 100
    
    def run_complete_pipeline(self):
        """Run the complete training and testing pipeline"""
        logger.info("Starting complete deep learning pipeline...")
        
        try:
            # Train ensemble
            self.train_ensemble()
            
            # Test dual-stage system
            asyncio.run(self.test_dual_stage_system())
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


class ModelEvaluator:
    """
    Comprehensive model evaluation and performance analysis
    """
    
    def __init__(self, ensemble_detector):
        self.ensemble_detector = ensemble_detector
        
    def evaluate_models(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """Comprehensive evaluation of all models"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Get predictions from ensemble
        ensemble_results = self.ensemble_detector.ensemble_predict(test_texts)
        
        # Calculate metrics for ensemble
        ensemble_metrics = self._calculate_metrics(
            ensemble_results['predictions'], test_labels
        )
        
        # Calculate metrics for individual models
        individual_metrics = {}
        
        for model_name in ['transformer', 'cnn_lstm', 'gnn']:
            if ensemble_results['individual_predictions'].get(model_name):
                preds = ensemble_results['individual_predictions'][model_name]['predictions']
                metrics = self._calculate_metrics(preds, test_labels)
                individual_metrics[model_name] = metrics
        
        evaluation_results = {
            'ensemble_metrics': ensemble_metrics,
            'individual_metrics': individual_metrics,
            'ensemble_weights': ensemble_results['ensemble_weights']
        }
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def _calculate_metrics(self, predictions: List[int], 
                          true_labels: List[int]) -> Dict:
        """Calculate comprehensive metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, roc_auc_score
        )
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='binary')
        recall = recall_score(true_labels, predictions, average='binary')
        f1 = f1_score(true_labels, predictions, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(cm[1, 1]),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0])
        }
        
        return metrics


# Main training function
def main():
    """Main training function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    trainer = AdvancedTrainer()
    trainer.run_complete_pipeline()


if __name__ == "__main__":
    main()

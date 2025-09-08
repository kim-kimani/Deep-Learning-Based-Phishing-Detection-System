"""
Advanced Transformer-Based Spam Detection Model
Implements BERT/RoBERTa with custom classification heads for multi-modal processing
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    mixed_precision: bool = True

class SpamDataset(Dataset):
    """Custom dataset for spam detection with transformer tokenization"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiModalTransformerModel(nn.Module):
    """
    Advanced Transformer model with multi-modal input processing
    Supports text, URLs, and metadata features
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        self.transformer_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        self.transformer = AutoModel.from_pretrained(
            config.model_name,
            config=self.transformer_config
        )
        
        # Custom classification heads
        self.text_classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_labels)
        )
        
        # URL feature classifier
        self.url_classifier = nn.Sequential(
            nn.Linear(20, 64),  # URL features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_labels)
        )
        
        # Metadata classifier
        self.metadata_classifier = nn.Sequential(
            nn.Linear(10, 32),  # Metadata features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, config.num_labels)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.num_labels * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_labels)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=config.num_labels,
            num_heads=2,
            batch_first=True
        )
        
    def forward(self, input_ids, attention_mask, url_features=None, metadata_features=None):
        # Process text through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output or last hidden state
        if hasattr(transformer_outputs, 'pooler_output') and transformer_outputs.pooler_output is not None:
            text_features = transformer_outputs.pooler_output
        else:
            text_features = transformer_outputs.last_hidden_state[:, 0]  # Use [CLS] token
        
        # Text classification
        text_logits = self.text_classifier(text_features)
        
        # Initialize final logits with text logits
        final_logits = text_logits
        
        # Process URL features if available
        if url_features is not None:
            url_logits = self.url_classifier(url_features)
            
            # Process metadata features if available
            if metadata_features is not None:
                metadata_logits = self.metadata_classifier(metadata_features)
                
                # Stack all features for attention
                all_features = torch.stack([text_logits, url_logits, metadata_logits], dim=1)
                
                # Apply attention mechanism
                attended_features, _ = self.attention(all_features, all_features, all_features)
                
                # Flatten and pass through fusion layer
                fused_features = attended_features.reshape(attended_features.size(0), -1)
                final_logits = self.fusion_layer(fused_features)
            else:
                # Only text and URL features
                combined_features = torch.cat([text_logits, url_logits], dim=1)
                final_logits = self.fusion_layer(
                    torch.cat([combined_features, torch.zeros_like(text_logits)], dim=1)
                )
        
        return final_logits

class TransformerSpamDetector:
    """
    Main class for Transformer-based spam detection
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def initialize_model(self):
        """Initialize tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = MultiModalTransformerModel(self.config)
            self.model.to(self.device)
            
            logger.info(f"Initialized {self.config.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def prepare_datasets(self, train_texts, train_labels, val_texts, val_labels):
        """Prepare training and validation datasets"""
        train_dataset = SpamDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        val_dataset = SpamDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_length
        )
        
        return train_dataset, val_dataset
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str], val_labels: List[int]):
        """Train the transformer model"""
        
        if self.model is None:
            self.initialize_model()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(
            train_texts, train_labels, val_texts, val_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./models/transformer_checkpoints",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.mixed_precision,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Starting transformer training...")
        try:
            self.trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, texts: List[str], url_features: Optional[np.ndarray] = None, 
                metadata_features: Optional[np.ndarray] = None) -> Dict:
        """
        Make predictions on input texts
        
        Args:
            texts: List of input texts
            url_features: Optional URL feature matrix (batch_size, 20)
            metadata_features: Optional metadata feature matrix (batch_size, 10)
            
        Returns:
            Dictionary with predictions, probabilities, and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                # Tokenize input
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Prepare additional features
                url_feat = None
                if url_features is not None:
                    url_feat = torch.tensor(url_features[i:i+1], dtype=torch.float32).to(self.device)
                
                meta_feat = None
                if metadata_features is not None:
                    meta_feat = torch.tensor(metadata_features[i:i+1], dtype=torch.float32).to(self.device)
                
                # Forward pass
                logits = self.model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    url_features=url_feat,
                    metadata_features=meta_feat
                )
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                
                predictions.append(pred.cpu().numpy()[0])
                probabilities.append(probs.cpu().numpy()[0])
        
        # Calculate confidence scores
        confidence_scores = [max(prob) * 100 for prob in probabilities]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'labels': ['HAM', 'SPAM']
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer': self.tokenizer
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.tokenizer = checkpoint['tokenizer']
        
        self.model = MultiModalTransformerModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {path}")

# Uncertainty quantification using Monte Carlo Dropout
class UncertaintyQuantifier:
    """Quantify prediction uncertainty using Monte Carlo Dropout"""
    
    def __init__(self, model, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
    
    def enable_dropout(self):
        """Enable dropout for uncertainty estimation"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, input_ids, attention_mask, 
                                url_features=None, metadata_features=None):
        """Predict with uncertainty quantification"""
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    url_features=url_features,
                    metadata_features=metadata_features
                )
                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty

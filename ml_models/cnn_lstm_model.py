"""
Hybrid CNN-LSTM Architecture for Spam Detection
Combines convolutional layers with bidirectional LSTM and attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

logger = logging.getLogger(__name__)


class CNNLSTMConfig:
    """Configuration for CNN-LSTM model"""
    def __init__(self):
        self.vocab_size = 50000
        self.embedding_dim = 300
        self.cnn_filters = [64, 128, 256]
        self.kernel_sizes = [3, 4, 5]
        self.lstm_hidden_size = 128
        self.lstm_layers = 2
        self.dropout = 0.3
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.num_epochs = 10
        self.max_length = 256
        self.num_classes = 2


class TextProcessor:
    """Advanced text preprocessing for CNN-LSTM model"""
    
    def __init__(self, vocab_size=50000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.char_to_idx = {}
        self.idx_to_char = {}
        
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from training texts"""
        # Tokenize all texts
        all_words = []
        all_chars = set()
        
        for text in texts:
            words = self.tokenize_text(text)
            all_words.extend(words)
            all_chars.update(text.lower())
        
        # Build word vocabulary
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        self.word_to_idx = {'<UNK>': 0, '<PAD>': 1}
        self.idx_to_word = {0: '<UNK>', 1: '<PAD>'}
        
        for i, (word, _) in enumerate(most_common, 2):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        # Build character vocabulary
        self.char_to_idx = {'<UNK>': 0, '<PAD>': 1}
        self.idx_to_char = {0: '<UNK>', 1: '<PAD>'}
        
        for i, char in enumerate(sorted(all_chars), 2):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        
        logger.info(f"Built vocabulary: {len(self.word_to_idx)} words, "
                   f"{len(self.char_to_idx)} characters")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        words = text.split()
        return words
    
    def text_to_sequence(self, text: str) -> Tuple[List[int], List[int]]:
        """Convert text to word and character sequences"""
        # Word sequence
        words = self.tokenize_text(text)
        word_seq = [self.word_to_idx.get(word, 0) for word in words]
        
        # Pad or truncate
        if len(word_seq) < self.max_length:
            word_seq.extend([1] * (self.max_length - len(word_seq)))
        else:
            word_seq = word_seq[:self.max_length]
        
        # Character sequence
        char_seq = [self.char_to_idx.get(char, 0) for char in text.lower()]
        
        # Pad or truncate character sequence
        if len(char_seq) < self.max_length * 4:
            char_seq.extend([1] * (self.max_length * 4 - len(char_seq)))
        else:
            char_seq = char_seq[:self.max_length * 4]
        
        return word_seq, char_seq


class SpamDatasetCNNLSTM(Dataset):
    """Dataset for CNN-LSTM model"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 text_processor: TextProcessor):
        self.texts = texts
        self.labels = labels
        self.text_processor = text_processor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        word_seq, char_seq = self.text_processor.text_to_sequence(text)
        
        return {
            'word_sequence': torch.tensor(word_seq, dtype=torch.long),
            'char_sequence': torch.tensor(char_seq, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text_length': torch.tensor(len(text), dtype=torch.float32)
        }


class AttentionLayer(nn.Module):
    """Attention mechanism for feature fusion"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(
            self.attention(lstm_output), dim=1
        )
        attended_output = torch.sum(
            attention_weights * lstm_output, dim=1
        )
        return attended_output, attention_weights


class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model with attention mechanism
    """
    
    def __init__(self, config: CNNLSTMConfig, vocab_size: int, 
                 char_vocab_size: int):
        super().__init__()
        self.config = config
        
        # Word embedding
        self.word_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=1
        )
        
        # Character embedding
        self.char_embedding = nn.Embedding(
            char_vocab_size, config.embedding_dim // 4, padding_idx=1
        )
        
        # CNN layers for character-level features
        self.char_convs = nn.ModuleList([
            nn.Conv1d(
                config.embedding_dim // 4, 
                filters, 
                kernel_size
            )
            for filters, kernel_size in zip(
                config.cnn_filters, config.kernel_sizes
            )
        ])
        
        # CNN layers for word-level features
        self.word_convs = nn.ModuleList([
            nn.Conv1d(
                config.embedding_dim, 
                filters, 
                kernel_size
            )
            for filters, kernel_size in zip(
                config.cnn_filters, config.kernel_sizes
            )
        ])
        
        # Bidirectional LSTM
        lstm_input_size = sum(config.cnn_filters) * 2  # char + word CNNs
        self.lstm = nn.LSTM(
            lstm_input_size,
            config.lstm_hidden_size,
            config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(
            config.lstm_hidden_size * 2  # bidirectional
        )
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(config.lstm_hidden_size * 2 + 1, 256),  # +1 for text length
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, word_sequence, char_sequence, text_length):
        batch_size = word_sequence.size(0)
        
        # Word embeddings
        word_embedded = self.word_embedding(word_sequence)
        # Shape: (batch_size, seq_len, embedding_dim)
        
        # Character embeddings
        char_embedded = self.char_embedding(char_sequence)
        # Shape: (batch_size, char_seq_len, embedding_dim//4)
        
        # CNN for word-level features
        word_embedded_T = word_embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        word_conv_outputs = []
        
        for conv in self.word_convs:
            conv_out = F.relu(conv(word_embedded_T))  # (batch_size, filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, filters, 1)
            word_conv_outputs.append(pooled.squeeze(2))  # (batch_size, filters)
        
        word_features = torch.cat(word_conv_outputs, dim=1)  # (batch_size, sum(filters))
        
        # CNN for character-level features
        char_embedded_T = char_embedded.transpose(1, 2)
        char_conv_outputs = []
        
        for conv in self.char_convs:
            conv_out = F.relu(conv(char_embedded_T))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            char_conv_outputs.append(pooled.squeeze(2))
        
        char_features = torch.cat(char_conv_outputs, dim=1)
        
        # Combine word and character features
        combined_features = torch.cat([word_features, char_features], dim=1)
        # Shape: (batch_size, combined_feature_size)
        
        # Reshape for LSTM (add sequence dimension)
        lstm_input = combined_features.unsqueeze(1)  # (batch_size, 1, feature_size)
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        # lstm_output shape: (batch_size, 1, lstm_hidden_size * 2)
        
        # Apply attention
        attended_output, attention_weights = self.attention(lstm_output)
        # attended_output shape: (batch_size, lstm_hidden_size * 2)
        
        # Enhance features with additional information
        enhanced_features = torch.cat([
            attended_output, 
            text_length.unsqueeze(1)
        ], dim=1)
        
        enhanced_features = self.feature_enhancer(enhanced_features)
        
        # Final classification
        logits = self.classifier(enhanced_features)
        
        return logits, attention_weights


class CNNLSTMSpamDetector:
    """
    Main class for CNN-LSTM spam detection
    """
    
    def __init__(self, config: CNNLSTMConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_processor = TextProcessor(
            config.vocab_size, config.max_length
        )
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def prepare_data(self, texts: List[str], labels: List[int], 
                    is_training: bool = True):
        """Prepare data for training or inference"""
        if is_training and not hasattr(self.text_processor, 'word_to_idx'):
            self.text_processor.build_vocabulary(texts)
        
        dataset = SpamDatasetCNNLSTM(texts, labels, self.text_processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=4 if is_training else 1
        )
        
        return dataloader
    
    def initialize_model(self):
        """Initialize the CNN-LSTM model"""
        vocab_size = len(self.text_processor.word_to_idx)
        char_vocab_size = len(self.text_processor.char_to_idx)
        
        self.model = CNNLSTMModel(
            self.config, vocab_size, char_vocab_size
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"CNN-LSTM model initialized on {self.device}")
        logger.info(f"Model parameters: "
                   f"{sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            word_seq = batch['word_sequence'].to(self.device)
            char_seq = batch['char_sequence'].to(self.device)
            text_len = batch['text_length'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, attention_weights = self.model(word_seq, char_seq, text_len)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                word_seq = batch['word_sequence'].to(self.device)
                char_seq = batch['char_sequence'].to(self.device)
                text_len = batch['text_length'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(word_seq, char_seq, text_len)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int]):
        """Train the CNN-LSTM model"""
        # Prepare data
        train_loader = self.prepare_data(train_texts, train_labels, True)
        val_loader = self.prepare_data(val_texts, val_labels, False)
        
        # Initialize model
        self.initialize_model()
        
        best_val_accuracy = 0
        patience_counter = 0
        patience = 5
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                self.save_model("./models/cnn_lstm_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    
    def predict(self, texts: List[str]) -> Dict:
        """Make predictions on input texts"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data (with dummy labels)
        dummy_labels = [0] * len(texts)
        dataloader = self.prepare_data(texts, dummy_labels, False)
        
        self.model.eval()
        predictions = []
        probabilities = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                word_seq = batch['word_sequence'].to(self.device)
                char_seq = batch['char_sequence'].to(self.device)
                text_len = batch['text_length'].to(self.device)
                
                logits, attention_weights = self.model(word_seq, char_seq, text_len)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                attention_weights_list.extend(attention_weights.cpu().numpy())
        
        confidence_scores = [max(prob) * 100 for prob in probabilities]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'attention_weights': attention_weights_list,
            'labels': ['HAM', 'SPAM']
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'text_processor': self.text_processor,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
        logger.info(f"CNN-LSTM model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.text_processor = checkpoint['text_processor']
        
        vocab_size = len(self.text_processor.word_to_idx)
        char_vocab_size = len(self.text_processor.char_to_idx)
        
        self.model = CNNLSTMModel(
            self.config, vocab_size, char_vocab_size
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"CNN-LSTM model loaded from {path}")

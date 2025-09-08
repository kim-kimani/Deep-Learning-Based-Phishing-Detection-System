"""
Graph Neural Network Architecture for Spam Detection
Implements GraphSAGE for relationship-based detection and behavioral analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import networkx as nx
from collections import defaultdict
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class GNNConfig:
    """Configuration for Graph Neural Network model"""
    def __init__(self):
        self.node_features = 768
        self.hidden_channels = 256
        self.num_layers = 3
        self.dropout = 0.2
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.num_epochs = 15
        self.aggregation = "mean"
        self.num_classes = 2
        self.use_attention = True


class GraphConstructor:
    """Construct graphs from text data for GNN processing"""
    
    def __init__(self):
        self.domain_graph = nx.Graph()
        self.word_graph = nx.Graph()
        self.feature_extractor = FeatureExtractor()
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text for graph construction"""
        entities = {
            'urls': [],
            'emails': [],
            'phones': [],
            'keywords': [],
            'domains': []
        }
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        entities['urls'] = urls
        
        # Extract domains from URLs
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if domain:
                    entities['domains'].append(domain)
            except:
                pass
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        entities['emails'] = emails
        
        # Extract phone numbers
        phone_pattern = r'(\+?\d{1,4}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        entities['phones'] = [phone[0] + phone[1] + phone[2] for phone in phones if any(phone)]
        
        # Extract suspicious keywords
        spam_keywords = [
            'free', 'win', 'prize', 'urgent', 'click', 'offer', 'deal',
            'limited', 'act now', 'congratulations', 'selected', 'bonus'
        ]
        keywords = [word for word in spam_keywords if word in text.lower()]
        entities['keywords'] = keywords
        
        return entities
    
    def build_message_graph(self, text: str, label: int) -> Data:
        """Build a graph representation of a single message"""
        entities = self.extract_entities(text)
        
        # Create nodes for different entity types
        nodes = []
        node_features = []
        node_types = []
        
        # Add text node (central node)
        nodes.append('text_node')
        text_features = self.feature_extractor.extract_text_features(text)
        node_features.append(text_features)
        node_types.append(0)  # text type
        
        # Add URL nodes
        for i, url in enumerate(entities['urls']):
            nodes.append(f'url_{i}')
            url_features = self.feature_extractor.extract_url_features(url)
            node_features.append(url_features)
            node_types.append(1)  # URL type
        
        # Add domain nodes
        for i, domain in enumerate(entities['domains']):
            nodes.append(f'domain_{i}')
            domain_features = self.feature_extractor.extract_domain_features(domain)
            node_features.append(domain_features)
            node_types.append(2)  # domain type
        
        # Add keyword nodes
        for i, keyword in enumerate(entities['keywords']):
            nodes.append(f'keyword_{i}')
            keyword_features = self.feature_extractor.extract_keyword_features(keyword)
            node_features.append(keyword_features)
            node_types.append(3)  # keyword type
        
        # Create edges (connections between entities)
        edges = []
        
        # Connect text node to all other nodes
        for i in range(1, len(nodes)):
            edges.append([0, i])  # text node to entity
            edges.append([i, 0])  # entity to text node
        
        # Connect related entities
        for i in range(1, len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Connect nodes of the same type or related types
                if (node_types[i] == node_types[j] or 
                    (node_types[i] in [1, 2] and node_types[j] in [1, 2])):
                    edges.append([i, j])
                    edges.append([j, i])
        
        # Convert to tensors
        if not node_features:
            # If no features, create dummy features
            node_features = [[0.0] * 768]
            nodes = ['dummy_node']
            edges = []
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def build_batch_graph(self, texts: List[str], labels: List[int]) -> List[Data]:
        """Build graph representations for a batch of messages"""
        graphs = []
        for text, label in zip(texts, labels):
            graph = self.build_message_graph(text, label)
            graphs.append(graph)
        return graphs


class FeatureExtractor:
    """Extract features for different entity types"""
    
    def __init__(self):
        self.feature_dim = 768
        
    def extract_text_features(self, text: str) -> List[float]:
        """Extract features from text"""
        features = [0.0] * self.feature_dim
        
        # Basic text statistics
        features[0] = len(text)
        features[1] = len(text.split())
        features[2] = text.count('!')
        features[3] = text.count('?')
        features[4] = text.count('$')
        features[5] = text.count('%')
        features[6] = sum(c.isupper() for c in text)
        features[7] = sum(c.isdigit() for c in text)
        
        # Spam indicators
        spam_words = ['free', 'win', 'prize', 'urgent', 'click', 'offer']
        features[8] = sum(word in text.lower() for word in spam_words)
        
        # URL indicators
        features[9] = text.count('http')
        features[10] = text.count('www')
        
        return features
    
    def extract_url_features(self, url: str) -> List[float]:
        """Extract features from URL"""
        features = [0.0] * self.feature_dim
        
        try:
            parsed = urlparse(url)
            
            features[0] = len(url)
            features[1] = url.count('.')
            features[2] = url.count('-')
            features[3] = url.count('_')
            features[4] = url.count('/')
            features[5] = url.count('?')
            features[6] = url.count('&')
            features[7] = url.count('=')
            features[8] = int('https' in url.lower())
            features[9] = len(parsed.path)
            features[10] = len(parsed.query)
            
            # Suspicious TLD check
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
            features[11] = int(any(tld in url for tld in suspicious_tlds))
            
        except:
            pass
        
        return features
    
    def extract_domain_features(self, domain: str) -> List[float]:
        """Extract features from domain"""
        features = [0.0] * self.feature_dim
        
        features[0] = len(domain)
        features[1] = domain.count('.')
        features[2] = domain.count('-')
        features[3] = sum(c.isdigit() for c in domain)
        
        # Check for suspicious patterns
        suspicious_patterns = ['secure', 'update', 'verify', 'account']
        features[4] = sum(pattern in domain.lower() for pattern in suspicious_patterns)
        
        return features
    
    def extract_keyword_features(self, keyword: str) -> List[float]:
        """Extract features from keywords"""
        features = [0.0] * self.feature_dim
        
        # Keyword importance weights
        spam_weights = {
            'free': 0.9,
            'win': 0.8,
            'prize': 0.8,
            'urgent': 0.7,
            'click': 0.6,
            'offer': 0.5
        }
        
        features[0] = len(keyword)
        features[1] = spam_weights.get(keyword.lower(), 0.1)
        
        return features


class GraphSAGEModel(nn.Module):
    """GraphSAGE model for spam detection"""
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(config.node_features, config.hidden_channels))
        
        for _ in range(config.num_layers - 2):
            self.convs.append(SAGEConv(config.hidden_channels, config.hidden_channels))
        
        self.convs.append(SAGEConv(config.hidden_channels, config.hidden_channels))
        
        # Attention layers (optional)
        if config.use_attention:
            self.attention_convs = nn.ModuleList()
            for _ in range(config.num_layers):
                self.attention_convs.append(
                    GATConv(config.hidden_channels, config.hidden_channels, heads=4, concat=False)
                )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(config.num_layers):
            self.batch_norms.append(nn.BatchNorm1d(config.hidden_channels))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, config.hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 4, config.num_classes)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, edge_index, batch):
        # Apply GraphSAGE convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Apply attention if enabled
            if self.config.use_attention and i < len(self.attention_convs):
                x_att = self.attention_convs[i](x, edge_index)
                x = x + x_att  # Residual connection
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class GNNSpamDetector:
    """
    Main class for Graph Neural Network spam detection
    """
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_constructor = GraphConstructor()
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def initialize_model(self):
        """Initialize the GNN model"""
        self.model = GraphSAGEModel(self.config).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"GNN model initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    is_training: bool = True) -> DataLoader:
        """Prepare graph data for training or inference"""
        graphs = self.graph_constructor.build_batch_graph(texts, labels)
        
        dataloader = DataLoader(
            graphs,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=2 if is_training else 1
        )
        
        return dataloader
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
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
                batch = batch.to(self.device)
                
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(logits, batch.y)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int]):
        """Train the GNN model"""
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
                self.save_model("./models/gnn_best.pth")
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
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
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
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
        logger.info(f"GNN model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        self.model = GraphSAGEModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"GNN model loaded from {path}")


class GraphAnalyzer:
    """Analyze graph structures for insights"""
    
    def __init__(self):
        self.graph_stats = defaultdict(list)
    
    def analyze_message_graph(self, text: str) -> Dict:
        """Analyze the graph structure of a message"""
        constructor = GraphConstructor()
        entities = constructor.extract_entities(text)
        
        analysis = {
            'num_urls': len(entities['urls']),
            'num_domains': len(entities['domains']),
            'num_emails': len(entities['emails']),
            'num_phones': len(entities['phones']),
            'num_keywords': len(entities['keywords']),
            'total_entities': sum(len(v) for v in entities.values()),
            'entity_diversity': len([k for k, v in entities.items() if v]),
            'suspicious_indicators': self._count_suspicious_indicators(entities)
        }
        
        return analysis
    
    def _count_suspicious_indicators(self, entities: Dict) -> int:
        """Count suspicious indicators in entities"""
        count = 0
        
        # Multiple URLs
        if len(entities['urls']) > 2:
            count += 1
        
        # Suspicious domains
        for domain in entities['domains']:
            if any(susp in domain for susp in ['secure', 'update', 'verify']):
                count += 1
        
        # Many spam keywords
        if len(entities['keywords']) > 3:
            count += 1
        
        return count

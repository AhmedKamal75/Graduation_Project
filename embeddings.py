# embeddings.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss implementation with cosine margin for enhanced feature discrimination.
    Applies an angular margin penalty to the target logit 
        (20, 6, 14) --> (30.0, 0.5).
        (3) --> (45.0, 0.7)
    """
    def __init__(self, in_features, out_features, scale=45.0, margin=0.7, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight parameter representing the prototype vectors
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute angular margin parameters
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # Normalize feature vectors
        x = F.normalize(input)
        # Normalize weights
        W = F.normalize(self.weight)
        
        # Compute cosine similarity
        cosine = F.linear(x, W)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Add angular margin
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert labels to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to target classes only
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale logits
        output *= self.scale
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class DeepResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeepResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += residual
        out = F.relu(out)
        return out
    
class EnhancedFaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, input_size=224):
        super(EnhancedFaceRecognitionModel, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Deep residual layers (total >30 conv layers)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)      # 6 conv layers
        self.layer2 = self._make_layer(64, 128, 4, stride=2)     # 8 conv layers
        self.layer3 = self._make_layer(128, 256, 6, stride=2)    # 12 conv layers
        self.layer4 = self._make_deep_layer(256, 512, 3, stride=2) # 9 conv layers
        
        # Calculate feature size
        feature_map_size = input_size // 32  # Due to downsampling
        self.feature_size = feature_map_size * feature_map_size * 512
        
        # Simplified FC layers (2 layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        self.arc_face = ArcFaceLoss(embedding_size, num_classes)
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _make_deep_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(DeepResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(DeepResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        embedding = self.fc_layers(x)
        embedding_normalized = F.normalize(embedding, p=2, dim=1)
        
        if labels is not None:
            output = self.arc_face(embedding_normalized, labels)
            return output, embedding_normalized
        else:
            return embedding_normalized

class EmbeddingPredictor():
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.load_model()

    # Load model
    def load_model(self):
        print(f"Loading embedding model from {self.model_path}")
        print(f"Running on device: {self.device}")
        
        self.model = EnhancedFaceRecognitionModel(
            num_classes=4605,  # Original number of classes from VGGFace2
            embedding_size=512,
            input_size=224
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model = self.model.to(self.device)
        self.model.eval()   
        
        print(f"Embedding model loaded.")
        
    def preprocess_face(self, face):
        """
        Preprocess a face image for the model.
        Args:
            face (numpy.ndarray): Cropped face image.
        Returns:
            torch.Tensor: Preprocessed face tensor.
        """
        face = cv2.resize(face, (224, 224))  # Resize to match model input size
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face = np.transpose(face, (2, 0, 1))  # HWC to CHW
        face = face / 255.0  # Normalize to [0, 1]
        face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return face
    
    def generate_embedding(self, face):
        """generate embeddings for a given face

        Args:
            face (numpy.ndarray): Cropped face image.

        Returns:
            numpy.ndarray: Embedding vector.
        """
        face_tensor = self.preprocess_face(face).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor).cpu().numpy()
        return embedding
    
    
if __name__ == "__main__":
    # and example
    embedding_predictor = EmbeddingPredictor(model_path='models/resarksgd/resarksgd95.pth', device='cpu')
    # example face using random generator
    face = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    embedding = embedding_predictor.generate_embedding(face=face)
    print(embedding)
    print(f"shape: {embedding.shape}")
    print(f"shape after flattening: {embedding.flatten().shape}")
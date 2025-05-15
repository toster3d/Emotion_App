import torch
import torch.nn as nn
import torchvision.models as models

class AudioResNet(nn.Module):
    """
    ResNet-based model for audio emotion classification
    
    The model uses a modified ResNet-18 architecture with a single input channel
    for processing melspectrograms or other audio features.
    """
    
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5):
        """
        Initialize the AudioResNet model
        
        Args:
            num_classes: Number of emotion classes to predict
            dropout_rate: Dropout rate for regularization
        """
        super(AudioResNet, self).__init__()
        
        # Use ResNet18 as the base architecture
        self.resnet = models.resnet18(weights=None)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the number of features from the final fully connected layer
        num_features = self.resnet.fc.in_features
        
        # Replace final FC layer with identity to extract features
        self.resnet.fc = nn.Identity()
        
        # Add dropout and our own fully connected layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for different layers using appropriate initialization methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor containing audio features (batch, channel, height, width)
                
        Returns:
            Logits for each emotion class
        """
        # Pass through the ResNet backbone
        x = self.resnet(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc(x)
        
        return x
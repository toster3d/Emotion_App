import torch.nn as nn
from torch import Tensor
from typing import cast
from typing_extensions import override
from torchvision.models import resnet18 # type: ignore[reportMissingTypeStubs]

class AudioResNet(nn.Module):
    """
    ResNet-based model for audio emotion classification.

    Uses ResNet-18 with a single-channel input (mel spectrogram) instead of RGB.
    """    
    resnet: nn.Module
    dropout: nn.Dropout
    fc: nn.Linear

    def __init__(
        self,
        num_classes: int = 6,
        dropout_rate: float = 0.5
    ) -> None:
        super().__init__() # type: ignore[reportUnknownMemberType]
        
        # 1) ResNet-18 backbone without pretrained weights
        self.resnet = resnet18(weights=None)
        
        # 2) Change the first convolutional layer (conv1) for 1-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # 3) Get the number of features from the original fc layer
        num_features: int = self.resnet.fc.in_features
        
        # 4) Remove the old fc layer and add dropout + our new fc layer
        self.resnet.fc = cast(nn.Module, nn.Identity()) # type: ignore[reportAttributeTypeMismatch]
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=num_features, out_features=num_classes)
        
        # 5) Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize the weights of selected layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight nie może być None przy bias=False
                assert m.weight is not None
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                assert m.weight is not None and m.bias is not None
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                assert m.weight is not None and m.bias is not None
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (N, 1, H, W) – batch size, channel, height, width

        Returns:
            Tensor of shape (N, num_classes) – non-linear logits
        """
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

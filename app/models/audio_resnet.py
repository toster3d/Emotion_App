import torch.nn as nn
from torch import Tensor
from typing import cast
from typing_extensions import override
from torchvision.models import resnet18 # type: ignore[reportMissingTypeStubs]

class AudioResNet(nn.Module):
    """
    ResNet-based model for audio emotion classification.

    Uses ResNet-18 z jednowymiarowym wejściem (melspektrogram) zamiast RGB.
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
        
        # 1) Backbone ResNet-18 bez pretrenowanych wag
        self.resnet = resnet18(weights=None)
        
        # 2) Zmieniamy pierwszą warstwę conv1 na 1-kanał wejściowy
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # 3) Wyciągamy liczbę cech z oryginalnej warstwy fc
        num_features: int = self.resnet.fc.in_features
        
        # 4) Usuwamy starą warstwę fc i dodajemy dropout + naszą fc
        self.resnet.fc = cast(nn.Module, nn.Identity()) # type: ignore[reportAttributeTypeMismatch]
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=num_features, out_features=num_classes)
        
        # 5) Inicjalizacja wag
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Zainicjalizuj wagi wybranych warstw."""
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
            x: Tensor o kształcie (N, 1, H, W) – batch, kanał, wysokość, szerokość

        Returns:
            Tensor o kształcie (N, num_classes) – nieliniowe logity
        """
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

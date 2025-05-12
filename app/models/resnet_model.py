import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Klasa definiująca model AudioResNet oparty na architekturze ResNet
class AudioResNet(nn.Module):
    """
    Model bazujący na ResNet18 do klasyfikacji emocji na podstawie cech audio.
    
    Argumenty:
        num_classes (int): Liczba klas emocji do klasyfikacji
        feature_type (str): Typ cech audio ("melspectrogram", "mfcc", "chroma")
        dropout_rate (float): Współczynnik dropout dla warstwy regularyzacyjnej
    """
    def __init__(self, feature_type=None, num_classes=6, dropout_rate=0.5):
        super(AudioResNet, self).__init__()
        self.feature_type = feature_type
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.resnet.fc.in_features
        self.dropout = nn.Dropout(dropout_rate)
        self.resnet.fc = nn.Identity()  # Ostatnia warstwa została usunięta
        self.fc = nn.Linear(num_features, num_classes)
        
        # Inicjalizacja wag dla różnych warstw modelu
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
        Przechodzenie do przodu przez model.
        
        Argumenty:
            x (torch.Tensor): Tensor cech audio o kształcie [batch_size, 1, height, width]
            
        Zwraca:
            torch.Tensor: Logits dla każdej klasy, kształt [batch_size, num_classes]
        """
        # Przechodzenie danych przez model
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x 
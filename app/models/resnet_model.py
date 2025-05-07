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
    def __init__(self, num_classes=6, feature_type="melspectrogram", dropout_rate=0.5):
        super(AudioResNet, self).__init__()
        self.feature_type = feature_type
        
        # Wczytanie pretrainowanego modelu ResNet18
        self.resnet = models.resnet18(weights=None)
        
        # Modyfikacja pierwszej warstwy konwolucyjnej, aby akceptowała 1 kanał
        # (domyślnie 3 kanały dla RGB)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Zastąpienie ostatniej warstwy
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Usunięcie oryginalnej warstwy fc
        
        # Dodanie własnych warstw
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        """
        Przechodzenie do przodu przez model.
        
        Argumenty:
            x (torch.Tensor): Tensor cech audio o kształcie [batch_size, 1, height, width]
            
        Zwraca:
            torch.Tensor: Logits dla każdej klasy, kształt [batch_size, num_classes]
        """
        # Ekstrakcja cech przez ResNet
        x = self.resnet(x)
        
        # Dropout dla regularyzacji
        x = self.dropout(x)
        
        # Warstwa klasyfikacyjna
        x = self.fc(x)
        
        return x 
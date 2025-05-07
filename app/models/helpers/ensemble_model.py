import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedEnsembleModel(nn.Module):
    """
    Ważony model zespołowy, który łączy prognozy z wielu modeli różnych typów cech.
    
    Argumenty:
        models_dict (dict): Słownik modeli w formacie {typ_cechy: model}
        weights (dict, optional): Początkowe wagi dla każdego typu cechy
        temperature (float, optional): Parametr temperatury dla kalibracji prawdopodobieństw
        regularization_strength (float, optional): Siła regularyzacji L1 dla wag
    """
    def __init__(self, models_dict, weights=None, temperature=1.0, regularization_strength=0.01):
        super(WeightedEnsembleModel, self).__init__()
        self.models = nn.ModuleDict(models_dict)
        self.feature_types = list(models_dict.keys())
        
        # Inicjalizacja wag
        if weights is None:
            # Równomierne przypisanie wag
            weights_tensor = torch.ones(len(self.feature_types)) / len(self.feature_types)
        else:
            # Wykorzystanie podanego słownika wag
            weights_tensor = torch.tensor([weights[ft] for ft in self.feature_types])
        
        # Umożliwienie uczenia się wag jako parametrów, jeśli to konieczne
        self.weights = nn.Parameter(weights_tensor, requires_grad=False)
        
        # Parametr temperatury do regulacji prognoz
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        
        # Ustalenie siły regulacji
        self.regularization_strength = regularization_strength
        
        # Generowanie znormalizowanych wag do wnioskowania
        self._update_normalized_weights()
            
    def _update_normalized_weights(self):
        """Aktualizacja znormalizowanych wag"""
        normalized = F.softmax(self.weights, dim=0)
        self.normalized_weights = {ft: normalized[i].item() 
                                  for i, ft in enumerate(self.feature_types)}
        return self.normalized_weights
    
    def forward(self, inputs):
        """
        Przechodzenie do przodu modelu zespołowego.
        
        Argumenty:
            inputs (dict): Słownik tensorów wejściowych w formacie {typ_cechy: tensor}
            
        Zwraca:
            torch.Tensor: Ważona suma prawdopodobieństw z modeli
        """
        outputs = []
        available_features = []
        
        for i, ft in enumerate(self.feature_types):
            if ft in inputs:
                # Uzyskiwanie wyjścia modelu
                model_output = self.models[ft](inputs[ft])
                # Skalowanie wyjścia za pomocą temperatury
                scaled_output = model_output / self.temperature
                # Zastosowanie softmax w celu uzyskania prawdopodobieństw
                probs = F.softmax(scaled_output, dim=1)
                outputs.append(probs)
                available_features.append(i)
        
        if not outputs:
            raise ValueError("Brak danych wejściowych dla modeli")
        
        # Uzyskiwanie wag dla dostępnych cech i ich normalizacja
        available_weights = self.weights[available_features]
        normalized_weights = F.softmax(available_weights, dim=0)
        
        # Zastosowanie wag do wyjścia każdego modelu
        weighted_sum = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, normalized_weights):
            weighted_sum += output * weight
            
        return weighted_sum
    
    def get_weights(self):
        """Zwracanie aktualnych znormalizowanych wag"""
        return self._update_normalized_weights()
    
    def set_weights(self, weights_dict):
        """
        Ustawianie nowych wag na podstawie słownika
        
        Argumenty:
            weights_dict (dict): Słownik wag w formacie {typ_cechy: waga}
        """
        for i, ft in enumerate(self.feature_types):
            if ft in weights_dict:
                self.weights.data[i] = weights_dict[ft]
        self._update_normalized_weights()
    
    def l1_regularization(self):
        """Zastosowanie regulacji L1 w celu promowania rzadkich wag"""
        return self.regularization_strength * torch.norm(self.weights, p=1)
        
    def save(self, path):
        """
        Zapis modelu wraz z wagami i parametrami
        
        Argumenty:
            path (str): Ścieżka do zapisu modelu
        """
        try:
            state = {
                'model_state_dict': self.state_dict(),
                'feature_types': self.feature_types,
                'normalized_weights': self.normalized_weights,
                'temperature': self.temperature.item(),
                'regularization_strength': self.regularization_strength,
                'model_version': '1.0',  # Dodanie wersji modelu
                'pytorch_version': torch.__version__  # Dodanie wersji PyTorch
            }
            torch.save(state, path)
        except Exception as e:
            raise RuntimeError(f"Błąd podczas zapisywania modelu: {str(e)}")
    
    def save_weights_only(self, path):
        """
        Zapis tylko wag modelu (mniejszy rozmiar pliku)
        """
        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise RuntimeError(f"Błąd podczas zapisywania wag modelu: {str(e)}")
    
    @classmethod
    def load(cls, path, models_dict, device='cpu', strict=True):
        """
        Ładowanie modelu z pliku
        
        Argumenty:
            path (str): Ścieżka do pliku modelu
            models_dict (dict): Słownik modeli w formacie {typ_cechy: model}
            device (str): Urządzenie na którym ma być załadowany model ('cpu' lub 'cuda')
            strict (bool): Czy ściśle sprawdzać zgodność kluczy przy ładowaniu state_dict
            
        Zwraca:
            WeightedEnsembleModel: Załadowany model
        """
        try:
            # Dodanie weights_only=False aby rozwiązać problem z nowymi zabezpieczeniami PyTorch 2.6+
            state = torch.load(path, map_location=device, weights_only=False)
            
            # Sprawdzenie wersji modelu (opcjonalne)
            if 'model_version' in state and state['model_version'] != '1.0':
                print(f"Ostrzeżenie: Ładowany model ma inną wersję ({state['model_version']})")
            
            # Tworzenie modelu z tymi samymi parametrami
            model = cls(
                models_dict=models_dict,
                temperature=state['temperature'],
                regularization_strength=state['regularization_strength']
            )
            
            # Ładowanie stanu
            model.load_state_dict(state['model_state_dict'], strict=strict)
            model.to(device)
            
            # Aktualizacja wag po załadowaniu
            model._update_normalized_weights()
            
            return model
        except Exception as e:
            raise RuntimeError(f"Błąd podczas ładowania modelu: {str(e)}") 
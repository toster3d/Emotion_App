import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging

logger = logging.getLogger(__name__)

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
        
        # Użycie torch.inference_mode() dla optymalizacji
        with torch.inference_mode():
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
        
    def save(self, path, class_names=None, version="1.0"):
        """
        Zapis modelu wraz z wagami i parametrami
        
        Argumenty:
            path (str): Ścieżka do zapisu modelu
            class_names (list, optional): Lista nazw klas
            version (str, optional): Wersja modelu
        """
        # Przygotuj stan do zapisu
        state = {
            'model_state_dict': self.state_dict(),
            'feature_types': list(self.feature_types),
            'normalized_weights': {k: float(v) for k, v in self.normalized_weights.items()},
            'temperature': float(self.temperature.item()),
            'regularization_strength': float(self.regularization_strength),
            'class_names': [str(name) for name in class_names] if class_names is not None else None,
            'model_version': version,
            'pytorch_version': torch.__version__
        }
        
        # Zapisz z obsługą różnych wersji PyTorch
        torch.save(state, path)
        
        # Przygotowanie metadanych w formacie JSON (tylko serializowalne typy)
        json_metadata = {
            'feature_types': list(self.feature_types),
            'normalized_weights': {k: float(v) for k, v in self.normalized_weights.items()},
            'temperature': float(self.temperature.item()),
            'regularization_strength': float(self.regularization_strength),
            'class_names': [str(name) for name in class_names] if class_names is not None else None,
            'model_version': version,
            'pytorch_version': torch.__version__
        }
        
        # Zapisz również metadane oddzielnie dla łatwiejszego dostępu
        metadata_path = str(path).replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        logger.info(f"Model ensemble zapisany do {path}, metadane do {metadata_path}")
    
    @classmethod
    def load(cls, path, models_dict):
        """
        Ładowanie modelu z pliku z obsługą różnych wersji PyTorch
        
        Argumenty:
            path (str): Ścieżka do pliku modelu
            models_dict (dict): Słownik modeli w formacie {typ_cechy: model}
            
        Zwraca:
            tuple: (model, class_names)
        """
        logger.info(f"Rozpoczynanie ładowania modelu ensemble z {path}")
        
        # Spróbuj najpierw załadować z metadanych JSON (szybsza i bezpieczniejsza opcja)
        try:
            metadata_path = str(path).replace('.pt', '_metadata.json')
            logger.info(f"Próba ładowania metadanych z {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            temperature = metadata.get('temperature', 1.0)
            regularization_strength = metadata.get('regularization_strength', 0.01)
            class_names = metadata.get('class_names')
            feature_types = metadata.get('feature_types', list(models_dict.keys()))
            weights = metadata.get('normalized_weights')
            
            logger.info(f"Metadane załadowane prawidłowo. Feature types: {feature_types}")
            
            # Stwórz model z metadanych
            model = cls(
                models_dict=models_dict,
                temperature=temperature,
                regularization_strength=regularization_strength
            )
            
            # Ustaw wagi, jeśli są dostępne
            if weights:
                model.set_weights(weights)
                logger.info(f"Wczytano wagi z metadanych: {weights}")
            
            # Ładujemy stan tylko dla warstw
            try:
                # Rejestracja bezpiecznych typów dla PyTorch 2.6+
                try:
                    import torch.serialization
                    try:
                        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
                    except:
                        logger.warning("Nie można dodać TorchVersion do bezpiecznych globali")
                except:
                    logger.warning("Nie można zaimportować torch.serialization")
                
                # Próba ładowania z weights_only=False (mniej bezpieczna, ale bardziej kompatybilna)
                state = torch.load(path, weights_only=False, map_location="cpu")
                logger.info("Załadowano model z weights_only=False")
            except Exception as e:
                # Jeśli nie działa, próba z weights_only=True
                try:
                    state = torch.load(path, weights_only=True, map_location="cpu")
                    logger.info("Załadowano model z weights_only=True")
                except Exception as e2:
                    logger.error(f"Błąd ładowania modelu: {str(e2)}")
                    # W przypadku problemów z ładowaniem, zachowujemy model z metadanymi
                    logger.warning("Używanie modelu tylko z metadanych, bez wag warstw")
                    model.eval()
                    return model, class_names
            
            # Ładuj stan modelu bezpiecznie
            if 'model_state_dict' in state:
                state_dict = state['model_state_dict']
            else:
                state_dict = state
            
            # Obsługa niekompatybilności kluczy z torch.compile()
            model_state_dict = model.state_dict()
            
            # Pobieranie listy kluczy z bieżącego i załadowanego stanu
            current_keys = set(model_state_dict.keys())
            loaded_keys = set(state_dict.keys())
            
            # Sprawdzanie i logowanie różnic
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys
            
            if missing_keys or unexpected_keys:
                logger.warning(f"Wykryto różnice w kluczach stanu modelu. Brakujące: {len(missing_keys)}, nieoczekiwane: {len(unexpected_keys)}")
                logger.debug(f"Przykłady brakujących kluczy: {list(missing_keys)[:3] if missing_keys else 'brak'}")
                logger.debug(f"Przykłady nieoczekiwanych kluczy: {list(unexpected_keys)[:3] if unexpected_keys else 'brak'}")
                
                # Jeśli mamy nieoczekiwane klucze z 'models.X.resnet' zamiast 'models.X._orig_mod.resnet'
                # to możemy spróbować mapować je
                remapped_state_dict = {}
                for key in state_dict:
                    if key in current_keys:
                        # Klucz pasuje bezpośrednio
                        remapped_state_dict[key] = state_dict[key]
                    elif '_orig_mod' in key:
                        # Klucz zawiera _orig_mod, ale może nie być w bieżącym stanie
                        plain_key = key.replace('_orig_mod.', '')
                        if plain_key in current_keys:
                            remapped_state_dict[plain_key] = state_dict[key]
                    elif any(key.startswith(f'models.{ft}.') for ft in model.feature_types):
                        # Klucz bez _orig_mod, ale może być potrzebny z _orig_mod
                        for ft in model.feature_types:
                            if key.startswith(f'models.{ft}.'):
                                new_key = key.replace(f'models.{ft}.', f'models.{ft}._orig_mod.')
                                if new_key in current_keys:
                                    remapped_state_dict[new_key] = state_dict[key]
                
                # Użyj zmapowanych kluczy
                if remapped_state_dict:
                    logger.info(f"Udało się zmapować {len(remapped_state_dict)} kluczy")
                    try:
                        missing_after_remap = [k for k in current_keys if k not in remapped_state_dict]
                        logger.debug(f"Brakujące klucze po remapowaniu: {len(missing_after_remap)}")
                        # Załaduj dostępne klucze (strict=False)
                        model.load_state_dict(remapped_state_dict, strict=False)
                        logger.info("Załadowano częściowy stan modelu po remapowaniu kluczy")
                    except Exception as e:
                        logger.error(f"Błąd podczas ładowania zmapowanego stanu: {str(e)}")
                        # Kontynuuj z modelem bez ładowania stanu
                else:
                    logger.warning("Nie udało się zmapować kluczy, używanie domyślnych wag modelu")
            else:
                # Standardowe ładowanie, gdy klucze są zgodne
                try:
                    model.load_state_dict(state_dict)
                    logger.info("Pomyślnie załadowano stan modelu ensemble")
                except Exception as e:
                    logger.error(f"Błąd podczas standardowego ładowania stanu: {str(e)}")
                    # Ładowanie z strict=False, aby załadować dostępne parametry
                    model.load_state_dict(state_dict, strict=False)
                    logger.warning("Załadowano stan częściowo (strict=False)")
            
            model.eval()
            return model, class_names
            
        except Exception as main_error:
            logger.error(f"Błąd podczas ładowania modelu z użyciem metadanych: {str(main_error)}")
            # Jeśli nie udało się załadować z metadanych, spróbuj starą metodą
            try:
                # Próba z weights_only=False (mniej bezpieczna)
                state = torch.load(path, weights_only=False, map_location="cpu")
                
                model = cls(
                    models_dict=models_dict,
                    temperature=state.get('temperature', 1.0),
                    regularization_strength=state.get('regularization_strength', 0.01)
                )
                
                # Ładuj stan modelu bezpiecznie
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                
                model.eval()
                logger.info("Załadowano model awaryjnie")
                return model, state.get('class_names')
                
            except Exception as e:
                logger.error(f"Wszystkie próby ładowania modelu nie powiodły się: {str(e)}")
                
                # Stwórz nowy model, gdy ładowanie kompletnie zawiedzie
                logger.warning("Tworzenie nowego modelu ensemble z domyślnymi wagami")
                model = cls(
                    models_dict=models_dict,
                    temperature=1.0,
                    regularization_strength=0.01
                )
                model.eval()
                return model, None

import numpy as np
import librosa
import torch
import json
import io
import soundfile as sf
from typing import Any, Union
from pathlib import Path

from app.core.config import settings

class AudioFeatureExtractor:
    """
    Ekstraktor cech audio zgodny z pipeline treningowym.
    Obsługuje: melspectrogram, mfcc, chroma.
    """
    def __init__(
        self,
        feature_type: str = settings.DEFAULT_FEATURE_TYPE,
        normalization_dir: str | Path = settings.DEFAULT_STATS_PATH
    ):
        """
        Ekstraktor cech audio zgodny z pipeline treningowym.
        Obsługuje: melspectrogram, mfcc, chroma.
        
        Args:
            feature_type: Typ ekstrakcji cechy
            normalization_dir: Katalog z plikami statystyk
        """
        self.feature_type = feature_type
        
        # Konwersja do Path i pobranie katalogu nadrzędnego
        self.normalization_dir = Path(normalization_dir).parent
        
        # Dodatkowe wyświetlenie informacji diagnostycznych
        print(f"Inicjalizacja AudioFeatureExtractor")
        print(f"Typ cechy: {feature_type}")
        print(f"Katalog statystyk: {self.normalization_dir}")
        print(f"Pełna ścieżka: {self.normalization_dir.resolve()}")
        
        # Wczytaj statystyki
        self.normalization_stats = self._load_normalization_stats()
        
        # Parametry muszą być identyczne jak w treningu!
        self.extraction_params = {
            'melspectrogram': {
                'n_fft': settings.N_FFT,
                'hop_length': settings.HOP_LENGTH,
                'n_mels': settings.N_MELS,
                'target_length': settings.TARGET_LENGTH
            },
            'mfcc': {
                'n_mfcc': settings.N_MFCC,
                'n_fft': settings.N_FFT,
                'hop_length': settings.HOP_LENGTH,
                'target_length': settings.TARGET_LENGTH
            },
            'chroma': {
                'n_fft': settings.N_FFT,
                'hop_length': settings.HOP_LENGTH,
                'n_chroma': settings.N_CHROMA,
                'target_length': settings.TARGET_LENGTH
            }
        }

    def _load_normalization_stats(self) -> dict:
        """
        Wczytaj statystyki normalizacyjne dla danego typu cechy.
        """
        stats_path = self.normalization_dir / f"{self.feature_type}_stats.json"
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Brak pliku statystyk: {stats_path}")
        
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        # Ekstrakcja mean i std z pliku o złożonej strukturze
        # Dla melspektrogramu używamy channel_means i channel_stds
        if self.feature_type == 'melspectrogram':
            # Próba pobrania wartości z różnych możliwych kluczy
            mean_candidates = [
                stats.get('channel_means', [0])[0],
                stats.get('mean', 0),
                stats.get('normalized_mean', 0)
            ]
            std_candidates = [
                stats.get('channel_stds', [1])[0],
                stats.get('std', 1),
                stats.get('normalized_std', 1)
            ]
            
            # Wybierz pierwszą niepustą wartość
            mean = next((m for m in mean_candidates if m is not None), 0)
            std = next((s for s in std_candidates if s is not None), 1)
            
            # Rozszerz mean i std do rozmiaru cechy (128 pasm mel)
            mean = np.full(128, mean)
            std = np.full(128, std)
        else:
            # Dla innych typów cech (mfcc, chroma)
            mean = np.array([stats.get('mean', 0)])
            std = np.array([stats.get('std', 1)])
        
        # Zabezpieczenie przed zerowymi wartościami std
        std = np.where(std == 0, 1, std)
        
        return {
            "mean": mean, 
            "std": std
        }

    def _load_audio_data(
        self,
        audio_input: Union[str, bytes, np.ndarray, io.BytesIO, torch.Tensor, Any],
        sr: int | None = None
    ) -> tuple[np.ndarray, int]:
        """
        Uniwersalne wczytywanie audio (ścieżka, bytes, numpy, tensor, io.BytesIO).
        Zwraca: (audio_array, sample_rate)
        """
        # PyTorch tensor
        if isinstance(audio_input, torch.Tensor):
            audio_input = audio_input.squeeze().cpu().numpy()
        # NumPy array
        if isinstance(audio_input, np.ndarray):
            return audio_input, sr or settings.DEFAULT_SAMPLE_RATE
        # Ścieżka do pliku
        if isinstance(audio_input, (str, Path)):
            audio_array, file_sr = librosa.load(str(audio_input), sr=sr or settings.DEFAULT_SAMPLE_RATE, mono=True)
            return audio_array, file_sr
        # Bytes
        if isinstance(audio_input, bytes):
            audio_input = io.BytesIO(audio_input)
        # io.BytesIO
        if isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            try:
                audio_array, file_sr = sf.read(audio_input)
            except Exception:
                audio_input.seek(0)
                audio_array, file_sr = librosa.load(audio_input, sr=sr or settings.DEFAULT_SAMPLE_RATE, mono=True)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array, file_sr
        # Ostatnia próba: librosa
        try:
            audio_array, file_sr = librosa.load(audio_input, sr=sr or settings.DEFAULT_SAMPLE_RATE, mono=True)
            return audio_array, file_sr
        except Exception as e:
            raise ValueError(f"Cannot process audio file: {str(e)}")

    def extract_feature(
        self,
        audio_data: Union[str, bytes, np.ndarray, io.BytesIO, torch.Tensor, Any],
        sr: int = settings.DEFAULT_SAMPLE_RATE
    ) -> torch.Tensor:
        """
        Ekstrakcja i normalizacja cechy audio (zgodnie z treningiem).
        Zwraca: torch.Tensor shape [1, 1, freq, time]
        """
        try:
            # 1. Wczytaj audio
            audio_array, file_sr = self._load_audio_data(audio_data, sr)
            
            # 2. Resample jeśli potrzeba
            if file_sr != sr:
                audio_array = librosa.resample(audio_array, orig_sr=file_sr, target_sr=sr)
            
            # 3. Ustal parametry ekstrakcji
            params = self.extraction_params[self.feature_type]
            target_length = params['target_length']
            
            # Bezpieczne parametry ekstrakcji
            n_fft = min(params.get('n_fft', 2048), 4096)
            hop_length = params.get('hop_length', 512)
            
            # 4. Kontrola długości sygnału
            max_audio_length = 10 * sr  # Maksymalnie 10 sekund
            audio_array = audio_array[:max_audio_length]
            
            # 5. Ekstrakcja cechy z zabezpieczeniami
            try:
                if self.feature_type == 'melspectrogram':
                    feature = librosa.feature.melspectrogram(
                        y=audio_array,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=min(params.get('n_mels', 128), 256)
                    )
                    feature = librosa.power_to_db(feature)
                elif self.feature_type == 'mfcc':
                    feature = librosa.feature.mfcc(
                        y=audio_array,
                        sr=sr,
                        n_mfcc=min(params.get('n_mfcc', 13), 40),
                        n_fft=n_fft,
                        hop_length=hop_length
                    )
                elif self.feature_type == 'chroma':
                    feature = librosa.feature.chroma_stft(
                        y=audio_array,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length
                    )
                else:
                    raise ValueError(f"Unknown feature type: {self.feature_type}")
            except Exception as e:
                print(f"Błąd ekstrakcji cechy: {e}")
                print(f"Parametry: n_fft={n_fft}, hop_length={hop_length}, audio_length={len(audio_array)}")
                raise
            
            # 6. Dopasuj długość cechy (time axis)
            if feature.shape[1] > target_length:
                feature = feature[:, :target_length]
            elif feature.shape[1] < target_length:
                pad_width = target_length - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            
            # 7. Normalizacja z zaawansowaną obsługą kształtów
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']
            
            # Dopasowanie kształtów mean i std do cechy
            if mean.ndim == 0:
                mean = np.full(feature.shape[0], mean)
            if std.ndim == 0:
                std = np.full(feature.shape[0], std)
            
            # Przycinanie lub dopełnianie mean i std
            if mean.shape[0] > feature.shape[0]:
                mean = mean[:feature.shape[0]]
                std = std[:feature.shape[0]]
            elif mean.shape[0] < feature.shape[0]:
                mean = np.pad(mean, (0, feature.shape[0] - mean.shape[0]), mode='constant')
                std = np.pad(std, (0, feature.shape[0] - std.shape[0]), mode='constant')
            
            # Normalizacja z zabezpieczeniami
            mean = mean.reshape(-1, 1)
            std = std.reshape(-1, 1)
            feature = (feature - mean) / (std + 1e-8)
            
            # 8. Konwersja do tensora [1, 1, freq, time]
            feature = torch.from_numpy(feature).float().unsqueeze(0).unsqueeze(0)
            
            # Dodatkowa diagnostyka
            print(f"Extracted feature shape: {feature.shape}")
            
            return feature
        
        except Exception as e:
            print(f"Całkowity błąd ekstrakcji: {e}")
            raise

def prepare_audio_features(
    audio_array: np.ndarray,
    sample_rate: int,
    required_features: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """
    Przygotuj cechy audio dla modelu (zgodnie z treningiem).
    
    Args:
        audio_array: Tablica NumPy z sygnałem audio
        sample_rate: Częstotliwość próbkowania
        required_features: Lista wymaganych typów cech
    
    Returns:
        Słownik z cechami audio
    """
    if required_features is None:
        required_features = [settings.DEFAULT_FEATURE_TYPE]
    
    extractor = AudioFeatureExtractor(feature_type=required_features[0])
    features = extractor.extract_feature(audio_array, sr=sample_rate)
    return {required_features[0]: features}
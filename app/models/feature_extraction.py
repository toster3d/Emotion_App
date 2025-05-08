import numpy as np
import librosa
import torch
import logging

# Ustawienie loggera
logger = logging.getLogger(__name__)

def normalize_audio(audio_array):
    """
    Normalizuje sygnał audio do zakresu [-1, 1].
    
    Args:
        audio_array: Sygnał audio w formie tablicy numpy
        
    Returns:
        Znormalizowany sygnał audio
    """
    # Sprawdź czy sygnał nie jest już znormalizowany
    max_val = np.max(np.abs(audio_array))
    if max_val > 1.0:
        logger.info(f"Normalizacja sygnału audio. Maksymalna amplituda przed: {max_val}")
        normalized = audio_array / max_val
        logger.info(f"Maksymalna amplituda po normalizacji: {np.max(np.abs(normalized))}")
        return normalized
    return audio_array

def extract_features(audio_array, sr, feature_type, max_length=3.0, 
                     n_mels=128, n_mfcc=40, n_chroma=12, 
                     n_fft=2048, hop_length=512, normalize=True):
    """Ekstrakcja różnych cech z sygnału audio.
    
    Args:
        audio_array: Sygnał audio w formie tablicy numpy
        sr: Częstotliwość próbkowania
        feature_type: Typ cechy do ekstrakcji
        max_length: Maksymalna długość sygnału w sekundach
        n_mels: Liczba pasm melowych dla melspektrogramu
        n_mfcc: Liczba współczynników MFCC
        n_chroma: Liczba pasm chromatycznych
        n_fft: Długość okna dla krótkoterminowej transformaty Fouriera
        hop_length: Przesunięcie okna między kolejnymi ramkami
        normalize: Czy normalizować wynikowe cechy
        
    Returns:
        Wyekstrahowane cechy w formie tablicy numpy
    """
    # Normalizacja sygnału audio przed ekstrakcją cech
    audio_array = normalize_audio(audio_array)
    
    # Ustalenie docelowej długości sygnału
    target_length = int(max_length * sr)
    logger.debug(f"Przetwarzanie sygnału: typ={feature_type}, długość={len(audio_array)}, docelowa długość={target_length}")
    
    if len(audio_array) > target_length:
        logger.debug(f"Przycinanie sygnału z {len(audio_array)} do {target_length} próbek")
        audio_array = audio_array[:target_length]
    else:
        padding = np.zeros(target_length - len(audio_array))
        logger.debug(f"Uzupełnianie sygnału z {len(audio_array)} do {target_length} próbek")
        audio_array = np.concatenate([audio_array, padding])
    
    feature = None
    
    if feature_type == "melspectrogram":
        # Ekstrakcja melspektrogramu
        logger.debug(f"Ekstrakcja melspektrogramu: n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
        S = librosa.feature.melspectrogram(
            y=audio_array, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
        feature = librosa.power_to_db(S, ref=np.max)
    
    elif feature_type == "spectrogram":
        # Obliczanie standardowego spektrogramu
        logger.debug(f"Ekstrakcja spektrogramu: n_fft={n_fft}, hop_length={hop_length}")
        D = np.abs(librosa.stft(audio_array, n_fft=n_fft, hop_length=hop_length))
        feature = librosa.amplitude_to_db(D, ref=np.max)
    
    elif feature_type == "mfcc":
        # Obliczanie MFCC (Mel-frequency cepstral coefficients)
        logger.debug(f"Ekstrakcja MFCC: n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}")
        feature = librosa.feature.mfcc(
            y=audio_array, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
    
    elif feature_type == "chroma":
        # Obliczanie chromagramu
        logger.debug(f"Ekstrakcja chromagramu: n_chroma={n_chroma}, n_fft={n_fft}, hop_length={hop_length}")
        feature = librosa.feature.chroma_stft(
            y=audio_array, sr=sr, n_chroma=n_chroma,
            n_fft=n_fft, hop_length=hop_length
        )
    
    elif feature_type == "spectral_contrast":
        # Obliczanie spektralnego kontrastu
        feature = librosa.feature.spectral_contrast(
            y=audio_array, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
    
    elif feature_type == "zcr":
        # Obliczanie Zero Crossing Rate
        feature = librosa.feature.zero_crossing_rate(
            audio_array, hop_length=hop_length
        )
        # Rozszerzanie wymiaru dla ZCR
        expanded = np.zeros((n_mels, feature.shape[1]))
        normalized_feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-8)
        for i in range(n_mels):
            scale_factor = 1.0 - (i / float(n_mels))
            expanded[i, :] = normalized_feature * scale_factor
        feature = expanded

    elif feature_type == "rms":
        # Obliczanie RMS Energy
        feature = librosa.feature.rms(
            y=audio_array, hop_length=hop_length
        )
        # Rozszerzanie wymiaru dla RMS
        expanded = np.zeros((n_mels, feature.shape[1]))
        normalized_feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + 1e-8)
        for i in range(n_mels):
            scale_factor = np.exp(-3.0 * (i / float(n_mels)))
            expanded[i, :] = normalized_feature * scale_factor
        feature = expanded

    elif feature_type == "tempogram":
        # Obliczanie tempogramu
        feature = librosa.feature.tempogram(
            y=audio_array, sr=sr, hop_length=hop_length
        )
    
    elif feature_type == "tonnetz":
        # Obliczanie Tonnetz - harmonicznych relacji
        y_harm = librosa.effects.harmonic(audio_array, margin=4.0)
        chroma = librosa.feature.chroma_cqt(
            y=y_harm, sr=sr, hop_length=hop_length
        )
        feature = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    
    elif feature_type == "delta_mfcc":
        # Obliczanie Delta MFCC - zmian w MFCC
        mfccs = librosa.feature.mfcc(
            y=audio_array, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        feature = librosa.feature.delta(mfccs)
    
    elif feature_type == "delta_tempogram":
        # Obliczanie Delta Tempogram - zmian w tempie
        tempogram = librosa.feature.tempogram(
            y=audio_array, sr=sr, hop_length=hop_length
        )
        feature = librosa.feature.delta(tempogram)
    
    else:
        raise ValueError(f"Nieznany typ cechy: {feature_type}")
    
    if feature is None:
        logger.error(f"Nie udało się wyodrębnić cechy typu {feature_type}")
        raise ValueError(f"Ekstrakcja cechy {feature_type} zwróciła None")
    
    logger.debug(f"Wyodrębniona cecha {feature_type} ma kształt {feature.shape}")
    
    # Normalizacja cech (opcjonalna)
    if normalize and feature is not None:
        if feature_type in ["mfcc", "delta_mfcc"]:
            # Normalizacja MFCC - zaimplementowana
            logger.debug(f"Normalizacja MFCC za pomocą librosa.util.normalize")
            feature = librosa.util.normalize(feature)
        elif feature_type in ["melspectrogram", "spectrogram"]:
            # Spektrogramy - przekształcone do dB
            logger.debug(f"Spektrogram już znormalizowany do dB")
            pass
        else:
            # Standardowa normalizacja min-max dla pozostałych cech
            feature_min = np.min(feature)
            feature_max = np.max(feature)
            logger.debug(f"Normalizacja min-max: min={feature_min}, max={feature_max}")
            if feature_max > feature_min:
                feature = (feature - feature_min) / (feature_max - feature_min)
    
    return feature

def prepare_audio_features(audio_array, sr, required_features=None):
    """
    Przygotowuje cechy audio dla modelu ensemble.
    
    Args:
        audio_array: Sygnał audio w formie tablicy numpy
        sr: Częstotliwość próbkowania
        required_features: Lista wymaganych typów cech (jeśli None, używa wszystkich podstawowych)
        
    Returns:
        Dict: Słownik tensorów cech w formacie {typ_cechy: tensor}
    """
    logger.info(f"Przygotowanie cech audio: sample rate={sr}, długość={len(audio_array)}")
    
    # Preprocessing audio - dodajemy preprocessing przed ekstrakcją cech
    # Zastosowanie preemfazy dla uwydatnienia wyższych częstotliwości
    preemphasis_coef = 0.97
    emphasized_audio = np.append(audio_array[0], audio_array[1:] - preemphasis_coef * audio_array[:-1])
    logger.info(f"Zastosowano preemfazę z współczynnikiem {preemphasis_coef}")
    
    if required_features is None:
        required_features = ["melspectrogram", "mfcc", "chroma"]
    
    logger.info(f"Wymagane cechy: {required_features}")
    
    # Ekstrakcja wymaganych cech
    features_dict = {}
    for feature_type in required_features:
        try:
            # Ekstrakcja cechy
            logger.info(f"Rozpoczęcie ekstrakcji cechy: {feature_type}")
            feature_array = extract_features(emphasized_audio, sr, feature_type)
            
            # Konwersja do tensora
            feature_tensor = torch.tensor(feature_array, dtype=torch.float32).unsqueeze(0)  # Dodaj wymiar batcha
            
            # Dodaj wymiar kanału dla zgodności z modelem (N, C, H, W)
            feature_tensor = feature_tensor.unsqueeze(1)
            
            logger.info(f"Cecha {feature_type} przekształcona do tensora o kształcie {feature_tensor.shape}")
            
            # Dodatkowa standaryzacja tensora
            if feature_type in ["mfcc", "melspectrogram"]:
                # Standaryzacja dla pełnego tensora
                mean = torch.mean(feature_tensor)
                std = torch.std(feature_tensor)
                if std > 0:
                    feature_tensor = (feature_tensor - mean) / std
                    logger.info(f"Standaryzacja tensora {feature_type}: mean={mean:.4f}, std={std:.4f}")
            
            # Zapisanie w słowniku
            features_dict[feature_type] = feature_tensor
        except Exception as e:
            logger.error(f"Błąd podczas ekstrakcji cechy {feature_type}: {str(e)}")
            raise
    
    logger.info(f"Pomyślnie przygotowano {len(features_dict)} cech audio")
    
    return features_dict 
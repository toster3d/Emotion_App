import numpy as np
import librosa
import logging
from numpy.typing import NDArray
from app.core.settings import settings
from typing import TypedDict


logger = logging.getLogger(__name__)

class AudioFeatures(TypedDict):
    melspectrogram: NDArray[np.float32]

def prepare_audio_features(
    audio_array: NDArray[np.float32],
    sample_rate: int,
    max_length: float = settings.MAX_AUDIO_LENGTH
) -> AudioFeatures:
    """
    Extract features from an audio signal
    
    Args:
        audio_array: Audio signal as a numpy array
        sample_rate: Sample rate of the audio
        max_length: Maximum length of audio to process in seconds
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        # Preprocess audio: trim and normalize
        audio_array = preprocess_audio(audio_array, sample_rate, max_length)
        
        # Extract MEL spectrogram
        mel_spectrogram: NDArray[np.float32] = extract_melspectrogram(audio_array, sample_rate)
        
        # Add batch and channel dimensions for model input
        mel_spectrogram = mel_spectrogram.reshape(1, 1, mel_spectrogram.shape[0], mel_spectrogram.shape[1])
        
        return {
            'melspectrogram': mel_spectrogram
        }
        
    except Exception as e:
        logger.error(f"Error extracting features from audio: {str(e)}")
        raise
    
def preprocess_audio(
    audio_array: NDArray[np.float32],
    sample_rate: int,
    max_length: float = settings.MAX_AUDIO_LENGTH
) -> NDArray[np.float32]:
    """
    Preprocess the audio signal for feature extraction
    
    Args:
        audio_array: Audio signal as a numpy array
        sample_rate: Sample rate of the audio
        max_length: Maximum length of audio to process in seconds
        
    Returns:
        Preprocessed audio signal
    """
    # Convert to mono if needed
    if len(audio_array.shape) > 1:
        audio_array = librosa.to_mono(audio_array) # type: ignore
    
    # Trim leading and trailing silence
    audio_array, _ = librosa.effects.trim(audio_array, top_db=20) # type: ignore
    
    # Apply amplitude normalization
    audio_array = librosa.util.normalize(audio_array) # type: ignore
    
    # Ensure consistent length
    target_length = int(max_length * sample_rate)
    if len(audio_array) > target_length:
        # Cut to max_length
        audio_array = audio_array[:target_length]
    else:
        # Pad with zeros
        padding = np.zeros(target_length - len(audio_array), dtype=np.float32)
        audio_array = np.concatenate([audio_array, padding])
    
    return audio_array

def extract_melspectrogram(
    audio_array: NDArray[np.float32],
    sample_rate: int
) -> NDArray[np.float32]:
    """
    Extract mel spectrogram from audio signal
    
    Args:
        audio_array: Preprocessed audio signal
        sample_rate: Sample rate of the audio
        
    Returns:
        Mel spectrogram as a 2D numpy array
    """
    # Extract mel spectrogram
    S = librosa.feature.melspectrogram( # type: ignore
        y=audio_array, 
        sr=sample_rate,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
        n_mels=settings.N_MELS
    )
    
    # Convert to dB scale
    S_db = librosa.power_to_db(S, ref=np.max) # type: ignore
    
    # Normalize
    mean, std = np.mean(S_db), np.std(S_db) # type: ignore
    S_db = (S_db - mean) / std
    
    return S_db.astype(np.float32)
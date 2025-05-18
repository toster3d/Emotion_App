from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import librosa
import numpy as np
import uuid
import os
import sys
import logging
from pathlib import Path
import joblib
import torch
import torch.nn as nn

logging.basicConfig(level=logging.DEBUG)

# Dołączamy ścieżkę do plików pomocniczych
sys.path.append(os.path.abspath(r"C:\Users\Dell\Desktop\STUDIA\zajecia\Seminarium\aplikacja\helpers"))
from resnet_model_definition import AudioResNet

# Stworzenie folderów na pliki tymczasowe
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# Stworzenie folderów na pliki statyczne
STATIC_DIR = Path("static")
CSS_DIR = STATIC_DIR / "css"
JS_DIR = STATIC_DIR / "js"

CSS_DIR.mkdir(exist_ok=True, parents=True)
JS_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(title="Klasyfikator emocji z plików audio")

# Konfiguracja szablonów i plików statycznych
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Wczytanie modelu
def load_model(model_path: str, device: torch.device, num_classes: int = 6, dropout_rate: float = 0.5):
    """
    Funkcja do załadowania modelu z pliku.
    
    model_path: Ścieżka do pliku .pt z wagami
    device: Urządzenie (CPU lub GPU)
    num_classes: Liczba klas w klasyfikacji
    dropout_rate: Wartość dropout dla modelu
    """
    # Inicjalizacja modelu
    model = AudioResNet(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Załadowanie wag z pliku
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Przełączenie modelu w tryb ewaluacji
    model = model.to(device)
    model.eval()  # Ważne, by włączyć tryb ewaluacji (np. dla dropout, batch norm itp.)
    
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "model/best_model_20250406_160046.pt"  # Ścieżka do Twojego pliku z wagami
model = load_model(model_path, device)

def extract_features(file_path, model_type):
    try:
        # Wczytanie pliku audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Ujednolicenie długości
        MAX_LENGTH = 5  # Ustawienie wartości MAX_LENGTH zgodnie z oryginalnym modelem
        target_length = int(MAX_LENGTH * sr)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            padding = np.zeros(target_length - len(y))
            y = np.concatenate([y, padding])
        
        # Ekstrakcja melspektrogramu
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Przygotowanie jak w oryginalnym kodzie
        features = np.array([S_db])
        features = features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])
        
        # Normalizacja danych (standardyzacja)
        # Tutaj wczytujemy wcześniej zapisane wartości mean i std z treningu
        mean = np.load('model/mean.npy')
        std = np.load('model/std.npy')
        features = (features - mean) / std
        
        # Zwróć przetworzone cechy w odpowiednim formacie
        if model_type == "pt":
            return features
        elif model_type == "h5":
            return features
        else:
            # Dla klasycznych modeli ML - spłaszcz dane
            return features.reshape(1, -1)
            
    except Exception as e:
        raise Exception(f"Błąd ekstrakcji cech: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Strona główna aplikacji"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Wczytanie pliku audio
        contents = await file.read()
       
        # Zapisanie tymczasowo pliku, aby użyć go z librosa
        temp_file = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
        with open(temp_file, "wb") as f:
            f.write(contents)
       
        # Przetwarzanie audio zgodnie z metodą extract_features
        features = extract_features(str(temp_file), "pt")
        features_tensor = torch.FloatTensor(features).to(device)
       
        # Predykcja
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
       
        # Wczytanie enkodera etykiet dla interpretacji wyników
        label_encoder = joblib.load('model/label_encoder.pkl')
        emotion_labels = list(label_encoder.classes_)
       
        # Usunięcie tymczasowego pliku
        os.remove(temp_file)
       
        return {
            "emotion": emotion_labels[predicted_class],
            "confidence": confidence,
            "all_emotions": {
                emotion: float(prob) for emotion, prob in zip(emotion_labels, probabilities[0].tolist())
            }
        }
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Uruchomienie aplikacji
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
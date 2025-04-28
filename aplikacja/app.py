from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import librosa
import numpy as np
import uuid
import os
import sys
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.abspath(r"C:\Users\Dell\Desktop\STUDIA\zajecia\Seminarium\aplikacja\heplers"))
from resnet_model_definition import AudioResNet

# Stworzenie folderów na pliki tymczasowe
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    
def predict_emotion(file_path):
    try:
        model, model_type = load_model()
        features = extract_features(file_path, model_type)
        
        # Wczytaj enkoder etykiet
        label_encoder = joblib.load('model/label_encoder.pkl')
        
        # Przewidywanie
        if model_type == "pytorch":
            import torch
            with torch.no_grad():
                tensor = torch.FloatTensor(features)
                outputs = model(tensor)
                # Konwersja predykcji na prawdopodobieństwa (Softmax)
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(outputs).numpy()[0]
        elif model_type == "h5":
            probs = model.predict(features)[0]
        else:
            # Dla modelu ML
            probs = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
        
        # Uzyskaj oryginalne nazwy klas z enkodera
        emotion_labels = list(label_encoder.classes_)
        
        if probs is not None:
            # Dla modeli z prawdopodobieństwami
            result = {
                "emotion": emotion_labels[np.argmax(probs)],
                "confidence": float(np.max(probs)),
                "all_emotions": {emotion: float(prob) for emotion, prob in zip(emotion_labels, probs)}
            }
        else:
            # Dla modeli bez prawdopodobieństw
            prediction = model.predict(features)[0]
            result = {
                "emotion": emotion_labels[prediction],
                "confidence": 1.0,
                "all_emotions": {emotion: 1.0 if i == prediction else 0.0 
                                for i, emotion in enumerate(emotion_labels)}
            }
        
        return result
    
    except Exception as e:
        raise Exception(f"Błąd przewidywania: {str(e)}")
    
@app.get("/", response_class=HTMLResponse)
async def root():
    # Zwróć stronę główną
    html_content = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klasyfikator emocji z plików audio</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 800px;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        .recorder-container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .upload-container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
        }
        #results {
            margin-top: 30px;
        }
        .btn-record {
            background-color: #dc3545;
            color: white;
        }
        .btn-stop {
            background-color: #0d6efd;
            color: white;
        }
        .emotion-chart {
            height: 300px;
            margin-top: 20px;
        }
        .emotion-bar {
            height: 25px;
            margin-bottom: 10px;
            background-color: #6c757d;
            border-radius: 5px;
            transition: width 0.5s ease-in-out;
        }
        .emotion-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .timer {
            font-size: 1.5rem;
            font-weight: bold;
            color: #dc3545;
            margin: 15px 0;
        }
        .result-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            background-color: #f8f9fa;
            display: none;
        }
        .main-emotion {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 20px;
            color: #0d6efd;
            text-align: center;
        }
        /* Nowy styl dla paska postępu czasu nagrywania */
        .recording-progress {
            height: 10px;
            margin-top: 10px;
            transition: width 0.1s linear;
            background-color: #dc3545;
        }
        .time-info {
            font-size: 0.9rem;
            color: #6c757d;
            text-align: center;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Klasyfikator emocji z plików audio</h1>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="record-tab" data-bs-toggle="tab" data-bs-target="#record" type="button" role="tab">Nagrywanie</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload" type="button" role="tab">Wgrywanie pliku</button>
            </li>
        </ul>
        
        <div class="tab-content mt-3" id="myTabContent">
            <!-- Zakładka nagrywania -->
            <div class="tab-pane fade show active" id="record" role="tabpanel">
                <div class="recorder-container">
                    <div class="text-center">
                        <button id="recordButton" class="btn btn-record">Rozpocznij nagrywanie</button>
                        <button id="stopButton" class="btn btn-stop" disabled>Zatrzymaj nagrywanie</button>
                    </div>
                    <div class="timer text-center" id="recordingTimer">00:05</div>
                    <div class="progress">
                        <div id="recordingProgress" class="progress-bar recording-progress" role="progressbar" style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    <div class="time-info">Maksymalny czas nagrania: 5 sekund</div>
                    <div class="text-center">
                        <audio id="audioPlayback" controls style="display: none; width: 100%; margin-top: 15px;"></audio>
                    </div>
                    <div class="text-center mt-3">
                        <button id="analyzeRecording" class="btn btn-primary" disabled>Analizuj nagranie</button>
                    </div>
                </div>
            </div>
            
            <!-- Zakładka wgrywania pliku -->
            <div class="tab-pane fade" id="upload" role="tabpanel">
                <div class="upload-container">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="audioFile" class="form-label">Wybierz plik audio (WAV, MP3, OGG)</label>
                            <input class="form-control" type="file" id="audioFile" accept=".wav,.mp3,.ogg">
                            <div class="time-info mt-2">Uwaga: Nagranie zostanie przycięte do 5 sekund podczas analizy</div>
                        </div>
                        <div class="text-center">
                            <button type="submit" class="btn btn-primary">Analizuj plik</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <!-- Wyniki analizy -->
        <div class="result-card" id="resultsCard">
            <div class="main-emotion" id="mainEmotion">Rozpoznana emocja</div>
            <div id="emotionBars"></div>
        </div>
        
        <div id="errorMsg" class="alert alert-danger mt-3" style="display: none;"></div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Zmienne globalne
        let mediaRecorder;
        let audioChunks = [];
        let audioBlob;
        let recordingInterval;
        let seconds = 5; // Startujemy od 5 sekund i odliczamy w dół
        const MAX_RECORDING_TIME = 5; // Maksymalny czas nagrywania w sekundách
        
        // Elementy DOM
        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');
        const audioPlayback = document.getElementById('audioPlayback');
        const analyzeButton = document.getElementById('analyzeRecording');
        const uploadForm = document.getElementById('uploadForm');
        const resultsCard = document.getElementById('resultsCard');
        const mainEmotion = document.getElementById('mainEmotion');
        const emotionBars = document.getElementById('emotionBars');
        const errorMsg = document.getElementById('errorMsg');
        const recordingTimer = document.getElementById('recordingTimer');
        const recordingProgress = document.getElementById('recordingProgress');
        
        // Obsługa nagrywania
        recordButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', stopRecording);
        analyzeButton.addEventListener('click', analyzeRecording);
        uploadForm.addEventListener('submit', handleUpload);
        
        // Rozpocznij nagrywanie
        async function startRecording() {
            try {
                audioChunks = [];
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    audioPlayback.style.display = 'block';
                    analyzeButton.disabled = false;
                };
                
                // Start recording
                mediaRecorder.start();
                recordButton.disabled = true;
                stopButton.disabled = false;
                
                // Start timer (odliczanie w dół)
                seconds = MAX_RECORDING_TIME;
                updateTimerDisplay();
                recordingProgress.style.width = '0%';
                
                recordingInterval = setInterval(() => {
                    seconds--;
                    // Aktualizuj timer i pasek postępu
                    updateTimerDisplay();
                    const progressPercentage = ((MAX_RECORDING_TIME - seconds) / MAX_RECORDING_TIME) * 100;
                    recordingProgress.style.width = progressPercentage + '%';
                    
                    // Automatycznie zatrzymaj nagrywanie po upływie maksymalnego czasu
                    if (seconds <= 0) {
                        stopRecording();
                    }
                }, 1000);
                
            } catch (err) {
                console.error('Błąd podczas nagrywania:', err);
                showError('Nie można uzyskać dostępu do mikrofonu.');
            }
        }
        
        // Aktualizacja wyświetlania timera (teraz odliczanie w dół)
        function updateTimerDisplay() {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            recordingTimer.textContent = `00:${remainingSeconds.toString().padStart(2, '0')}`;
        }
        
        // Zatrzymaj nagrywanie
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                clearInterval(recordingInterval);
                
                // Zatrzymaj wszystkie ścieżki
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                recordButton.disabled = false;
                stopButton.disabled = true;
                recordingProgress.style.width = '100%';
                recordingTimer.textContent = '00:00';
            }
        }
        
        // Analizuj nagranie
        async function analyzeRecording() {
            try {
                hideError();
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.wav');

                console.log("Wysyłanie nagrania do analizy...");
                
                // Wyślij plik do analizy
                const response = await fetch('/predict/', {
                    method: 'POST',
                    body: formData
                });

                console.log("Status odpowiedzi:", response.status);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("Błąd serwera:", errorText);
                    throw new Error(`Błąd serwera: ${response.status}`);
                }
                
                const result = await response.json();
                displayResults(result);
                
            } catch (err) {
                console.error('Błąd podczas analizy:', err);
                showError('Wystąpił błąd podczas analizy nagrania.');
            }
        }
        
        async function handleUpload(event) {
            event.preventDefault();
            
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('Proszę wybrać plik audio.');
                return;
            }
            
            console.log("Wybrany plik:", file.name, "Rozmiar:", file.size, "Typ:", file.type);
            
            try {
                hideError();
                const formData = new FormData();
                formData.append('file', file);
                
                console.log("Wysyłanie pliku do analizy...");
                
                // Wyślij plik do analizy
                const response = await fetch('/predict/', {
                    method: 'POST',
                    body: formData
                });
                
                console.log("Status odpowiedzi:", response.status);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("Błąd serwera:", errorText);
                    throw new Error(`Błąd serwera: ${response.status}`);
                }
                
                const result = await response.json();
                displayResults(result);
                
            } catch (err) {
                console.error('Błąd podczas analizy pliku:', err);
                showError('Wystąpił błąd podczas analizy pliku audio.');
            }
        }
        
        // Wyświetl wyniki
        function displayResults(result) {
            // Pokaż kartę wyników
            resultsCard.style.display = 'block';
            
            // Ustaw główną emocję
            mainEmotion.textContent = `Emocja: ${result.emotion} (${Math.round(result.confidence * 100)}%)`;
            
            // Wygeneruj paski dla wszystkich emocji
            emotionBars.innerHTML = '';
            const emotions = result.all_emotions;
            
            Object.keys(emotions).forEach(emotion => {
                const value = emotions[emotion];
                const percentage = Math.round(value * 100);
                
                const emotionContainer = document.createElement('div');
                emotionContainer.className = 'mb-3';
                
                const label = document.createElement('div');
                label.className = 'emotion-label';
                label.textContent = `${emotion}: ${percentage}%`;
                
                const barContainer = document.createElement('div');
                barContainer.className = 'progress';
                
                const bar = document.createElement('div');
                bar.className = 'progress-bar';
                bar.style.width = `${percentage}%`;
                bar.setAttribute('role', 'progressbar');
                bar.setAttribute('aria-valuenow', percentage);
                bar.setAttribute('aria-valuemin', '0');
                bar.setAttribute('aria-valuemax', '100');
                
                // Kolor paska w zależności od wartości
                if (percentage > 70) {
                    bar.classList.add('bg-success');
                } else if (percentage > 30) {
                    bar.classList.add('bg-primary');
                } else {
                    bar.classList.add('bg-secondary');
                }
                
                barContainer.appendChild(bar);
                emotionContainer.appendChild(label);
                emotionContainer.appendChild(barContainer);
                emotionBars.appendChild(emotionContainer);
            });
            
            // Przewiń do wyników
            resultsCard.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Pokaż błąd
        function showError(message) {
            errorMsg.textContent = message;
            errorMsg.style.display = 'block';
        }
        
        // Ukryj błąd
        function hideError() {
            errorMsg.style.display = 'none';
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

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



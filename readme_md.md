# Emotion Recognition API

API do rozpoznawania emocji w plikach audio przy użyciu modelu ResNet wytrenowanego na melspektrogramach.

## Cechy

- Rozpoznawanie emocji z plików audio
- Obsługa różnych formatów audio (mp3, wav, ogg, flac, m4a, webm)
- API REST z pełną dokumentacją OpenAPI
- Moduł nagrywania dźwięku bezpośrednio z przeglądarki
- Zaimplementowany model ResNet dostosowany do analizy audio

## Wymagania

- Python 3.13
- PyTorch 2.7
- FastAPI 0.115.12
- Librosa
- Pozostałe zależności w pliku requirements.txt

## Instalacja

1. Klonuj repozytorium:
   ```
   git clone https://github.com/username/emotion-recognition-api.git
   cd emotion-recognition-api
   ```

2. Stwórz i aktywuj wirtualne środowisko:
   ```
   python -m venv venv
   source venv/bin/activate  # Na Windows użyj: venv\Scripts\activate
   ```

3. Zainstaluj zależności:
   ```
   pip install -r requirements.txt
   ```

4. Przygotuj model:
   Umieść plik wytrenowanego modelu w katalogu `model_outputs` pod nazwą `best_model.pt`

## Uruchomienie

1. Uruchom serwer API:
   ```
   python -m app.main
   ```

2. Otwórz dokumentację API:
   ```
   http://localhost:8000/docs
   ```

## Endpointy API

- `GET /api/v1/health` - sprawdza status serwera i modelu
- `POST /api/v1/predict` - analizuje przesłany plik audio
- `POST /api/v1/record` - analizuje audio nagrane bezpośrednio w przeglądarce

## Struktura projektu

```
emotion_recognition_api/
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   └── settings.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── schemas.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── audio_resnet.py
│   │   └── feature_extraction.py
│   └── main.py
├── model_outputs/
│   └── best_model.pt
├── requirements.txt
└── README.md
```

## Rozpoznawane emocje

- anger (złość)
- fear (strach)
- happiness (szczęście)
- neutral (neutralność)
- sadness (smutek)
- surprised (zaskoczenie)

## Licencja

MIT

document.addEventListener('DOMContentLoaded', function() {
    // Elementy dla przesyłania pliku
    const uploadForm = document.getElementById('uploadForm');
    const audioFileInput = document.getElementById('audioFile');
    
    // Elementy dla nagrywania
    const startRecordButton = document.getElementById('startRecording');
    const stopRecordButton = document.getElementById('stopRecording');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingComplete = document.getElementById('recordingComplete');
    const analyzeRecordingButton = document.getElementById('analyzeRecording');
    const recordingTimeDisplay = document.getElementById('recordingTime');
    
    // Elementy wspólne
    const audioPlayer = document.getElementById('audioPlayer');
    const player = document.getElementById('player');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const mainEmotion = document.getElementById('mainEmotion');
    const confidenceBar = document.getElementById('confidenceBar');
    const emotionDetails = document.getElementById('emotionDetails');
    
    // Zmienne dla nagrywania
    let mediaRecorder;
    let audioChunks = [];
    let recordedBlob;
    let recordingTimer;
    let recordingStartTime;
    let MAX_RECORDING_TIME = 5000; // 5 sekund w milisekundach
    
    let emotionChart = null;
    
    // Kolory dla różnych emocji w wykresie
    const emotionColors = {
        'happiness': 'rgba(255, 193, 7, 0.8)',
        'sadness': 'rgba(13, 110, 253, 0.8)',
        'anger': 'rgba(220, 53, 69, 0.8)',
        'fear': 'rgba(111, 66, 193, 0.8)',
        'neutral': 'rgba(108, 117, 125, 0.8)',
        'surprised': 'rgba(253, 126, 20, 0.8)',
        // Dodaj więcej emocji jeśli potrzebne
    };
    
    // Tłumaczenia emocji na polski
    const emotionTranslations = {
        'happiness': 'Radość',
        'sadness': 'Smutek',
        'anger': 'Złość',
        'fear': 'Strach',
        'neutral': 'Neutralność',
        'surprised': 'Zaskoczenie',
        // Dodaj więcej tłumaczeń jeśli potrzebne
    };
    
    // Obsługa przesyłania pliku
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = audioFileInput.files[0];
        if (!file) {
            alert('Wybierz plik audio do analizy.');
            return;
        }
        
        processAudioFile(file);
    });
    
    // Obsługa nagrywania
    startRecordButton.addEventListener('click', startRecording);
    stopRecordButton.addEventListener('click', stopRecording);
    analyzeRecordingButton.addEventListener('click', function() {
        if (recordedBlob) {
            processAudioFile(new File([recordedBlob], "recorded-audio.wav", { 
                type: 'audio/wav' 
            }));
        }
    });
    
    // Funkcja do inicjalizacji nagrywania
    async function startRecording() {
        try {
            // Resetowanie stanu aplikacji
            resetAppState();
            
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.addEventListener('stop', () => {
                // Zatrzymanie wszystkich ścieżek w strumieniu
                stream.getTracks().forEach(track => track.stop());
                
                // Tworzenie pliku audio
                recordedBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioURL = URL.createObjectURL(recordedBlob);
                
                // Ustawianie źródła dla odtwarzacza
                player.src = audioURL;
                audioPlayer.classList.remove('d-none');
                
                // Aktualizacja interfejsu
                recordingStatus.classList.add('d-none');
                recordingComplete.classList.remove('d-none');
                
                // Zatrzymanie licznika czasu
                clearInterval(recordingTimer);
            });
            
            // Rozpoczęcie nagrywania
            mediaRecorder.start();
            
            // Aktualizacja interfejsu
            startRecordButton.disabled = true;
            stopRecordButton.disabled = false;
            recordingStatus.classList.remove('d-none');
            recordingComplete.classList.add('d-none');
            
            // Ustawienie limitu czasu nagrywania (5 sekund)
            recordingStartTime = Date.now();
            recordingTimer = setInterval(updateRecordingTime, 100);
            
            // Automatyczne zatrzymanie po MAX_RECORDING_TIME
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    stopRecording();
                }
            }, MAX_RECORDING_TIME);
            
        } catch (err) {
            console.error('Błąd podczas rozpoczynania nagrywania:', err);
            alert('Nie udało się uzyskać dostępu do mikrofonu. Sprawdź uprawnienia.');
            
            startRecordButton.disabled = false;
            stopRecordButton.disabled = true;
        }
    }
    
    // Funkcja do aktualizacji wyświetlanego czasu nagrywania
    function updateRecordingTime() {
        const elapsedTime = Date.now() - recordingStartTime;
        const seconds = Math.floor(elapsedTime / 1000);
        const milliseconds = Math.floor((elapsedTime % 1000) / 10);
        
        const timeLeft = Math.max(0, MAX_RECORDING_TIME - elapsedTime);
        const secondsLeft = Math.floor(timeLeft / 1000);
        const millisecondsLeft = Math.floor((timeLeft % 1000) / 10);
        
        recordingTimeDisplay.textContent = `${seconds}.${milliseconds.toString().padStart(2, '0')}s / ${Math.floor(MAX_RECORDING_TIME/1000)}s`;
    }
    
    // Funkcja do zatrzymania nagrywania
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            
            // Aktualizacja interfejsu
            startRecordButton.disabled = false;
            stopRecordButton.disabled = true;
        }
    }
    
    // Funkcja do przetwarzania pliku audio (zarówno przesłanego jak i nagranego)
    function processAudioFile(file) {
        // Wyświetlenie odtwarzacza audio
        const fileURL = URL.createObjectURL(file);
        player.src = fileURL;
        audioPlayer.classList.remove('d-none');
        
        // Przygotowanie danych do wysłania
        const formData = new FormData();
        formData.append('file', file);
        
        // Resetowanie widoków
        results.classList.add('d-none');
        error.classList.add('d-none');
        loading.classList.remove('d-none');
        
        // Wysłanie zapytania do API
        fetch('/predict/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd serwera');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(err => {
            console.error('Error:', err);
            error.classList.remove('d-none');
            loading.classList.add('d-none');
        });
    }
    
    // Resetowanie stanu aplikacji
    function resetAppState() {
        // Ukrycie wyników i błędów
        results.classList.add('d-none');
        error.classList.add('d-none');
        loading.classList.add('d-none');
        
        // Resetowanie nagrywania
        recordingStatus.classList.add('d-none');
        recordingComplete.classList.add('d-none');
        
        // Resetowanie odtwarzacza
        audioPlayer.classList.add('d-none');
        player.src = '';
    }
    
    function displayResults(data) {
        loading.classList.add('d-none');
        results.classList.remove('d-none');
        
        // Wyświetlenie głównej emocji
        const dominantEmotion = data.emotion;
        const translatedEmotion = emotionTranslations[dominantEmotion] || dominantEmotion;
        const confidence = (data.confidence * 100).toFixed(1);
        
        // Wyświetl główną emocję z wyróżnieniem
        mainEmotion.textContent = `${translatedEmotion} (${confidence}%)`;
        mainEmotion.classList.add('emotion-main'); // Dodanie klasy dla stylizacji
        
        confidenceBar.style.width = `${confidence}%`;
        confidenceBar.textContent = `${confidence}%`;
        confidenceBar.setAttribute('aria-valuenow', confidence);
        
        // Tworzenie listy szczegółów
        emotionDetails.innerHTML = '';
        const emotions = data.all_emotions;
        
        // Sortowanie emocji wg wartości (od najwyższej)
        const sortedEmotions = Object.entries(emotions)
            .sort((a, b) => b[1] - a[1]);
            
        sortedEmotions.forEach(([emotion, value]) => {
            const percentage = (value * 100).toFixed(1);
            const translated = emotionTranslations[emotion] || emotion;
            
            const li = document.createElement('li');
            li.className = `list-group-item d-flex justify-content-between align-items-center emotion-${emotion}`;
            
            // Dodanie klasy w zależności od wartości procentowej dla dodatkowego wyróżnienia
            if (value > 0.5) {
                li.classList.add('emotion-high');
            } else if (value > 0.2) {
                li.classList.add('emotion-medium');
            }
            
            li.innerHTML = `
                <span class="emotion-label">${translated}</span>
                <span class="badge bg-primary rounded-pill">${percentage}%</span>
            `;
            emotionDetails.appendChild(li);
        });
        
        // Tworzenie wykresu kołowego
        createChart(data.all_emotions);
    }
    
function createChart(emotions) {
    // Przygotowanie danych dla wykresu
    const labels = [];
    const data = [];
    const backgroundColors = [];
    
    for (const [emotion, value] of Object.entries(emotions)) {
        if (value > 0.01) {  // Pokazuj tylko wartości powyżej 1%
            const translated = emotionTranslations[emotion] || emotion;
            labels.push(translated);
            data.push((value * 100).toFixed(1));
            backgroundColors.push(emotionColors[emotion] || getRandomColor());
        }
    }
    
    // Zniszczenie poprzedniego wykresu jeśli istnieje
    if (emotionChart) {
        emotionChart.destroy();
    }
    
    // Stworzenie nowego wykresu
    const ctx = document.getElementById('emotionChart').getContext('2d');
    emotionChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 28,  // Zwiększony rozmiar czcionki w legendzie
                            weight: 'bold'
                        },
                        boxWidth: 35,  // Zwiększona szerokość kolorowego pola w legendzie
                        boxHeight: 20,  // Zwiększona wysokość kolorowego pola w legendzie
                        color: '#333'   // Ciemniejszy kolor dla lepszej czytelności
                    }
                },
                title: {
                    display: true,
                    text: 'Rozkład emocji w nagraniu',
                    font: {
                        size: 28,  // Zwiększony tytuł wykresu
                        weight: 'bold'
                    },
                    padding: {
                        top: 10,
                        bottom: 30
                    },
                    color: '#333'  // Ciemniejszy kolor dla lepszej czytelności
                },
                tooltip: {
                    titleFont: {
                        size: 22  // Zwiększony rozmiar czcionki w podpowiedzi (tytuł)
                    },
                    bodyFont: {
                        size: 20   // Zwiększony rozmiar czcionki w podpowiedzi (treść)
                    },
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Zwiększ kontener wykresu dla lepszej widoczności
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        chartContainer.style.height = '600px';
    }
}

function getRandomColor() {
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    return `rgba(${r}, ${g}, ${b}, 0.8)`;
}
});
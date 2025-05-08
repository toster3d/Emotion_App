import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button, Card, ProgressBar, Alert } from 'react-bootstrap';
import { FaMicrophone, FaStop, FaTrash, FaPlay, FaPause } from 'react-icons/fa';
import { useReactMediaRecorder } from 'react-media-recorder';

const AudioRecorder = ({ onRecordingComplete }) => {
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [processingAudio, setProcessingAudio] = useState(false); // Flag to prevent multiple submissions
  const [audioSent, setAudioSent] = useState(false); // Flag to track if audio has been sent already
  
  const maxRecordingTime = 10; // Maximum recording time in seconds
  const audioContextRef = useRef(null);
  
  // Funkcja do sprawdzania wspieranych formatów audio
  const getSupportedMimeType = useCallback(() => {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/ogg;codecs=opus'
    ];
    return types.find(type => MediaRecorder.isTypeSupported(type)) || '';
  }, []);
  
  const {
    status,
    startRecording,
    stopRecording,
    mediaBlobUrl,
    clearBlobUrl,
    error
  } = useReactMediaRecorder({
    audio: true,
    video: false,
    echoCancellation: true,
    autoGainControl: true,
    noiseSuppression: true,
    mediaRecorderOptions: {
      mimeType: getSupportedMimeType(),
      audioBitsPerSecond: 128000, // 128 kbps
    },
    onError: (err) => {
      console.error('Media recorder error:', err);
      setErrorMessage('Nie można uzyskać dostępu do mikrofonu. Sprawdź uprawnienia przeglądarki.');
    }
  });
  
  // Konwersja audio jeśli potrzebne dla backendu
  const processAudioForBackend = useCallback(async (audioBlob) => {
    try {
      // Jeśli backend wymaga WAV, ale przeglądarka nagrała inny format
      // można tutaj wykonać konwersję
      // Dla uproszczenia zwracam oryginalny blob
      return audioBlob;
    } catch (error) {
      console.error('Error processing audio format:', error);
      setErrorMessage('Błąd przetwarzania nagrania audio.');
      return audioBlob;
    }
  }, []);
  
  const fetchRecordingBlob = useCallback(async (url) => {
    if (!url || processingAudio || audioSent) return;
    
    try {
      setProcessingAudio(true);
      const response = await fetch(url);
      const originalBlob = await response.blob();
      const processedBlob = await processAudioForBackend(originalBlob);
      
      // Save flag that we've sent this audio
      setAudioSent(true);
      onRecordingComplete(processedBlob);
    } catch (error) {
      console.error('Error fetching recording blob:', error);
      setErrorMessage('Błąd podczas pobierania nagrania.');
    } finally {
      setProcessingAudio(false);
    }
  }, [onRecordingComplete, processAudioForBackend, processingAudio, audioSent]);
  
  // Handle recording time counter
  useEffect(() => {
    let interval;
    if (status === 'recording') {
      interval = setInterval(() => {
        setRecordingTime((prevTime) => {
          if (prevTime >= maxRecordingTime) {
            stopRecording();
            return maxRecordingTime;
          }
          return prevTime + 1;
        });
      }, 1000);
    } else if (status === 'stopped' && recordingTime > 0 && mediaBlobUrl && !audioSent) {
      // Only create audio element and fetch blob if audio hasn't been sent yet
      const audio = new Audio(mediaBlobUrl);
      setAudioElement(audio);
      fetchRecordingBlob(mediaBlobUrl);
    }
    
    return () => clearInterval(interval);
  }, [status, mediaBlobUrl, stopRecording, recordingTime, fetchRecordingBlob, audioSent]);
  
  // Obsługa błędów z MediaRecorder
  useEffect(() => {
    if (error) {
      console.error('MediaRecorder error:', error);
      setErrorMessage('Wystąpił błąd podczas nagrywania. Spróbuj ponownie.');
    }
  }, [error]);
  
  // Czyszczenie zasobów przy odmontowaniu komponentu
  useEffect(() => {
    return () => {
      if (audioElement) {
        audioElement.pause();
      }
      if (audioContextRef.current) {
        // Skopiuj wartość do lokalnej zmiennej przed użyciem
        const currentAudioContext = audioContextRef.current;
        currentAudioContext.close().catch(console.error);
      }
    };
  }, [audioElement]);
  
  const handleStartRecording = () => {
    setErrorMessage(null);
    setRecordingTime(0);
    setAudioSent(false); // Reset the flag when starting a new recording
    
    if (audioElement) {
      audioElement.pause();
      setIsPlaying(false);
    }
    clearBlobUrl();
    startRecording();
  };
  
  const handlePlayRecording = () => {
    if (audioElement) {
      if (isPlaying) {
        audioElement.pause();
      } else {
        audioElement.onended = () => setIsPlaying(false);
        audioElement.play().catch(error => {
          console.error('Error playing audio:', error);
          setErrorMessage('Nie można odtworzyć nagrania.');
        });
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  const handleReset = () => {
    setErrorMessage(null);
    setAudioSent(false); // Reset the flag when resetting
    
    if (audioElement) {
      audioElement.pause();
      setIsPlaying(false);
    }
    clearBlobUrl();
    setRecordingTime(0);
    setAudioElement(null);
  };
  
  const renderRecordingStatus = () => {
    if (status === 'recording') {
      return (
        <div className="d-flex align-items-center mb-3">
          <div className="recording-indicator" style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: 'red',
            marginRight: '8px',
            animation: 'pulse 1s infinite'
          }}></div>
          <span>Recording... {recordingTime}s / {maxRecordingTime}s</span>
        </div>
      );
    }
    return null;
  };
  
  return (
    <Card className="mb-4">
      <Card.Body>
        <Card.Title>Record Audio</Card.Title>
        
        {errorMessage && (
          <Alert variant="danger" dismissible onClose={() => setErrorMessage(null)}>
            {errorMessage}
          </Alert>
        )}
        
        {renderRecordingStatus()}
        
        {status === 'recording' && (
          <ProgressBar 
            now={(recordingTime / maxRecordingTime) * 100} 
            variant="danger" 
            className="mb-3" 
          />
        )}
        
        {processingAudio && (
          <Alert variant="info">
            <Alert.Heading>Trwa przetwarzanie...</Alert.Heading>
            <p>Proszę czekać, nagranie jest przygotowywane do analizy.</p>
          </Alert>
        )}
        
        <div className="d-flex gap-2">
          {status !== 'recording' && (
            <Button 
              variant="primary" 
              onClick={handleStartRecording} 
              disabled={status === 'acquiring_media' || processingAudio}
            >
              <FaMicrophone className="me-2" />
              {status === 'acquiring_media' ? 'Preparing...' : 'Start Recording'}
            </Button>
          )}
          
          {status === 'recording' && (
            <Button variant="danger" onClick={stopRecording}>
              <FaStop className="me-2" />
              Stop Recording
            </Button>
          )}
          
          {mediaBlobUrl && (
            <>
              <Button variant="success" onClick={handlePlayRecording} disabled={processingAudio}>
                {isPlaying ? <><FaPause className="me-2" />Pause</> : <><FaPlay className="me-2" />Play</>} Recording
              </Button>
              
              <Button variant="outline-danger" onClick={handleReset} disabled={processingAudio}>
                <FaTrash className="me-2" />
                Reset
              </Button>
            </>
          )}
        </div>
        
        {mediaBlobUrl && (
          <div className="mt-3">
            <audio src={mediaBlobUrl} controls className="w-100" />
          </div>
        )}
        
        {status === 'idle' && !mediaBlobUrl && (
          <div className="text-muted mt-3">
            <small>
              Nagraj swoją wypowiedź, aby wykryć emocje w głosie. 
              Najlepsze rezultaty uzyskasz mówiąc wyraźnie, bez hałasu w tle.
            </small>
          </div>
        )}

        <style jsx="true">{`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
          }
        `}</style>
      </Card.Body>
    </Card>
  );
};

export default AudioRecorder;

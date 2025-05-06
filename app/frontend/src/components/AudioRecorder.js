import React, { useState, useEffect } from 'react';
import { Button, Card, ProgressBar } from 'react-bootstrap';
import { FaMicrophone, FaStop, FaTrash } from 'react-icons/fa';
import { useReactMediaRecorder } from 'react-media-recorder';

const AudioRecorder = ({ onRecordingComplete }) => {
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState(null);
  const maxRecordingTime = 10; // Maximum recording time in seconds
  
  const {
    status,
    startRecording,
    stopRecording,
    mediaBlobUrl,
    clearBlobUrl
  } = useReactMediaRecorder({
    audio: true,
    video: false,
    blobPropertyBag: { type: 'audio/wav' }
  });
  
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
    } else {
      if (status === 'stopped' && recordingTime > 0 && mediaBlobUrl) {
        // Create audio element from the recorded blob
        const audio = new Audio(mediaBlobUrl);
        setAudioElement(audio);
        
        // Pass the recording to the parent component
        fetchRecordingBlob();
      }
    }
    
    return () => clearInterval(interval);
  }, [status, mediaBlobUrl, stopRecording, recordingTime]);
  
  // Reset everything when starting a new recording
  const handleStartRecording = () => {
    setRecordingTime(0);
    if (audioElement) {
      audioElement.pause();
      setIsPlaying(false);
    }
    clearBlobUrl();
    startRecording();
  };
  
  // Play the recorded audio
  const handlePlayRecording = () => {
    if (audioElement) {
      if (isPlaying) {
        audioElement.pause();
      } else {
        audioElement.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  // Reset everything
  const handleReset = () => {
    if (audioElement) {
      audioElement.pause();
      setIsPlaying(false);
    }
    clearBlobUrl();
    setRecordingTime(0);
    setAudioElement(null);
  };
  
  // Get the actual Blob from the mediaBlobUrl
  const fetchRecordingBlob = async () => {
    if (!mediaBlobUrl) return;
    
    try {
      const response = await fetch(mediaBlobUrl);
      const blob = await response.blob();
      onRecordingComplete(blob);
    } catch (error) {
      console.error('Error fetching recording blob:', error);
    }
  };
  
  // Render recording status
  const renderRecordingStatus = () => {
    if (status === 'recording') {
      return (
        <div className="d-flex align-items-center mb-3">
          <div className="recording-indicator"></div>
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
        
        {renderRecordingStatus()}
        
        {status === 'recording' && (
          <ProgressBar 
            now={(recordingTime / maxRecordingTime) * 100} 
            variant="danger" 
            className="mb-3" 
          />
        )}
        
        <div className="d-flex gap-2">
          {status !== 'recording' && (
            <Button 
              variant="primary" 
              onClick={handleStartRecording} 
              disabled={status === 'acquiring_media'}
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
              <Button variant="success" onClick={handlePlayRecording}>
                {isPlaying ? 'Pause' : 'Play'} Recording
              </Button>
              
              <Button variant="outline-danger" onClick={handleReset}>
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
      </Card.Body>
    </Card>
  );
};

export default AudioRecorder; 
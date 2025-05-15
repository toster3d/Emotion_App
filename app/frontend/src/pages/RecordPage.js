import React, { useState } from 'react';
import { Row, Col, Card, Alert } from 'react-bootstrap';
import AudioRecorder from '../components/AudioRecorder';
import EmotionResult from '../components/EmotionResult';
import { predictEmotionFromRecording } from '../services/api';

const RecordPage = () => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handleRecordingComplete = async (blob) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // The blob from the recorder is already in the correct format
      const prediction = await predictEmotionFromRecording(blob);
      setResult(prediction);
    } catch (err) {
      console.error('Error analyzing recording:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to analyze recording');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div>
      <h1 className="mb-4">Record & Analyze</h1>
      
      <Row>
        <Col lg={6}>
          <AudioRecorder onRecordingComplete={handleRecordingComplete} />
          
          {isLoading && (
            <Alert variant="info">
              <Alert.Heading>Analyzing...</Alert.Heading>
              <p>
                Please wait while we analyze your audio. This may take a few moments.
              </p>
            </Alert>
          )}
        </Col>
        
        <Col lg={6}>
          {result || error ? (
            <EmotionResult result={result} error={error} />
          ) : (
            <Card className="text-center p-4">
              <Card.Body>
                <Card.Title>Results will appear here</Card.Title>
                <Card.Text>
                  Record audio to see emotion analysis results
                </Card.Text>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
      
      <div className="mt-4">
        <h3>How it works</h3>
        <p>
          The audio is recorded in your browser, then sent to our server where:
        </p>
        <ol>
          <li>The audio is processed to extract the melspectrogram</li>
          <li>The melspectrogram is analyzed by a specialized ResNet18 neural network</li>
          <li>The result is used for the final prediction</li>
        </ol>
      </div>
    </div>
  );
};

export default RecordPage; 
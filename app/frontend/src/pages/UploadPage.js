import React, { useState } from 'react';
import { Row, Col, Card, Alert } from 'react-bootstrap';
import FileUploader from '../components/FileUploader';
import EmotionResult from '../components/EmotionResult';
import { predictEmotionFromFile } from '../services/api';

const UploadPage = () => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handleFileSelected = async (file) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Send the file to the API for analysis
      const prediction = await predictEmotionFromFile(file);
      setResult(prediction);
      
      return prediction; // Return the result to the FileUploader component
    } catch (err) {
      console.error('Error analyzing file:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to analyze file';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div>
      <h1 className="mb-4">Upload & Analyze</h1>
      
      <Row>
        <Col lg={6}>
          <FileUploader onFileSelected={handleFileSelected} />
          
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
                  Upload an audio file to see emotion analysis results
                </Card.Text>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
      
      <div className="mt-4">
        <h3>Supported File Types</h3>
        <p>
          You can upload audio files in the following formats:
        </p>
        <ul>
          <li><strong>MP3</strong> - MPEG Audio Layer III</li>
          <li><strong>WAV</strong> - Waveform Audio File Format</li>
          <li><strong>OGG</strong> - Ogg Vorbis</li>
          <li><strong>FLAC</strong> - Free Lossless Audio Codec</li>
          <li><strong>M4A</strong> - MPEG-4 Audio</li>
          <li><strong>WEBM</strong> - Web Media</li>
        </ul>
        <p>
          For best results, use recordings with clear speech and minimal background noise.
        </p>
      </div>
    </div>
  );
};

export default UploadPage; 
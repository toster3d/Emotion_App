import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FaMicrophone, FaUpload, FaCog } from 'react-icons/fa';
import { checkApiHealth } from '../services/api';

const HomePage = () => {
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const checkStatus = async () => {
      try {
        setLoading(true);
        const healthData = await checkApiHealth();
        setApiStatus(healthData);
        setError(null);
      } catch (err) {
        setError('Could not connect to the API. Please ensure the server is running.');
        console.error('Health check error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    checkStatus();
  }, []);
  
  return (
    <div>
      <div className="text-center mb-5">
        <h1>Audio Emotion Detection</h1>
        <p className="lead">
          Analyze the emotions in your voice using our advanced PyTorch ensemble model.
        </p>
      </div>
      
      {error && (
        <Alert variant="danger" className="mb-4">
          <Alert.Heading>Connection Error</Alert.Heading>
          <p>{error}</p>
        </Alert>
      )}
      
      {apiStatus && !loading && (
        <Alert variant="success" className="mb-4">
          <Alert.Heading>API Connected</Alert.Heading>
          <p>
            The API is running on {apiStatus.device}. 
            {apiStatus.models_loaded ? 
              `Models loaded: ${apiStatus.available_models.join(', ')}` : 
              'Models are still loading...'}
          </p>
        </Alert>
      )}
      
      <Row>
        <Col md={6} className="mb-4">
          <Card>
            <Card.Body className="text-center p-5">
              <FaMicrophone className="display-1 mb-3 text-primary" />
              <Card.Title className="mb-3">Record Audio</Card.Title>
              <Card.Text>
                Record your voice directly in the browser and analyze the emotions in real-time.
              </Card.Text>
              <Button 
                as={Link} 
                to="/record" 
                variant="primary" 
                size="lg" 
                className="mt-3"
                disabled={loading || error}
              >
                Start Recording
              </Button>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6} className="mb-4">
          <Card>
            <Card.Body className="text-center p-5">
              <FaUpload className="display-1 mb-3 text-success" />
              <Card.Title className="mb-3">Upload Audio</Card.Title>
              <Card.Text>
                Upload an existing audio file and analyze the emotions expressed in it.
              </Card.Text>
              <Button 
                as={Link} 
                to="/upload" 
                variant="success" 
                size="lg" 
                className="mt-3"
                disabled={loading || error}
              >
                Upload File
              </Button>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <div className="mt-3 text-center">
        <h4>Technology Stack</h4>
        <p>
          This application uses a PyTorch ensemble model with ResNet18 architecture to analyze 
          emotions in audio. The model processes different audio features including 
          melspectrograms, MFCCs, and chromatograms.
        </p>
      </div>
    </div>
  );
};

export default HomePage; 
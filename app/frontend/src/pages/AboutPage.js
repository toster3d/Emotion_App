import React from 'react';
import { Card, Row, Col, ListGroup } from 'react-bootstrap';
import { FaPython, FaReact, FaDocker, FaBrain } from 'react-icons/fa';
import { SiPytorch, SiFastapi } from 'react-icons/si';

const AboutPage = () => {
  return (
    <div>
      <h1 className="mb-4">About</h1>
      
      <Card className="mb-4">
        <Card.Body>
          <Card.Title>Audio Emotion Detection</Card.Title>
          <Card.Text>
            This application analyzes audio recordings to detect emotions using a PyTorch 
            ensemble model. The model combines predictions from three different ResNet18 
            networks, each trained on different audio features.
          </Card.Text>
        </Card.Body>
      </Card>
      
      <h2 className="mb-3">Technology Stack</h2>
      
      <Row>
        <Col md={6} className="mb-4">
          <Card>
            <Card.Header>Backend</Card.Header>
            <ListGroup variant="flush">
              <ListGroup.Item>
                <FaPython className="me-2" /> Python 3.10+
              </ListGroup.Item>
              <ListGroup.Item>
                <SiFastapi className="me-2" /> FastAPI
              </ListGroup.Item>
              <ListGroup.Item>
                <SiPytorch className="me-2" /> PyTorch
              </ListGroup.Item>
              <ListGroup.Item>
                <FaBrain className="me-2" /> ResNet18 Neural Networks
              </ListGroup.Item>
            </ListGroup>
          </Card>
        </Col>
        
        <Col md={6} className="mb-4">
          <Card>
            <Card.Header>Frontend</Card.Header>
            <ListGroup variant="flush">
              <ListGroup.Item>
                <FaReact className="me-2" /> React
              </ListGroup.Item>
              <ListGroup.Item>
                <FaReact className="me-2" /> React Bootstrap
              </ListGroup.Item>
              <ListGroup.Item>
                <FaReact className="me-2" /> Recharts
              </ListGroup.Item>
              <ListGroup.Item>
                <FaDocker className="me-2" /> Docker for deployment
              </ListGroup.Item>
            </ListGroup>
          </Card>
        </Col>
      </Row>
      
      <h2 className="mb-3">How it Works</h2>
      
      <Card className="mb-4">
        <Card.Body>
          <h5>Audio Processing Pipeline</h5>
          <ol>
            <li>
              <strong>Audio Input</strong>: The system accepts audio files or recorded audio 
              from the browser.
            </li>
            <li>
              <strong>Feature Extraction</strong>: Three types of audio features are extracted:
              <ul>
                <li>Mel Spectrograms - Frequency representation based on human hearing</li>
                <li>MFCCs (Mel-Frequency Cepstral Coefficients) - Compact representation of audio</li>
                <li>Chromatograms - Representation based on musical pitch classes</li>
              </ul>
            </li>
            <li>
              <strong>Model Inference</strong>: Each feature is processed by its own specialized 
              ResNet18 neural network.
            </li>
            <li>
              <strong>Ensemble Prediction</strong>: The weighted ensemble model combines predictions 
              from individual models.
            </li>
            <li>
              <strong>Result</strong>: The final emotion prediction is returned with confidence scores.
            </li>
          </ol>
        </Card.Body>
      </Card>
      
      <Card>
        <Card.Body>
          <h5>Optimizations</h5>
          <p>
            The model leverages several PyTorch optimizations for efficient inference:
          </p>
          <ul>
            <li>TorchScript JIT compilation for faster execution</li>
            <li>Frozen model weights for reduced memory usage</li>
            <li>Inference mode to disable gradient computation</li>
            <li>Batch processing for multiple features</li>
          </ul>
        </Card.Body>
      </Card>
    </div>
  );
};

export default AboutPage; 
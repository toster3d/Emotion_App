import React from 'react';
import { Card, Row, Col, ListGroup } from 'react-bootstrap';
import { FaPython, FaReact, FaDocker, FaBrain } from 'react-icons/fa';
import { SiPytorch, SiFastapi, SiNumpy, SiPydantic } from 'react-icons/si';

const AboutPage = () => {
  return (
    <div>
      <h1 className="mb-4">About</h1>
      
      <Card className="mb-4">
        <Card.Body>
          <Card.Title>Audio Emotion Detection</Card.Title>
          <Card.Text>
            This application analyzes audio recordings to detect emotions using a PyTorch 
            ResNet18 model trained on mel spectrograms extracted from audio data.
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
                <FaPython className="me-2" /> Python 3.13+
              </ListGroup.Item>
              <ListGroup.Item>
                <SiFastapi className="me-2" /> FastAPI 0.115+
              </ListGroup.Item>
              <ListGroup.Item>
                <SiPytorch className="me-2" /> PyTorch 2.7+
              </ListGroup.Item>
              <ListGroup.Item>
                <FaBrain className="me-2" /> TorchVision 0.18+
              </ListGroup.Item>
              <ListGroup.Item>
                <SiNumpy className="me-2" /> NumPy 2.2+
              </ListGroup.Item>
              <ListGroup.Item>
                <SiPydantic className="me-2" /> Pydantic 2.11+
              </ListGroup.Item>
              <ListGroup.Item>
                <FaBrain className="me-2" /> Librosa 0.11.0 (Audio Processing)
              </ListGroup.Item>
            </ListGroup>
          </Card>
        </Col>
        
        <Col md={6} className="mb-4">
          <Card>
            <Card.Header>Frontend</Card.Header>
            <ListGroup variant="flush">
              <ListGroup.Item>
                <FaReact className="me-2" /> React 18.2+
              </ListGroup.Item>
              <ListGroup.Item>
                <FaReact className="me-2" /> React Bootstrap 2.9+
              </ListGroup.Item>
              <ListGroup.Item>
                <FaReact className="me-2" /> Recharts 2.10+
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
              <strong>Feature Extraction</strong>: Mel spectrograms are extracted from the audio:
              <ul>
                <li>Mel Spectrograms - Frequency representation based on human hearing perception</li>
                <li>Processing includes normalization and conversion to decibel scale</li>
                <li>The resulting spectrograms capture the audio characteristics needed for emotion detection</li>
              </ul>
            </li>
            <li>
              <strong>Model Inference</strong>: The extracted features are processed by a ResNet18 
              neural network trained specifically for emotion recognition.
            </li>
            <li>
              <strong>Result</strong>: The final emotion prediction is returned with confidence scores 
              for each emotion.
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
            <li>Efficient preprocessing of audio data</li>
            <li>Frozen model weights for reduced memory usage</li>
            <li>Inference mode to disable gradient computation</li>
            <li>GPU acceleration when available</li>
          </ul>
        </Card.Body>
      </Card>
    </div>
  );
};

export default AboutPage; 
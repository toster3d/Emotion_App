import React, { useState } from 'react';
import { Form, Button, Card, ProgressBar, Alert } from 'react-bootstrap';
import { FaUpload, FaTrash } from 'react-icons/fa';

const FileUploader = ({ onFileSelected }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const validFileTypes = ['audio/mp3', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/webm'];
  const maxFileSize = 10 * 1024 * 1024; // 10MB
  
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setError(null);
    
    if (!file) {
      setSelectedFile(null);
      setPreview(null);
      return;
    }
    
    // Validate file type
    if (!validFileTypes.includes(file.type) && 
        !file.name.endsWith('.mp3') && 
        !file.name.endsWith('.wav') && 
        !file.name.endsWith('.ogg') && 
        !file.name.endsWith('.flac') && 
        !file.name.endsWith('.m4a') && 
        !file.name.endsWith('.webm')) {
      setError('Please select a valid audio file (MP3, WAV, OGG, FLAC, M4A, WEBM)');
      setSelectedFile(null);
      setPreview(null);
      return;
    }
    
    // Validate file size
    if (file.size > maxFileSize) {
      setError(`File size exceeds the limit of ${maxFileSize / 1024 / 1024}MB`);
      setSelectedFile(null);
      setPreview(null);
      return;
    }
    
    setSelectedFile(file);
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);
  };
  
  const handleSubmit = (event) => {
    event.preventDefault();
    
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }
    
    setIsUploading(true);
    
    // Simulate progress
    const progressInterval = setInterval(() => {
      setUploadProgress((prevProgress) => {
        const nextProgress = prevProgress + 10;
        return nextProgress > 90 ? 90 : nextProgress;
      });
    }, 300);
    
    // Call the parent component's handler
    onFileSelected(selectedFile)
      .then(() => {
        // Complete the progress bar
        setUploadProgress(100);
        
        // Clear the interval
        clearInterval(progressInterval);
        
        // Reset uploading state after a short delay
        setTimeout(() => {
          setIsUploading(false);
        }, 500);
      })
      .catch((err) => {
        setError(`Upload failed: ${err.message}`);
        setIsUploading(false);
        clearInterval(progressInterval);
      });
  };
  
  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setError(null);
    setUploadProgress(0);
  };
  
  return (
    <Card className="mb-4">
      <Card.Body>
        <Card.Title>Upload Audio File</Card.Title>
        
        {error && (
          <Alert variant="danger" onClose={() => setError(null)} dismissible>
            {error}
          </Alert>
        )}
        
        <Form onSubmit={handleSubmit}>
          <Form.Group controlId="audioFile" className="mb-3">
            <Form.Label>Select an audio file</Form.Label>
            <Form.Control 
              type="file" 
              onChange={handleFileChange}
              accept=".mp3,.wav,.ogg,.flac,.m4a,.webm"
              disabled={isUploading}
            />
            <Form.Text className="text-muted">
              Supported formats: MP3, WAV, OGG, FLAC, M4A, WEBM (max 10MB)
            </Form.Text>
          </Form.Group>
          
          {selectedFile && (
            <div className="mb-3">
              <p>
                Selected file: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
              
              {preview && (
                <audio controls className="w-100 mb-3">
                  <source src={preview} type={selectedFile.type} />
                  Your browser does not support the audio element.
                </audio>
              )}
            </div>
          )}
          
          {isUploading && (
            <ProgressBar 
              now={uploadProgress} 
              label={`${uploadProgress}%`} 
              className="mb-3" 
            />
          )}
          
          <div className="d-flex gap-2">
            <Button 
              variant="primary" 
              type="submit" 
              disabled={!selectedFile || isUploading}
            >
              <FaUpload className="me-2" />
              {isUploading ? 'Processing...' : 'Analyze'}
            </Button>
            
            <Button 
              variant="outline-danger" 
              onClick={handleClear}
              disabled={!selectedFile || isUploading}
            >
              <FaTrash className="me-2" />
              Clear
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default FileUploader; 
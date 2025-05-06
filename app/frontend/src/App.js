import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Container } from 'react-bootstrap';

import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import RecordPage from './pages/RecordPage';
import UploadPage from './pages/UploadPage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <div className="App">
      <Navigation />
      <Container className="mt-4">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/record" element={<RecordPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App; 
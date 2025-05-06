import React from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { Link, useLocation } from 'react-router-dom';
import { FaMicrophone, FaUpload, FaInfoCircle, FaHome } from 'react-icons/fa';

const Navigation = () => {
  const location = useLocation();
  
  return (
    <Navbar bg="dark" variant="dark" expand="lg">
      <Container>
        <Navbar.Brand as={Link} to="/">Audio Emotion Detection</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link as={Link} to="/" active={location.pathname === '/'}>
              <FaHome className="me-1" /> Home
            </Nav.Link>
            <Nav.Link as={Link} to="/record" active={location.pathname === '/record'}>
              <FaMicrophone className="me-1" /> Record
            </Nav.Link>
            <Nav.Link as={Link} to="/upload" active={location.pathname === '/upload'}>
              <FaUpload className="me-1" /> Upload
            </Nav.Link>
            <Nav.Link as={Link} to="/about" active={location.pathname === '/about'}>
              <FaInfoCircle className="me-1" /> About
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default Navigation; 
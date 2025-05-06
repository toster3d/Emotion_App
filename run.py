#!/usr/bin/env python
"""
Helper script to run the application.
This script provides a simple CLI to run different parts of the application.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_models():
    """Run the script to set up dummy models."""
    print("Setting up dummy models for testing...")
    subprocess.run([sys.executable, "setup_models.py"])

def start_backend(host="127.0.0.1", port=8000, reload=True):
    """Start the FastAPI backend."""
    print(f"Starting backend server at http://{host}:{port}...")
    reload_arg = "--reload" if reload else ""
    subprocess.run([
        sys.executable, 
        "-m", "uvicorn", 
        "app.main:app", 
        "--host", host, 
        "--port", str(port),
        reload_arg
    ])

def start_frontend():
    """Start the React frontend."""
    os.chdir("app/frontend")
    print("Starting frontend development server...")
    subprocess.run(["npm", "start"])

def build_frontend():
    """Build the frontend for production."""
    os.chdir("app/frontend")
    print("Building frontend for production...")
    subprocess.run(["npm", "run", "build"])

def start_docker():
    """Start the application using Docker Compose."""
    print("Starting the application with Docker Compose...")
    subprocess.run(["docker-compose", "up", "--build"])

def main():
    """Parse command line arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(description="Audio Emotion Detection App Runner")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup models command
    setup_parser = subparsers.add_parser("setup", help="Set up dummy models for testing")
    
    # Backend command
    backend_parser = subparsers.add_parser("backend", help="Start the FastAPI backend server")
    backend_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    backend_parser.add_argument("--port", default=8000, type=int, help="Port to bind to")
    backend_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Frontend commands
    frontend_parser = subparsers.add_parser("frontend", help="Start the React frontend development server")
    build_parser = subparsers.add_parser("build-frontend", help="Build the frontend for production")
    
    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Start the application with Docker Compose")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "setup":
        setup_models()
    elif args.command == "backend":
        start_backend(args.host, args.port, not args.no_reload)
    elif args.command == "frontend":
        start_frontend()
    elif args.command == "build-frontend":
        build_frontend()
    elif args.command == "docker":
        start_docker()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
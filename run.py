#!/usr/bin/env python
"""
Helper script to run the application.
This script provides a simple CLI to run different parts of the application.
"""

import os
import sys
import argparse
import subprocess
import shutil
import webbrowser
import time
from pathlib import Path

def setup_models():
    """Run the script to set up dummy models."""
    print("Setting up dummy models for testing...")
    subprocess.run([sys.executable, "setup_models.py"])

def start_backend(host="127.0.0.1", port=8000, reload=True, background=False):
    """Start the FastAPI backend."""
    print(f"Starting backend server at http://{host}:{port}...")
    cmd = [
        sys.executable, 
        "-m", "uvicorn", 
        "app.main:app", 
        "--host", host, 
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    if background:
        print("Running backend in background mode...")
        if sys.platform.startswith('win'):
            # Na Windows używamy osobnej konsoli
            startupinfo = None
            try:
                # Próbujemy uruchomić w osobnym oknie
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                backend_proc = subprocess.Popen(
                    cmd, 
                    startupinfo=startupinfo,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
            except Exception:
                backend_proc = subprocess.Popen(
                    cmd, 
                    shell=True,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
            
            # Poczekaj 5 sekund na uruchomienie backendu
            print("Czekam 5 sekund na uruchomienie API...")
            time.sleep(5)
            
            # Otwórz stronę dokumentacji API
            webbrowser.open(f"http://{host}:{port}/docs")
            print(f"Backend API uruchomiony pod adresem: http://{host}:{port}")
            print(f"Dokumentacja API dostępna pod: http://{host}:{port}/docs")
            return backend_proc
        else:
            # Na Linux/Mac
            backend_proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            time.sleep(5)
            webbrowser.open(f"http://{host}:{port}/docs")
            print(f"Backend API uruchomiony pod adresem: http://{host}:{port}")
            return backend_proc
    else:
        # Uruchom w głównym procesie (blokująco)
        subprocess.run(cmd)
        return None

def start_frontend():
    """Start the React frontend."""
    # Sprawdź, czy npm jest zainstalowany
    if not shutil.which("npm"):
        print("Błąd: npm nie jest zainstalowany lub nie jest dostępny w ścieżce systemowej.")
        print("Zainstaluj Node.js i npm z https://nodejs.org/, a następnie spróbuj ponownie.")
        sys.exit(1)
    
    # Zapisz oryginalną ścieżkę i zmień katalog
    original_dir = os.getcwd()
    frontend_dir = os.path.join(original_dir, "app", "frontend")
    
    try:
        if not os.path.exists(frontend_dir):
            print(f"Błąd: Katalog frontendu nie istnieje: {frontend_dir}")
            sys.exit(1)
            
        os.chdir(frontend_dir)
        print("Starting frontend development server...")
        
        # Dostosuj komendę w zależności od systemu operacyjnego
        if sys.platform.startswith('win'):
            subprocess.run(["npm.cmd", "start"], shell=True)
        else:
            subprocess.run(["npm", "start"], shell=True)
    except Exception as e:
        print(f"Wystąpił błąd podczas uruchamiania frontendu: {str(e)}")
        sys.exit(1)
    finally:
        # Wróć do oryginalnego katalogu
        os.chdir(original_dir)

def build_frontend():
    """Build the frontend for production."""
    # Sprawdź, czy npm jest zainstalowany
    if not shutil.which("npm"):
        print("Błąd: npm nie jest zainstalowany lub nie jest dostępny w ścieżce systemowej.")
        print("Zainstaluj Node.js i npm z https://nodejs.org/, a następnie spróbuj ponownie.")
        sys.exit(1)
    
    original_dir = os.getcwd()
    frontend_dir = os.path.join(original_dir, "app", "frontend")
    
    try:
        if not os.path.exists(frontend_dir):
            print(f"Błąd: Katalog frontendu nie istnieje: {frontend_dir}")
            sys.exit(1)
            
        os.chdir(frontend_dir)
        print("Building frontend for production...")
        
        # Dostosuj komendę w zależności od systemu operacyjnego
        if sys.platform.startswith('win'):
            subprocess.run(["npm.cmd", "run", "build"], shell=True)
        else:
            subprocess.run(["npm", "run", "build"], shell=True)
    except Exception as e:
        print(f"Wystąpił błąd podczas budowania frontendu: {str(e)}")
        sys.exit(1)
    finally:
        # Wróć do oryginalnego katalogu
        os.chdir(original_dir)

def start_docker():
    """Start the application using Docker Compose."""
    print("Starting the application with Docker Compose...")
    subprocess.run(["docker-compose", "up", "--build"], shell=True)

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

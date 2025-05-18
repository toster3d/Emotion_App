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

def setup_models():
    """Run the script to set up dummy models."""
    print("Setting up dummy models for testing...")
    subprocess.run([sys.executable, "setup_models.py"])

def start_backend(host: str = "127.0.0.1", port: int = 8000, reload: bool = True, background: bool = False) -> subprocess.Popen[bytes] | None:
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
            # On Windows, use a separate console
            startupinfo = None
            try:
                # Attempt to run in a separate window
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                backend_proc: subprocess.Popen[bytes] = subprocess.Popen(
                    cmd,
                    startupinfo=startupinfo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception:
                backend_proc: subprocess.Popen[bytes] = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            # Wait 5 seconds for the backend to start
            print("Waiting 5 seconds for the API to start...")
            time.sleep(5)

            # Open the API documentation page
            webbrowser.open(f"http://{host}:{port}/docs")
            print(f"Backend API running at: http://{host}:{port}")
            print(f"API documentation available at: http://{host}:{port}/docs")
            return backend_proc
        else:
            # On Linux/Mac
            backend_proc: subprocess.Popen[bytes] = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(5)
            webbrowser.open(f"http://{host}:{port}/docs")
            print(f"Backend API running at: http://{host}:{port}")
            return backend_proc
    else:
        # Run in the main process (blocking)
        subprocess.run(cmd)
        return None

def start_frontend():
    """Start the React frontend."""
    # Check if npm is installed
    if not shutil.which("npm"):
        print("Error: npm is not installed or not available in the system path.")
        print("Install Node.js and npm from https://nodejs.org/, and then try again.")
        sys.exit(1)

    # Save the original path and change directory
    original_dir = os.getcwd()
    frontend_dir = os.path.join(original_dir, "app", "frontend")

    try:
        if not os.path.exists(frontend_dir):
            print(f"Error: Frontend directory does not exist: {frontend_dir}")
            sys.exit(1)

        os.chdir(frontend_dir)
        print("Starting frontend development server...")

        # Ensure that the .env file exists and has the correct content
        backend_host = "127.0.0.1"  # Default host for backend
        backend_port = 8000         # Default port for backend
        
        print(f"Setting up .env with REACT_APP_API_URL=http://{backend_host}:{backend_port}/api/v1")
        with open('.env', 'w') as f:
            f.write(f"REACT_APP_API_URL=http://{backend_host}:{backend_port}/api/v1\n")

        # Adjust the command based on the operating system
        if sys.platform.startswith('win'):
            subprocess.run(["npm.cmd", "start"])
        else:
            subprocess.run(["npm", "start"])

    except Exception as e:
        print(f"An error occurred while starting the frontend: {str(e)}")
        sys.exit(1)
    finally:
        # Remove the .env file after completion
        env_path = os.path.join(frontend_dir, '.env')
        if os.path.exists(env_path):
            try:
                os.remove(env_path)
                print("Removed temporary .env file")
            except:
                pass
        # Return to the original directory
        os.chdir(original_dir)

def build_frontend():
    """Build the frontend for production."""
    # Check if npm is installed
    if not shutil.which("npm"):
        print("Error: npm is not installed or not available in the system path.")
        print("Install Node.js and npm from https://nodejs.org/, and then try again.")
        sys.exit(1)

    original_dir = os.getcwd()
    frontend_dir = os.path.join(original_dir, "app", "frontend")

    try:
        if not os.path.exists(frontend_dir):
            print(f"Error: Frontend directory does not exist: {frontend_dir}")
            sys.exit(1)

        os.chdir(frontend_dir)
        print("Building frontend for production...")

        # Adjust the command based on the operating system
        if sys.platform.startswith('win'):
            subprocess.run(["npm.cmd", "run", "build"], shell=True)
        else:
            subprocess.run(["npm", "run", "build"], shell=True)
    except Exception as e:
        print(f"An error occurred while building the frontend: {str(e)}")
        sys.exit(1)
    finally:
        # Return to the original directory
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
    subparsers.add_parser("setup", help="Set up dummy models for testing")

    # Backend command
    backend_parser = subparsers.add_parser("backend", help="Start the FastAPI backend server")
    backend_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    backend_parser.add_argument("--port", default=8000, type=int, help="Port to bind to")
    backend_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    # Frontend commands
    subparsers.add_parser("frontend", help="Start the React frontend development server")
    subparsers.add_parser("build-frontend", help="Build the React frontend for production")

    # Docker command
    subparsers.add_parser("docker", help="Start the application using Docker Compose")

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

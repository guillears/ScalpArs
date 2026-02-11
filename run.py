#!/usr/bin/env python3
"""
SCALPARS Trading Platform - Run Script
"""
import subprocess
import sys
import os

def main():
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 50)
    print("  SCALPARS Trading Platform")
    print("=" * 50)
    print()
    
    # Check if virtual environment exists
    venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        
        # Determine pip path
        if sys.platform == 'win32':
            pip_path = os.path.join(venv_path, 'Scripts', 'pip')
            python_path = os.path.join(venv_path, 'Scripts', 'python')
        else:
            pip_path = os.path.join(venv_path, 'bin', 'pip')
            python_path = os.path.join(venv_path, 'bin', 'python')
        
        print("Installing dependencies...")
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
    else:
        if sys.platform == 'win32':
            python_path = os.path.join(venv_path, 'Scripts', 'python')
        else:
            python_path = os.path.join(venv_path, 'bin', 'python')
    
    print()
    print("Starting SCALPARS Trading Platform...")
    print("Open http://localhost:8000 in your browser")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    print()
    
    # Run the FastAPI app
    subprocess.run([
        python_path, '-m', 'uvicorn', 
        'main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000',
        '--reload'
    ])

if __name__ == '__main__':
    main()

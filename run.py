#!/usr/bin/env python
"""
Run script for Enhanced Melanoma Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if virtual environment exists and packages are installed."""
    venv_path = Path('venv')
    
    if not venv_path.exists():
        print("Virtual environment not found. Creating one...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
        
        # Activate and install requirements
        if os.name == 'nt':  # Windows
            pip_path = venv_path / 'Scripts' / 'pip'
        else:  # Unix/Linux
            pip_path = venv_path / 'bin' / 'pip'
        
        print("Installing requirements...")
        subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'])
    
    return venv_path

def run_tests():
    """Run system tests."""
    print("\nRunning system tests...")
    if os.name == 'nt':  # Windows
        python_path = Path('venv') / 'Scripts' / 'python'
    else:
        python_path = Path('venv') / 'bin' / 'python'
    
    result = subprocess.run([str(python_path), 'test_system.py'])
    return result.returncode == 0

def run_app():
    """Run the Streamlit application."""
    print("\n" + "="*50)
    print("Starting Enhanced Melanoma Detection System...")
    print("="*50)
    
    if os.name == 'nt':  # Windows
        streamlit_path = Path('venv') / 'Scripts' / 'streamlit'
    else:
        streamlit_path = Path('venv') / 'bin' / 'streamlit'
    
    subprocess.run([str(streamlit_path), 'run', 'app.py'])

def main():
    """Main entry point."""
    print("="*50)
    print("Enhanced Melanoma Detection System Launcher")
    print("="*50)
    
    # Check and setup environment
    venv_path = check_requirements()
    
    # Ask if user wants to run tests
    response = input("\nRun system tests first? (recommended) [y/N]: ")
    
    if response.lower() == 'y':
        if not run_tests():
            print("\n⚠️ Some tests failed. Continue anyway? [y/N]: ")
            if input().lower() != 'y':
                sys.exit(1)
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main()

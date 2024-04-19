#!/bin/bash

# Change directory to server
cd server || exit

# Check if venv folder exists, if not, create it
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run the server with uvicorn
uvicorn main:app --reload

#!/bin/bash
source venv/bin/activate
pip install -r requirements.txt
echo "Virtual environment activated and dependencies installed."
echo "To start the server, run: uvicorn main:app --reload"
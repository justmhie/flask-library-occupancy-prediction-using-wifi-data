#!/bin/bash

# Quick Start Script for Library Occupancy Dashboard
# Automates the entire setup process

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Library Occupancy Dashboard - Quick Start Script      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "⚠ Node.js not found. Will skip frontend setup."
    echo "  Install Node.js from: https://nodejs.org/"
    SKIP_FRONTEND=true
else
    echo "✓ Node.js found: $(node --version)"
    SKIP_FRONTEND=false
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "STEP 1: Backend Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed"
else
    echo "✗ Failed to install Python dependencies"
    exit 1
fi

# Check if data file exists
if [ ! -f "all_data_cleaned.csv" ]; then
    echo "⚠ Warning: all_data_cleaned.csv not found"
    echo "  Please place your cleaned data file in this directory"
    echo "  Then run: python3 api_backend.py"
    exit 1
fi

echo "✓ Data file found: all_data_cleaned.csv"

# Start backend in background
echo ""
echo "Starting Flask backend..."
python3 api_backend.py &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
echo "  API available at: http://localhost:5000"

# Wait for backend to start
sleep 3

# Test API
echo ""
echo "Testing API..."
API_RESPONSE=$(curl -s http://localhost:5000/api/status)
if [ $? -eq 0 ]; then
    echo "✓ API is responding"
else
    echo "✗ API not responding"
    kill $BACKEND_PID
    exit 1
fi

if [ "$SKIP_FRONTEND" = true ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✓ Backend Setup Complete!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Backend is running at: http://localhost:5000"
    echo "API endpoints:"
    echo "  • http://localhost:5000/api/predictions"
    echo "  • http://localhost:5000/api/status"
    echo ""
    echo "To stop backend: kill $BACKEND_PID"
    echo ""
    echo "To set up frontend, install Node.js and run:"
    echo "  ./setup_frontend.sh"
    echo ""
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "STEP 2: Frontend Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if frontend directory exists
if [ ! -d "library-dashboard" ]; then
    echo "Creating React app..."
    npx create-react-app library-dashboard --silent
    
    if [ $? -eq 0 ]; then
        echo "✓ React app created"
    else
        echo "✗ Failed to create React app"
        kill $BACKEND_PID
        exit 1
    fi
else
    echo "✓ Frontend directory exists"
fi

# Navigate to frontend
cd library-dashboard

# Install dependencies
echo ""
echo "Installing frontend dependencies..."
npm install recharts --silent

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    cd ..
    kill $BACKEND_PID
    exit 1
fi

# Copy dashboard component
if [ -f "../react_dashboard.jsx" ]; then
    echo "✓ Copying dashboard component..."
    cp ../react_dashboard.jsx src/App.js
fi

# Update API URL in App.js
echo "✓ Configuring API endpoint..."
sed -i.bak "s|const response = await fetch('/api/predictions')|const response = await fetch('http://localhost:5000/api/predictions')|g" src/App.js

# Start frontend
echo ""
echo "Starting frontend..."
npm start &
FRONTEND_PID=$!

echo "✓ Frontend starting (PID: $FRONTEND_PID)"
echo "  Dashboard will open at: http://localhost:3000"

cd ..

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "🎉 Your dashboard is ready!"
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "The dashboard will:"
echo "  • Auto-refresh every 60 seconds"
echo "  • Update predictions automatically"
echo "  • Show graphs for each library"
echo ""
echo "Process IDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo "To stop all services:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Or run: ./stop_services.sh"
echo ""

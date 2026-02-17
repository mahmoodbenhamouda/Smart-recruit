#!/bin/bash

# TalentBridge - Quick Start Script (Mac/Linux)
# Run all services with one command

echo "========================================"
echo "  TalentBridge - Starting All Services"
echo "========================================"
echo ""

# Check if MongoDB is running
echo "[1/4] Checking MongoDB..."
if pgrep -x "mongod" > /dev/null; then
    echo "   ✅ MongoDB is running"
else
    echo "   ⚠️  MongoDB not running. Please start MongoDB first."
    echo "   Run: brew services start mongodb-community (Mac)"
    echo "   Or: sudo systemctl start mongod (Linux)"
    exit 1
fi

echo ""
echo "[2/4] Starting Python ATS API (Port 8000)..."
if [ -d "venv" ]; then
    source venv/bin/activate
fi
gnome-terminal -- bash -c "python ats_api_service.py; exec bash" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd \"'$(pwd)'\" && source venv/bin/activate && python ats_api_service.py"' 2>/dev/null || \
python ats_api_service.py &

sleep 5
echo "   ✅ Python ATS API started"

echo ""
echo "[3/4] Starting Node.js Backend (Port 3000)..."
gnome-terminal -- bash -c "cd server && npm run dev; exec bash" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd \"'$(pwd)'/server\" && npm run dev"' 2>/dev/null || \
(cd server && npm run dev &)

sleep 3
echo "   ✅ Node.js Backend started"

echo ""
echo "[4/4] Starting React Frontend (Port 5173)..."
gnome-terminal -- bash -c "cd client && npm run dev; exec bash" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd \"'$(pwd)'/client\" && npm run dev"' 2>/dev/null || \
(cd client && npm run dev &)

sleep 3
echo "   ✅ React Frontend started"

echo ""
echo "========================================"
echo "  🎉 All Services Started Successfully!"
echo "========================================"
echo ""
echo "📊 Service URLs:"
echo "   • Frontend:    http://localhost:5173"
echo "   • Backend API: http://localhost:3000"
echo "   • Python ATS:  http://localhost:8000"
echo "   • MongoDB:     mongodb://localhost:27017"
echo ""
echo "💡 To stop all services, close the terminal windows"
echo ""

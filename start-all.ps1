# TalentBridge - Quick Start Script
# Run all services with one command

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TalentBridge - Starting All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if MongoDB is running
Write-Host "[1/4] Checking MongoDB..." -ForegroundColor Yellow
$mongoProcess = Get-Process mongod -ErrorAction SilentlyContinue
if ($null -eq $mongoProcess) {
    Write-Host "   ⚠️  MongoDB not running. Please start MongoDB first." -ForegroundColor Red
    Write-Host "   Run: net start MongoDB" -ForegroundColor Gray
    exit 1
} else {
    Write-Host "   ✅ MongoDB is running" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/4] Starting Python ATS API (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; if (Test-Path venv\Scripts\Activate.ps1) { .\venv\Scripts\Activate.ps1 }; python ats_api_service.py"

Write-Host "   ⏳ Waiting 5 seconds for Python API to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 5
Write-Host "   ✅ Python ATS API started" -ForegroundColor Green

Write-Host ""
Write-Host "[3/4] Starting Node.js Backend (Port 3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\server'; npm run dev"

Write-Host "   ⏳ Waiting 3 seconds for Node.js to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 3
Write-Host "   ✅ Node.js Backend started" -ForegroundColor Green

Write-Host ""
Write-Host "[4/4] Starting React Frontend (Port 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\client'; npm run dev"

Write-Host "   ⏳ Waiting 3 seconds for Vite to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 3
Write-Host "   ✅ React Frontend started" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  🎉 All Services Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor Yellow
Write-Host "   • Frontend:    http://localhost:5173" -ForegroundColor White
Write-Host "   • Backend API: http://localhost:3000" -ForegroundColor White
Write-Host "   • Python ATS:  http://localhost:8000" -ForegroundColor White
Write-Host "   • MongoDB:     mongodb://localhost:27017" -ForegroundColor White
Write-Host ""
Write-Host "💡 To stop all services, close the terminal windows" -ForegroundColor Gray
Write-Host ""

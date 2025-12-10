# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Green
pip install -r requirements.txt

# Start the Flask server
Write-Host "`nStarting SentimemeNet backend server..." -ForegroundColor Green
Write-Host "The server will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press CTRL+C to stop the server`n" -ForegroundColor Yellow

python app.py

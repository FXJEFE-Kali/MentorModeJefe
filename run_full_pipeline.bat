@echo off
echo ============================================
echo  FXJEFE Beast Mode - Full Pipeline
echo  %date% %time%
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Training models (this takes 5-15 minutes)...
python train_beast_mode_fixed.py
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)
echo.

echo [2/3] Deploying models to MT5 terminals...
python model_deploy.py
if errorlevel 1 (
    echo ERROR: Deployment failed!
    pause
    exit /b 1
)
echo.

echo [3/3] Running quick model test...
python test_crypto_model.py
echo.

echo ============================================
echo  Pipeline complete!
echo  Check Logs/ for detailed training report
echo ============================================
pause

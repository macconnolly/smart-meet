@echo off
REM Windows Quick Setup for Cognitive Meeting Intelligence
REM This batch file provides easy setup options for Windows users

echo ========================================
echo Cognitive Meeting Intelligence Setup
echo ========================================
echo.
echo Choose your setup method:
echo 1. Docker (Recommended - No Python issues!)
echo 2. Quick Test with existing Python
echo 3. Full setup with helper script
echo 4. Emergency: Delete everything and start fresh
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto docker_setup
if "%choice%"=="2" goto quick_test
if "%choice%"=="3" goto helper_script
if "%choice%"=="4" goto clean_setup
goto end

:docker_setup
echo.
echo Starting Docker setup...
echo.
where docker >nul 2>1
if %errorlevel% neq 0 (
    echo ERROR: Docker not found!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    pause
    goto end
)

echo Starting containers...
docker-compose up -d
if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Docker containers are running.
    echo.
    echo To run tests:
    echo   docker-compose exec api pytest
    echo.
    echo To view API:
    echo   Open http://localhost:8000/docs
    echo.
    echo To stop:
    echo   docker-compose down
) else (
    echo ERROR: Failed to start Docker containers
)
pause
goto end

:quick_test
echo.
echo Running quick test...
python run_tests.py --minimal
pause
goto end

:helper_script
echo.
echo Running setup helper...
python setup_helper.py
pause
goto end

:clean_setup
echo.
echo WARNING: This will delete existing virtual environments!
set /p confirm="Are you sure? (yes/no): "
if not "%confirm%"=="yes" goto end

echo.
echo Cleaning up old environments...
if exist venv rmdir /s /q venv
if exist .venv rmdir /s /q .venv
if exist __pycache__ rmdir /s /q __pycache__

echo Creating fresh virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.11+ is installed
    pause
    goto end
)

echo Activating environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip wheel setuptools

echo Installing requirements...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Environment created.
    echo.
    echo To activate:
    echo   venv\Scripts\activate
    echo.
    echo To test:
    echo   python run_tests.py
) else (
    echo ERROR: Failed to install requirements
)
pause

:end

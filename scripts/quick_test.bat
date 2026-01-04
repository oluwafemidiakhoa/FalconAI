@echo off
REM Quick test script for FALCON-AI CLI (Windows)
REM Tests all major commands to verify installation

echo ==========================================
echo FALCON-AI CLI Quick Test
echo ==========================================
echo.

REM Test 1: Basic help
echo [1/6] Testing CLI help...
falcon-ai --help >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] CLI help works
) else (
    echo [ERROR] CLI help failed
    exit /b 1
)

REM Test 2: Run basic scenario
echo.
echo [2/6] Testing basic run command...
falcon-ai run --config configs/basic_config.yaml --length 50 --metrics %TEMP%\falcon_test_metrics.json >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Basic run works
) else (
    echo [ERROR] Basic run failed
    exit /b 1
)

REM Test 3: Benchmark
echo.
echo [3/6] Testing benchmark command...
falcon-ai benchmark --scenarios spike --decisions heuristic --energies simple --repeats 1 --output-dir %TEMP%\falcon_test_reports >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Benchmark works
) else (
    echo [ERROR] Benchmark failed
    exit /b 1
)

REM Test 4: Swarm demo
echo.
echo [4/6] Testing swarm demo...
falcon-ai swarm-demo --scenario spike --length 100 --agents 3 --output-dir %TEMP%\falcon_test_swarm >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Swarm demo works
) else (
    echo [ERROR] Swarm demo failed
    exit /b 1
)

REM Test 5: Check Python import
echo.
echo [5/6] Testing Python imports...
python -c "from falcon import FalconAI; from falcon.ml import NeuralPerception; from falcon.distributed import FalconSwarm; print('[OK] All imports successful')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python imports failed
    exit /b 1
)

REM Test 6: Verify config files
echo.
echo [6/6] Verifying config files...
dir /b configs\*.yaml 2>nul | find /c /v "" > %TEMP%\config_count.txt
set /p CONFIG_COUNT=<%TEMP%\config_count.txt
del %TEMP%\config_count.txt
if %CONFIG_COUNT% GEQ 5 (
    echo [OK] Found %CONFIG_COUNT% config files
) else (
    echo [WARNING] Only found %CONFIG_COUNT% config files (expected at least 5)
)

echo.
echo ==========================================
echo All Tests Passed!
echo ==========================================
echo.
echo Try these next steps:
echo   1. falcon-ai run --config configs/ml_config.yaml
echo   2. falcon-ai serve --config configs/wow.yaml --port 8000
echo   3. falcon-ai wow --config configs/wow.yaml
echo.

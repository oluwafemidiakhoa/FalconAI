#!/bin/bash
# Quick test script for FALCON-AI CLI
# Tests all major commands to verify installation

echo "=========================================="
echo "FALCON-AI CLI Quick Test"
echo "=========================================="
echo ""

# Test 1: Basic help
echo "[1/6] Testing CLI help..."
falcon-ai --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[OK] CLI help works"
else
    echo "[ERROR] CLI help failed"
    exit 1
fi

# Test 2: Run basic scenario
echo ""
echo "[2/6] Testing basic run command..."
falcon-ai run --config configs/basic_config.yaml --length 50 --metrics /tmp/falcon_test_metrics.json > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[OK] Basic run works"
else
    echo "[ERROR] Basic run failed"
    exit 1
fi

# Test 3: Benchmark
echo ""
echo "[3/6] Testing benchmark command..."
falcon-ai benchmark --scenarios spike --decisions heuristic --energies simple --repeats 1 --output-dir /tmp/falcon_test_reports > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[OK] Benchmark works"
else
    echo "[ERROR] Benchmark failed"
    exit 1
fi

# Test 4: Swarm demo
echo ""
echo "[4/6] Testing swarm demo..."
falcon-ai swarm-demo --scenario spike --length 100 --agents 3 --output-dir /tmp/falcon_test_swarm > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[OK] Swarm demo works"
else
    echo "[ERROR] Swarm demo failed"
    exit 1
fi

# Test 5: Check Python import
echo ""
echo "[5/6] Testing Python imports..."
python3 -c "from falcon import FalconAI; from falcon.ml import NeuralPerception; from falcon.distributed import FalconSwarm; print('[OK] All imports successful')"
if [ $? -ne 0 ]; then
    echo "[ERROR] Python imports failed"
    exit 1
fi

# Test 6: Verify config files
echo ""
echo "[6/6] Verifying config files..."
CONFIG_COUNT=$(ls configs/*.yaml 2>/dev/null | wc -l)
if [ $CONFIG_COUNT -ge 5 ]; then
    echo "[OK] Found $CONFIG_COUNT config files"
else
    echo "[WARNING] Only found $CONFIG_COUNT config files (expected at least 5)"
fi

echo ""
echo "=========================================="
echo "All Tests Passed!"
echo "=========================================="
echo ""
echo "Try these next steps:"
echo "  1. falcon-ai run --config configs/ml_config.yaml"
echo "  2. falcon-ai serve --config configs/wow.yaml --port 8000"
echo "  3. falcon-ai wow --config configs/wow.yaml"
echo ""

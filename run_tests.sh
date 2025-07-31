#!/bin/bash

# EvalSync Test Runner Script
# Execute various test suites with appropriate configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🧪 EvalSync Test Runner${NC}"
echo "================================"

# Parse command line arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:-}"

# Create reports directory if it doesn't exist
mkdir -p reports

# Function to run tests
run_tests() {
    local test_marker=$1
    local test_name=$2
    
    echo -e "\n${YELLOW}Running ${test_name} tests...${NC}"
    
    if [ "$VERBOSE" = "-v" ]; then
        pytest tests/ -m "$test_marker" -v
    else
        pytest tests/ -m "$test_marker" --tb=short
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${test_name} tests passed${NC}"
    else
        echo -e "${RED}❌ ${test_name} tests failed${NC}"
        exit 1
    fi
}

# Main test execution logic
case $TEST_TYPE in
    "integration")
        run_tests "integration" "Integration"
        ;;
    "fuzzing")
        run_tests "fuzzing" "Fuzzing"
        ;;
    "performance")
        run_tests "performance" "Performance"
        ;;
    "security")
        run_tests "security" "Security"
        ;;
    "smoke")
        run_tests "smoke" "Smoke"
        ;;
    "critical")
        run_tests "critical" "Critical"
        ;;
    "all")
        echo -e "${YELLOW}Running all tests...${NC}"
        if [ "$VERBOSE" = "-v" ]; then
            pytest tests/ -v
        else
            pytest tests/ --tb=short
        fi
        ;;
    "coverage")
        echo -e "${YELLOW}Running tests with coverage...${NC}"
        pytest tests/ --cov=src --cov-report=html:reports/coverage --cov-report=term
        echo -e "${GREEN}Coverage report generated in reports/coverage/index.html${NC}"
        ;;
    "quick")
        echo -e "${YELLOW}Running quick smoke tests...${NC}"
        pytest tests/ -m "smoke" --maxfail=1 -q
        ;;
    *)
        echo "Usage: $0 [test_type] [-v]"
        echo ""
        echo "Test types:"
        echo "  integration - Run integration tests"
        echo "  fuzzing     - Run fuzzing tests"
        echo "  performance - Run performance tests"
        echo "  security    - Run security tests"
        echo "  smoke       - Run smoke tests"
        echo "  critical    - Run critical tests only"
        echo "  all         - Run all tests (default)"
        echo "  coverage    - Run tests with coverage report"
        echo "  quick       - Run quick smoke tests"
        echo ""
        echo "Options:"
        echo "  -v          - Verbose output"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✨ Test execution completed${NC}"
echo -e "Reports available in: ${YELLOW}reports/${NC}"
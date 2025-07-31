"""
EvalSync Test Suites Package
Comprehensive test suites for LLM and API evaluation
"""

# Import all test suites for registration
from .integration_tests import IntegrationTestSuite
from .fuzzing_tests import FuzzingTestSuite
from .performance_tests import PerformanceTestSuite
from .schema_tests import SchemaTestSuite

__all__ = [
    "IntegrationTestSuite",
    "FuzzingTestSuite", 
    "PerformanceTestSuite",
    "SchemaTestSuite"
]
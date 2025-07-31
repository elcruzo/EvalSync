"""
Base Test Framework Components
Core classes and decorators for the EvalSync testing framework
"""

import time
import logging
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import functools

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """Test execution result with comprehensive metadata"""
    test_name: str
    status: TestStatus
    duration_ms: float
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    response_data: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Check if test passed"""
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """Check if test failed"""
        return self.status in [TestStatus.FAILED, TestStatus.ERROR]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'duration_ms': self.duration_ms,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'stack_trace': self.stack_trace,
            'response_data': self.response_data,
            'metadata': self.metadata
        }


@dataclass
class TestCase:
    """Test case metadata and configuration"""
    name: str
    category: str = "integration"
    priority: TestPriority = TestPriority.MEDIUM
    timeout: Optional[int] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    skip_condition: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


def test_case(category: str = "integration", 
              priority: TestPriority = TestPriority.MEDIUM,
              timeout: Optional[int] = None,
              retry_count: int = 0,
              retry_delay: float = 1.0,
              tags: Optional[List[str]] = None,
              description: Optional[str] = None,
              dependencies: Optional[List[str]] = None):
    """Decorator to mark methods as test cases with metadata"""
    
    def decorator(func: Callable) -> Callable:
        test_metadata = TestCase(
            name=func.__name__,
            category=category,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            tags=tags or [],
            description=description,
            dependencies=dependencies or []
        )
        
        # Attach metadata to function
        func._test_metadata = test_metadata
        func._is_test_case = True
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class BaseTestSuite(ABC):
    """Abstract base class for all test suites"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[TestResult] = []
        self.setup_complete = False
        self.teardown_complete = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.performance_metrics = {}
        self.benchmark_results = []
        
    def setup_suite(self):
        """Setup run once before all tests in the suite"""
        self.logger.info(f"Setting up test suite: {self.__class__.__name__}")
        self.setup_complete = True
    
    def teardown_suite(self):
        """Teardown run once after all tests in the suite"""
        self.logger.info(f"Tearing down test suite: {self.__class__.__name__}")
        self.teardown_complete = True
    
    def setup_method(self):
        """Setup run before each test method"""
        pass
    
    def teardown_method(self):
        """Teardown run after each test method"""
        pass
    
    def get_test_methods(self) -> List[Callable]:
        """Get all test methods in the suite"""
        test_methods = []
        
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if (callable(attr) and 
                hasattr(attr, '_is_test_case') and 
                attr._is_test_case):
                test_methods.append(attr)
        
        return test_methods
    
    def run_test_method(self, test_method: Callable) -> TestResult:
        """Execute a single test method with error handling and timing"""
        test_metadata = getattr(test_method, '_test_metadata', None)
        test_name = test_metadata.name if test_metadata else test_method.__name__
        
        start_time = time.time()
        
        try:
            # Setup method
            self.setup_method()
            
            # Execute test with timeout if specified
            if test_metadata and test_metadata.timeout:
                result = self._run_with_timeout(test_method, test_metadata.timeout)
            else:
                if asyncio.iscoroutinefunction(test_method):
                    result = asyncio.run(test_method())
                else:
                    result = test_method()
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Create successful result
            test_result = TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                response_data=result if isinstance(result, dict) else None,
                metadata={'category': test_metadata.category if test_metadata else 'unknown'}
            )
            
        except AssertionError as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            test_result = TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                error_type="AssertionError",
                stack_trace=traceback.format_exc()
            )
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            test_result = TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc()
            )
        
        finally:
            # Teardown method
            try:
                self.teardown_method()
            except Exception as e:
                self.logger.warning(f"Teardown failed for {test_name}: {e}")
        
        # Record result
        self.results.append(test_result)
        return test_result
    
    def _run_with_timeout(self, test_method: Callable, timeout: int) -> Any:
        """Run test method with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test {test_method.__name__} exceeded timeout of {timeout} seconds")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            if asyncio.iscoroutinefunction(test_method):
                return asyncio.run(asyncio.wait_for(test_method(), timeout=timeout))
            else:
                return test_method()
        finally:
            # Reset alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test methods in the suite"""
        if not self.setup_complete:
            self.setup_suite()
        
        test_methods = self.get_test_methods()
        
        # Sort by priority (critical first)
        def get_priority_order(method):
            metadata = getattr(method, '_test_metadata', None)
            if not metadata:
                return 2  # Medium priority
            
            priority_order = {
                TestPriority.CRITICAL: 0,
                TestPriority.HIGH: 1,
                TestPriority.MEDIUM: 2,
                TestPriority.LOW: 3
            }
            return priority_order.get(metadata.priority, 2)
        
        test_methods.sort(key=get_priority_order)
        
        results = []
        for test_method in test_methods:
            self.logger.info(f"Running test: {test_method.__name__}")
            result = self.run_test_method(test_method)
            results.append(result)
            
            # Log result
            if result.passed:
                self.logger.info(f"✅ {result.test_name} passed in {result.duration_ms:.1f}ms")
            else:
                self.logger.error(f"❌ {result.test_name} failed: {result.error_message}")
        
        if not self.teardown_complete:
            self.teardown_suite()
        
        return results
    
    def get_suite_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the test suite"""
        if not self.results:
            return {}
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if r.failed]
        
        return {
            'total_tests': len(self.results),
            'passed': len(passed_tests),
            'failed': len(failed_tests),
            'pass_rate': len(passed_tests) / len(self.results),
            'total_duration_ms': sum(r.duration_ms for r in self.results),
            'average_duration_ms': sum(r.duration_ms for r in self.results) / len(self.results),
            'fastest_test_ms': min(r.duration_ms for r in self.results),
            'slowest_test_ms': max(r.duration_ms for r in self.results)
        }
    
    def record_test_duration(self, duration: float):
        """Record test execution duration for performance tracking"""
        if 'execution_times' not in self.performance_metrics:
            self.performance_metrics['execution_times'] = []
        
        self.performance_metrics['execution_times'].append(duration)
    
    def record_benchmark_result(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """Record benchmark results"""
        benchmark_data = {
            'timestamp': time.time(),
            'metric': metric_name,
            'value': value,
            'metadata': metadata or {}
        }
        
        self.benchmark_results.append(benchmark_data)


class HealthCheckSuite(BaseTestSuite):
    """Basic health check test suite for service validation"""
    
    @test_case(category="health", priority=TestPriority.CRITICAL)
    def test_service_health_endpoints(self):
        """Test that all configured services have healthy endpoints"""
        from ..utils.client import LLMClient
        
        services = self.config.get('targets', {})
        health_results = {}
        
        for service_name, service_config in services.items():
            client = LLMClient(
                base_url=service_config['base_url'],
                timeout=10
            )
            
            try:
                # Try health endpoint first
                health_response = client.get('/health')
                if health_response.status_code == 200:
                    health_results[service_name] = 'healthy'
                else:
                    # Try basic endpoint
                    basic_response = client.generate("health check", max_tokens=10)
                    health_results[service_name] = 'healthy' if basic_response.status_code == 200 else 'unhealthy'
                    
            except Exception as e:
                health_results[service_name] = f'error: {str(e)}'
        
        # Assert all services are healthy
        unhealthy_services = [name for name, status in health_results.items() if status != 'healthy']
        
        assert not unhealthy_services, f"Unhealthy services detected: {unhealthy_services}"
        
        return health_results


class TestSuiteRegistry:
    """Registry for managing test suites"""
    
    def __init__(self):
        self._suites = {}
    
    def register(self, name: str, suite_class: type):
        """Register a test suite class"""
        if not issubclass(suite_class, BaseTestSuite):
            raise ValueError(f"Suite class must inherit from BaseTestSuite")
        
        self._suites[name] = suite_class
    
    def get_suite(self, name: str) -> Optional[type]:
        """Get a registered test suite class"""
        return self._suites.get(name)
    
    def list_suites(self) -> List[str]:
        """List all registered suite names"""
        return list(self._suites.keys())
    
    def create_suite(self, name: str, config: Dict[str, Any]) -> BaseTestSuite:
        """Create an instance of a registered test suite"""
        suite_class = self.get_suite(name)
        if not suite_class:
            raise ValueError(f"Unknown test suite: {name}")
        
        return suite_class(config)


# Global registry instance
suite_registry = TestSuiteRegistry()

# Register built-in suites
suite_registry.register("health_check", HealthCheckSuite)
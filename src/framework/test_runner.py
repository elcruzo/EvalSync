"""
Test Runner and Execution Engine
Orchestrates test execution, reporting, and result management
"""

import asyncio
import concurrent.futures
import time
import logging
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import json
import yaml
from dataclasses import asdict

from .base_test import BaseTestSuite, TestResult, TestStatus, TestPriority, suite_registry
# Import only if modules exist
try:
    from ..utils.reporters import HTMLReporter, JSONReporter, JUnitReporter
except ImportError:
    # Stub implementations
    class HTMLReporter:
        def __init__(self, path): self.path = path
        def generate_report(self, results): pass
    
    class JSONReporter:
        def __init__(self, path): self.path = path
        def generate_report(self, results): pass
    
    class JUnitReporter:
        def __init__(self, path): self.path = path
        def generate_report(self, results): pass

try:
    from ..utils.notifications import NotificationManager
except ImportError:
    class NotificationManager:
        def __init__(self, config): self.config = config
        def send_notification(self, message): pass

logger = logging.getLogger(__name__)


class TestExecutor:
    """Handles the execution of individual test suites"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_config = config.get('execution', {})
        self.parallel = self.execution_config.get('parallel', False)
        self.max_workers = self.execution_config.get('max_workers', 4)
        self.timeout = self.execution_config.get('timeout', 300)
        
    def execute_suite(self, suite: BaseTestSuite) -> List[TestResult]:
        """Execute a single test suite"""
        logger.info(f"Executing test suite: {suite.__class__.__name__}")
        
        start_time = time.time()
        results = suite.run_all_tests()
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Suite {suite.__class__.__name__} completed in {duration:.2f}s")
        
        return results
    
    def execute_suites_parallel(self, suites: List[BaseTestSuite]) -> Dict[str, List[TestResult]]:
        """Execute multiple test suites in parallel"""
        suite_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all suites for execution
            future_to_suite = {
                executor.submit(self.execute_suite, suite): suite 
                for suite in suites
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_suite, timeout=self.timeout):
                suite = future_to_suite[future]
                suite_name = suite.__class__.__name__
                
                try:
                    results = future.result()
                    suite_results[suite_name] = results
                    logger.info(f"✅ Suite {suite_name} completed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Suite {suite_name} failed: {e}")
                    # Create error result
                    error_result = TestResult(
                        test_name=f"{suite_name}_execution",
                        status=TestStatus.ERROR,
                        duration_ms=0,
                        start_time=time.time(),
                        end_time=time.time(),
                        error_message=str(e),
                        error_type=type(e).__name__
                    )
                    suite_results[suite_name] = [error_result]
        
        return suite_results
    
    def execute_suites_sequential(self, suites: List[BaseTestSuite]) -> Dict[str, List[TestResult]]:
        """Execute multiple test suites sequentially"""
        suite_results = {}
        
        for suite in suites:
            suite_name = suite.__class__.__name__
            try:
                results = self.execute_suite(suite)
                suite_results[suite_name] = results
                logger.info(f"✅ Suite {suite_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Suite {suite_name} failed: {e}")
                error_result = TestResult(
                    test_name=f"{suite_name}_execution",
                    status=TestStatus.ERROR,
                    duration_ms=0,
                    start_time=time.time(),
                    end_time=time.time(),
                    error_message=str(e),
                    error_type=type(e).__name__
                )
                suite_results[suite_name] = [error_result]
        
        return suite_results


class TestRunner:
    """Main test runner orchestrating the entire test execution process"""
    
    def __init__(self, config_path: str):
        """Initialize test runner with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.executor = TestExecutor(self.config)
        self.notification_manager = NotificationManager(self.config.get('notifications', {}))
        
        # Setup logging
        self._setup_logging()
        
        # Initialize reporters
        self.reporters = self._setup_reporters()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif self.config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_reporters(self) -> List[object]:
        """Setup test result reporters"""
        reporters = []
        reporting_config = self.config.get('reporting', {})
        
        formats = reporting_config.get('formats', ['html'])
        output_dir = Path(reporting_config.get('output_dir', 'reports'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'html' in formats:
            reporters.append(HTMLReporter(output_dir / 'test_report.html'))
        
        if 'json' in formats:
            reporters.append(JSONReporter(output_dir / 'test_results.json'))
            
        if 'junit' in formats:
            reporters.append(JUnitReporter(output_dir / 'junit_results.xml'))
        
        return reporters
    
    def discover_test_suites(self) -> List[str]:
        """Discover available test suites from configuration"""
        test_categories = self.config.get('test_categories', {})
        enabled_categories = [
            category for category, config in test_categories.items()
            if config.get('enabled', True)
        ]
        
        logger.info(f"Discovered test categories: {enabled_categories}")
        return enabled_categories
    
    def create_test_suites(self, categories: List[str]) -> List[BaseTestSuite]:
        """Create test suite instances for specified categories"""
        suites = []
        
        # Import test suites to register them
        self._import_test_suites()
        
        for category in categories:
            # Map category names to suite names
            suite_mapping = {
                'integration': 'integration',
                'fuzzing': 'fuzzing', 
                'performance': 'performance',
                'schema': 'schema',
                'health': 'health_check'
            }
            
            suite_name = suite_mapping.get(category, category)
            
            try:
                suite = suite_registry.create_suite(suite_name, self.config)
                suites.append(suite)
                logger.info(f"Created test suite: {suite_name}")
                
            except ValueError as e:
                logger.warning(f"Could not create suite '{suite_name}': {e}")
                
        return suites
    
    def _import_test_suites(self):
        """Import test suite modules to register them"""
        try:
            from ..test_suites.integration_tests import IntegrationTestSuite
            from ..test_suites.fuzzing_tests import FuzzingTestSuite
            from ..test_suites.performance_tests import PerformanceTestSuite
            from ..test_suites.schema_tests import SchemaTestSuite
            
            # Register suites
            suite_registry.register('integration', IntegrationTestSuite)
            suite_registry.register('fuzzing', FuzzingTestSuite)  
            suite_registry.register('performance', PerformanceTestSuite)
            suite_registry.register('schema', SchemaTestSuite)
            
        except ImportError as e:
            logger.warning(f"Could not import test suite modules: {e}")
    
    def run_tests(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run tests for specified categories or all enabled categories"""
        logger.info("🚀 Starting EvalSync test execution")
        
        start_time = time.time()
        
        # Discover and create test suites
        if categories is None:
            categories = self.discover_test_suites()
        
        suites = self.create_test_suites(categories)
        
        if not suites:
            logger.warning("No test suites to execute")
            return {'status': 'no_tests', 'results': {}}
        
        # Execute test suites
        logger.info(f"Executing {len(suites)} test suites")
        
        if self.executor.parallel and len(suites) > 1:
            suite_results = self.executor.execute_suites_parallel(suites)
        else:
            suite_results = self.executor.execute_suites_sequential(suites)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Compile overall results
        overall_results = self._compile_results(suite_results, total_duration)
        
        # Generate reports
        self._generate_reports(overall_results)
        
        # Send notifications
        self._send_notifications(overall_results)
        
        logger.info(f"✅ Test execution completed in {total_duration:.2f}s")
        
        return overall_results
    
    def _compile_results(self, suite_results: Dict[str, List[TestResult]], duration: float) -> Dict[str, Any]:
        """Compile overall test results and statistics"""
        all_results = []
        suite_summaries = {}
        
        for suite_name, results in suite_results.items():
            all_results.extend(results)
            
            # Calculate suite summary
            passed = len([r for r in results if r.passed])
            failed = len([r for r in results if r.failed])
            total = len(results)
            
            suite_summaries[suite_name] = {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / total if total > 0 else 0,
                'duration_ms': sum(r.duration_ms for r in results),
                'fastest_test_ms': min((r.duration_ms for r in results), default=0),
                'slowest_test_ms': max((r.duration_ms for r in results), default=0)
            }
        
        # Overall statistics
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.passed])
        failed_tests = len([r for r in all_results if r.failed])
        
        overall_results = {
            'timestamp': time.time(),
            'duration_seconds': duration,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration_ms': sum(r.duration_ms for r in all_results),
                'average_duration_ms': sum(r.duration_ms for r in all_results) / total_tests if total_tests > 0 else 0
            },
            'suite_summaries': suite_summaries,
            'detailed_results': {
                suite_name: [result.to_dict() for result in results]
                for suite_name, results in suite_results.items()
            },
            'status': 'passed' if failed_tests == 0 else 'failed'
        }
        
        return overall_results
    
    def _generate_reports(self, results: Dict[str, Any]):
        """Generate test reports in configured formats"""
        logger.info("Generating test reports...")
        
        for reporter in self.reporters:
            try:
                reporter.generate_report(results)
                logger.info(f"Generated {reporter.__class__.__name__} report")
                
            except Exception as e:
                logger.error(f"Failed to generate {reporter.__class__.__name__} report: {e}")
    
    def _send_notifications(self, results: Dict[str, Any]):
        """Send notifications about test results"""
        if results['summary']['failed'] > 0:
            self.notification_manager.send_failure_notification(results)
        else:
            self.notification_manager.send_success_notification(results)


class TestRunnerCLI:
    """Command-line interface for the test runner"""
    
    def __init__(self):
        self.runner = None
    
    def run_from_cli(self, args: List[str]):
        """Run tests from command line arguments"""
        import argparse
        
        parser = argparse.ArgumentParser(description="EvalSync Test Runner")
        parser.add_argument(
            '--config', '-c',
            type=str,
            default='config/test_config.yaml',
            help='Path to test configuration file'
        )
        parser.add_argument(
            '--categories',
            nargs='+',
            help='Test categories to run (default: all enabled)'
        )
        parser.add_argument(
            '--output-format',
            choices=['html', 'json', 'junit'],
            help='Output format (overrides config)'
        )
        parser.add_argument(
            '--output-file',
            type=str,
            help='Output file path (overrides config)'
        )
        parser.add_argument(
            '--parallel',
            action='store_true',
            help='Run test suites in parallel'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be executed without running tests'
        )
        
        parsed_args = parser.parse_args(args)
        
        # Initialize runner
        self.runner = TestRunner(parsed_args.config)
        
        # Override config with CLI arguments
        if parsed_args.parallel:
            self.runner.config['execution']['parallel'] = True
        
        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Dry run mode
        if parsed_args.dry_run:
            categories = parsed_args.categories or self.runner.discover_test_suites()
            logger.info("Dry run mode - would execute:")
            for category in categories:
                logger.info(f"  - {category}")
            return
        
        # Execute tests
        results = self.runner.run_tests(parsed_args.categories)
        
        # Exit with appropriate code
        exit_code = 0 if results['status'] == 'passed' else 1
        exit(exit_code)
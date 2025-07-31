"""
Performance Test Suite for EvalSync
Comprehensive performance testing including load, stress, and benchmark tests
"""

import asyncio
import concurrent.futures
import time
import statistics
import psutil
from typing import Dict, List, Any, Optional
import pytest

from ..framework.base_test import BaseTestSuite, test_case, TestPriority
from ..utils.client import LLMClient, MultiServiceClient


class PerformanceTestSuite(BaseTestSuite):
    """Performance and load testing suite for LLM services"""
    
    def setup_suite(self):
        """Setup performance test suite"""
        super().setup_suite()
        
        # Initialize clients
        self.clients = {}
        for service_name, service_config in self.config.get('targets', {}).items():
            self.clients[service_name] = LLMClient(
                base_url=service_config['base_url'],
                timeout=service_config.get('timeout', 60),  # Higher timeout for perf tests
                auth=service_config.get('auth')
            )
        
        # Multi-service client for comparative testing
        self.multi_client = MultiServiceClient(self.config.get('targets', {}))
        
        # Performance configuration
        self.perf_config = self.config.get('performance', {})
        self.thresholds = self.perf_config.get('thresholds', {})
        
        # Test data
        self.test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about technology.",
            "Describe the benefits of renewable energy.",
            "How does machine learning work?"
        ]
        
        # Performance tracking
        self.performance_data = {}
    
    @test_case(category="performance", priority=TestPriority.HIGH, tags=["latency", "sla"])
    def test_response_time_sla(self):
        """Test that response times meet SLA requirements"""
        max_response_time_ms = self.thresholds.get('max_response_time', 5000)
        
        for service_name, client in self.clients.items():
            response_times = []
            
            # Test multiple requests to get statistical significance
            for prompt in self.test_prompts:
                start_time = time.time()
                response = client.generate(prompt, max_tokens=50)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                # Check individual response
                assert response.status_code == 200, f"Request failed for {service_name}"
                assert response_time_ms < max_response_time_ms, (
                    f"{service_name} response time {response_time_ms:.0f}ms exceeds SLA {max_response_time_ms}ms"
                )
            
            # Statistical analysis
            avg_response_time = statistics.mean(response_times)
            p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
            
            # Store performance data
            self.performance_data[service_name] = {
                'avg_response_time_ms': avg_response_time,
                'p95_response_time_ms': p95_response_time,
                'max_response_time_ms': max(response_times),
                'min_response_time_ms': min(response_times)
            }
            
            self.logger.info(
                f"{service_name} performance: avg={avg_response_time:.1f}ms, p95={p95_response_time:.1f}ms"
            )
    
    @test_case(category="performance", priority=TestPriority.HIGH, tags=["throughput", "concurrent"])
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests"""
        concurrent_users = self.perf_config.get('concurrent_users', 10)
        min_success_rate = self.thresholds.get('min_success_rate', 0.95)
        
        for service_name, client in self.clients.items():
            def make_concurrent_request(request_id):
                try:
                    prompt = f"Concurrent test request {request_id}"
                    start_time = time.time()
                    response = client.generate(prompt, max_tokens=20)
                    end_time = time.time()
                    
                    return {
                        'request_id': request_id,
                        'status_code': response.status_code,
                        'response_time_ms': (end_time - start_time) * 1000,
                        'success': response.status_code == 200
                    }
                except Exception as e:
                    return {
                        'request_id': request_id,
                        'status_code': 0,
                        'response_time_ms': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Execute concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [
                    executor.submit(make_concurrent_request, i) 
                    for i in range(concurrent_users)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures, timeout=120):
                    result = future.result()
                    results.append(result)
            
            # Analyze results
            successful_requests = [r for r in results if r['success']]
            success_rate = len(successful_requests) / len(results)
            
            assert success_rate >= min_success_rate, (
                f"{service_name} concurrent success rate {success_rate:.1%} below threshold {min_success_rate:.1%}"
            )
            
            if successful_requests:
                response_times = [r['response_time_ms'] for r in successful_requests]
                avg_concurrent_response_time = statistics.mean(response_times)
                
                self.logger.info(
                    f"{service_name} concurrent test: {success_rate:.1%} success rate, "
                    f"avg response time {avg_concurrent_response_time:.1f}ms"
                )
    
    @test_case(category="performance", priority=TestPriority.MEDIUM, tags=["load", "sustained"])
    def test_sustained_load(self):
        """Test performance under sustained load"""
        duration_seconds = self.perf_config.get('load_test_duration', 60)
        requests_per_second = self.perf_config.get('requests_per_second', 5)
        max_error_rate = self.thresholds.get('max_error_rate', 0.05)
        
        for service_name, client in self.clients.items():
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            request_count = 0
            error_count = 0
            response_times = []
            
            self.logger.info(f"Starting {duration_seconds}s load test for {service_name}")
            
            while time.time() < end_time:
                batch_start = time.time()
                
                # Send batch of requests
                for _ in range(requests_per_second):
                    try:
                        request_start = time.time()
                        prompt = f"Load test request {request_count}"
                        response = client.generate(prompt, max_tokens=30)
                        request_end = time.time()
                        
                        request_count += 1
                        response_times.append((request_end - request_start) * 1000)
                        
                        if response.status_code != 200:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        self.logger.debug(f"Load test request failed: {e}")
                
                # Rate limiting - wait for remainder of second
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    time.sleep(1.0 - batch_duration)
            
            # Analyze load test results
            actual_duration = time.time() - start_time
            error_rate = error_count / request_count if request_count > 0 else 1.0
            
            assert error_rate <= max_error_rate, (
                f"{service_name} error rate {error_rate:.1%} exceeds threshold {max_error_rate:.1%}"
            )
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
                
                self.logger.info(
                    f"{service_name} load test completed: "
                    f"{request_count} requests in {actual_duration:.1f}s, "
                    f"error rate {error_rate:.1%}, "
                    f"avg response time {avg_response_time:.1f}ms"
                )
                
                # Store load test data
                if service_name not in self.performance_data:
                    self.performance_data[service_name] = {}
                
                self.performance_data[service_name]['load_test'] = {
                    'duration_seconds': actual_duration,
                    'total_requests': request_count,
                    'error_rate': error_rate,
                    'requests_per_second': request_count / actual_duration,
                    'avg_response_time_ms': avg_response_time,
                    'p95_response_time_ms': p95_response_time
                }
    
    @test_case(category="performance", priority=TestPriority.MEDIUM, tags=["memory", "resources"])
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable under load"""
        initial_memory = psutil.virtual_memory().used
        max_memory_increase = 1024 * 1024 * 1024  # 1GB threshold
        
        for service_name, client in self.clients.items():
            memory_samples = []
            
            # Make requests while monitoring memory
            for i in range(50):
                response = client.generate(f"Memory test request {i}", max_tokens=100)
                
                if i % 5 == 0:  # Sample memory every 5 requests
                    current_memory = psutil.virtual_memory().used
                    memory_increase = current_memory - initial_memory
                    memory_samples.append(memory_increase)
                    
                    assert memory_increase < max_memory_increase, (
                        f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB during test"
                    )
                
                # Small delay to allow garbage collection
                time.sleep(0.1)
            
            # Analyze memory trend
            if len(memory_samples) > 1:
                # Calculate trend (simple linear regression slope)
                x = list(range(len(memory_samples)))
                y = memory_samples
                
                n = len(x)
                slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (
                    n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2
                )
                
                # Memory should not grow excessively over time
                max_slope = 10 * 1024 * 1024  # 10MB per sample
                assert slope < max_slope, (
                    f"Memory usage growing too fast: {slope / 1024 / 1024:.1f}MB per sample"
                )
    
    @test_case(category="performance", priority=TestPriority.LOW, tags=["benchmark", "comparative"])
    def test_service_performance_comparison(self):
        """Compare performance across different services"""
        if len(self.clients) < 2:
            pytest.skip("Need at least 2 services for comparison")
        
        test_prompt = "Compare the performance of different AI models."
        service_results = {}
        
        # Test each service with the same prompt
        for service_name, client in self.clients.items():
            response_times = []
            
            # Multiple iterations for statistical significance
            for _ in range(5):
                start_time = time.time()
                response = client.generate(test_prompt, max_tokens=100)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append((end_time - start_time) * 1000)
            
            if response_times:
                service_results[service_name] = {
                    'avg_response_time_ms': statistics.mean(response_times),
                    'min_response_time_ms': min(response_times),
                    'max_response_time_ms': max(response_times),
                    'stddev_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0
                }
        
        # Analyze comparative performance
        if service_results:
            fastest_service = min(service_results.keys(), 
                                key=lambda s: service_results[s]['avg_response_time_ms'])
            slowest_service = max(service_results.keys(),
                                key=lambda s: service_results[s]['avg_response_time_ms'])
            
            fastest_time = service_results[fastest_service]['avg_response_time_ms']
            slowest_time = service_results[slowest_service]['avg_response_time_ms']
            
            self.logger.info(
                f"Performance comparison: {fastest_service} fastest ({fastest_time:.1f}ms), "
                f"{slowest_service} slowest ({slowest_time:.1f}ms)"
            )
            
            # Store comparison data
            self.performance_data['comparison'] = {
                'fastest_service': fastest_service,
                'slowest_service': slowest_service,
                'performance_ratio': slowest_time / fastest_time if fastest_time > 0 else 1,
                'service_results': service_results
            }
    
    @test_case(category="performance", priority=TestPriority.LOW, tags=["stress", "breaking_point"])
    def test_breaking_point_analysis(self):
        """Find the breaking point under extreme load"""
        for service_name, client in self.clients.items():
            max_concurrent = 100
            success_threshold = 0.5  # 50% success rate minimum
            
            # Binary search for breaking point
            low, high = 1, max_concurrent
            breaking_point = 1
            
            while low <= high:
                mid = (low + high) // 2
                success_rate = self._test_concurrent_load(client, mid, requests_per_user=3)
                
                if success_rate >= success_threshold:
                    breaking_point = mid
                    low = mid + 1
                else:
                    high = mid - 1
                
                self.logger.debug(
                    f"{service_name}: {mid} concurrent users -> {success_rate:.1%} success rate"
                )
            
            self.logger.info(
                f"{service_name} breaking point: {breaking_point} concurrent users "
                f"(>{success_threshold:.0%} success rate)"
            )
            
            # Store breaking point data
            if service_name not in self.performance_data:
                self.performance_data[service_name] = {}
            
            self.performance_data[service_name]['breaking_point'] = {
                'max_concurrent_users': breaking_point,
                'success_threshold': success_threshold
            }
    
    def _test_concurrent_load(self, client: LLMClient, concurrent_users: int, 
                            requests_per_user: int = 1) -> float:
        """Test concurrent load and return success rate"""
        def make_requests(user_id):
            successes = 0
            for i in range(requests_per_user):
                try:
                    response = client.generate(f"Stress test user {user_id} request {i}", 
                                             max_tokens=10)
                    if response.status_code == 200:
                        successes += 1
                except:
                    pass
            return successes
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            try:
                futures = [
                    executor.submit(make_requests, user_id)
                    for user_id in range(concurrent_users)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    result = future.result()
                    results.append(result)
                
                total_requests = concurrent_users * requests_per_user
                total_successes = sum(results)
                
                return total_successes / total_requests if total_requests > 0 else 0
                
            except concurrent.futures.TimeoutError:
                return 0  # Timeout indicates system overload
    
    def teardown_suite(self):
        """Cleanup and save performance data"""
        super().teardown_suite()
        
        # Close multi-client
        self.multi_client.close_all()
        
        # Log final performance summary
        if self.performance_data:
            self.logger.info("=== Performance Test Summary ===")
            for service_name, data in self.performance_data.items():
                if isinstance(data, dict) and 'avg_response_time_ms' in data:
                    self.logger.info(
                        f"{service_name}: avg={data['avg_response_time_ms']:.1f}ms, "
                        f"p95={data.get('p95_response_time_ms', 'N/A')}"
                    )
        
        # Save detailed performance data for analysis
        self.record_benchmark_result('performance_suite_summary', 0, self.performance_data)
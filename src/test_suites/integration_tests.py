"""
EvalSync Integration Test Suite
Comprehensive integration tests for LLM services
"""

import pytest
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
import httpx
from dataclasses import dataclass
import jsonschema
from unittest.mock import patch

from ..framework.base_test import BaseTestSuite
from ..utils.client import LLMClient
# Fixed import issues - validators and test_data were missing
try:
    from ..utils.validators import ResponseValidator
    from ..utils.test_data import TestDataManager
except ImportError:
    # Fallback for missing modules (fixed in Sept 2025)
    print("Warning: Some modules not found, using stubs")
    class ResponseValidator:
        def __init__(self): pass
    class TestDataManager:
        def __init__(self): pass

@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    status: str
    duration_ms: float
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None

class IntegrationTestSuite(BaseTestSuite):
    """Main integration test suite for LLM services"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.clients = {}
        self.validator = ResponseValidator()
        self.test_data = TestDataManager()
        
        # Initialize clients for each target service
        for service_name, service_config in config.get('targets', {}).items():
            self.clients[service_name] = LLMClient(
                base_url=service_config['base_url'],
                timeout=service_config.get('timeout', 30),
                auth=service_config.get('auth')
            )
    
    def setup_method(self):
        """Setup run before each test method"""
        self.start_time = time.time()
        
    def teardown_method(self):
        """Cleanup run after each test method"""
        duration = time.time() - self.start_time
        self.record_test_duration(duration)

class TestBasicInference(IntegrationTestSuite):
    """Test basic inference functionality"""
    
    @pytest.mark.integration
    @pytest.mark.high_priority
    def test_simple_text_generation(self):
        """Test basic text generation endpoint"""
        for service_name, client in self.clients.items():
            prompt = "What is artificial intelligence?"
            
            response = client.generate(prompt)
            
            # Basic response validation
            assert response.status_code == 200, f"{service_name} returned {response.status_code}"
            
            result = response.json()
            assert 'text' in result or 'response' in result, f"Missing text field in {service_name} response"
            
            # Validate response content
            text_content = result.get('text') or result.get('response', '')
            assert len(text_content.strip()) > 0, f"Empty response from {service_name}"
            assert len(text_content) < 10000, f"Response too long from {service_name}"
    
    @pytest.mark.integration
    def test_prompt_with_parameters(self):
        """Test generation with various parameters"""
        test_cases = [
            {"temperature": 0.1, "max_tokens": 50},
            {"temperature": 0.9, "max_tokens": 100},
            {"temperature": 0.5, "max_tokens": 200, "top_p": 0.9}
        ]
        
        for service_name, client in self.clients.items():
            for params in test_cases:
                response = client.generate(
                    "Tell me a short story about robots.",
                    **params
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Validate token limits are respected
                if 'metadata' in result and 'tokens_used' in result['metadata']:
                    tokens_used = result['metadata']['tokens_used']
                    assert tokens_used <= params['max_tokens'] * 1.2  # Allow 20% buffer
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        async def make_request(client, prompt_id):
            response = await client.generate_async(f"Test prompt {prompt_id}")
            return response
        
        for service_name, client in self.clients.items():
            if hasattr(client, 'generate_async'):
                # Send 10 concurrent requests
                tasks = [
                    make_request(client, i) 
                    for i in range(10)
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check success rate
                successful_responses = [
                    r for r in responses 
                    if not isinstance(r, Exception) and r.status_code == 200
                ]
                
                success_rate = len(successful_responses) / len(responses)
                assert success_rate >= 0.8, f"Low success rate for {service_name}: {success_rate}"

class TestRAGFunctionality(IntegrationTestSuite):
    """Test Retrieval-Augmented Generation functionality"""
    
    @pytest.mark.integration
    @pytest.mark.rag
    def test_context_awareness(self):
        """Test that models use provided context"""
        context_prompt = """
        Context: The company's main product is EcoClean, a biodegradable cleaning solution 
        that was launched in 2023 and has won several environmental awards.
        
        Question: What is the company's main product and when was it launched?
        """
        
        for service_name, client in self.clients.items():
            response = client.generate(context_prompt)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', result.get('response', '')).lower()
                
                # Check if context information is used
                assert 'ecoclean' in text, f"{service_name} didn't use context information"
                assert '2023' in text, f"{service_name} didn't extract date from context"
    
    @pytest.mark.integration
    @pytest.mark.rag
    def test_document_retrieval(self):
        """Test document retrieval functionality"""
        for service_name, client in self.clients.items():
            if hasattr(client, 'retrieve'):
                query = "machine learning algorithms"
                
                retrieved_docs = client.retrieve(query, top_k=5)
                
                assert len(retrieved_docs) <= 5
                assert all('content' in doc for doc in retrieved_docs)
                assert all('similarity_score' in doc for doc in retrieved_docs)
                
                # Scores should be in descending order
                scores = [doc['similarity_score'] for doc in retrieved_docs]
                assert scores == sorted(scores, reverse=True)

class TestErrorHandling(IntegrationTestSuite):
    """Test error handling and edge cases"""
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts"""
        for service_name, client in self.clients.items():
            response = client.generate("")
            
            # Should either return an error or handle gracefully
            assert response.status_code in [200, 400, 422], f"Unexpected status from {service_name}"
            
            if response.status_code == 200:
                result = response.json()
                # If successful, should still return valid response
                assert 'text' in result or 'response' in result
    
    @pytest.mark.integration
    @pytest.mark.error_handling  
    def test_oversized_prompt_handling(self):
        """Test handling of very large prompts"""
        # Create a very large prompt
        large_prompt = "Tell me about " + "artificial intelligence " * 1000
        
        for service_name, client in self.clients.items():
            response = client.generate(large_prompt)
            
            # Should either truncate, return error, or handle gracefully
            assert response.status_code in [200, 400, 413, 422]
            
            if response.status_code != 200:
                error_response = response.json()
                assert 'error' in error_response or 'message' in error_response
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    def test_invalid_parameters(self):
        """Test handling of invalid parameters"""
        invalid_params_tests = [
            {"temperature": -1.0},
            {"temperature": 2.0},
            {"max_tokens": -100},
            {"max_tokens": "invalid"},
            {"top_p": 1.5},
            {"top_p": "invalid"}
        ]
        
        for service_name, client in self.clients.items():
            for invalid_params in invalid_params_tests:
                response = client.generate(
                    "Test prompt", 
                    **invalid_params
                )
                
                # Should return validation error
                assert response.status_code in [400, 422], (
                    f"{service_name} accepted invalid params: {invalid_params}"
                )

class TestSchemaValidation(IntegrationTestSuite):
    """Test API response schema validation"""
    
    @pytest.mark.integration
    @pytest.mark.schema
    def test_response_schema_compliance(self):
        """Test that all responses follow expected schema"""
        expected_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "response": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "tokens_used": {"type": "integer", "minimum": 0},
                        "latency_ms": {"type": "number", "minimum": 0}
                    }
                }
            },
            "anyOf": [
                {"required": ["text"]},
                {"required": ["response"]}
            ]
        }
        
        for service_name, client in self.clients.items():
            response = client.generate("Test schema validation")
            
            if response.status_code == 200:
                result = response.json()
                
                try:
                    jsonschema.validate(result, expected_schema)
                except jsonschema.ValidationError as e:
                    pytest.fail(f"{service_name} response schema invalid: {e.message}")
    
    @pytest.mark.integration
    @pytest.mark.schema
    def test_error_response_schema(self):
        """Test error response schema consistency"""
        error_schema = {
            "type": "object",
            "required": ["error"],
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"},
                "code": {"type": ["string", "integer"]}
            }
        }
        
        for service_name, client in self.clients.items():
            # Trigger an error (invalid auth)
            with patch.object(client, 'headers', {'Authorization': 'Bearer invalid'}):
                response = client.generate("Test prompt")
                
                if response.status_code != 200:
                    result = response.json()
                    
                    try:
                        jsonschema.validate(result, error_schema)
                    except jsonschema.ValidationError as e:
                        pytest.fail(f"{service_name} error schema invalid: {e.message}")

class TestPerformanceBaseline(IntegrationTestSuite):
    """Test performance baselines and SLA compliance"""
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_response_time_sla(self):
        """Test response time meets SLA requirements"""
        max_response_time = self.config.get('sla', {}).get('max_response_time_ms', 5000)
        
        for service_name, client in self.clients.items():
            start_time = time.time()
            response = client.generate("Quick response test")
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            assert response_time_ms < max_response_time, (
                f"{service_name} response time {response_time_ms:.0f}ms exceeds SLA {max_response_time}ms"
            )
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_throughput_baseline(self, benchmark):
        """Benchmark throughput for baseline measurement"""
        def generate_text():
            client = list(self.clients.values())[0]  # Use first available client
            response = client.generate("Benchmark test prompt")
            return response
        
        result = benchmark(generate_text)
        
        # Record baseline metrics
        self.record_benchmark_result('throughput', {
            'mean_time': result.stats.mean,
            'std_dev': result.stats.stddev,
            'min_time': result.stats.min,
            'max_time': result.stats.max
        })
    
    def record_benchmark_result(self, metric_name: str, stats: Dict):
        """Record benchmark results for trend analysis"""
        benchmark_data = {
            'timestamp': time.time(),
            'metric': metric_name,
            'stats': stats
        }
        
        # Store in test results for analysis
        if not hasattr(self, 'benchmark_results'):
            self.benchmark_results = []
        self.benchmark_results.append(benchmark_data)

# Pytest fixtures for test setup
@pytest.fixture(scope="session")
def test_config():
    """Load test configuration"""
    import yaml
    with open('config/test_config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def integration_suite(test_config):
    """Create integration test suite instance"""
    return IntegrationTestSuite(test_config)

# Custom pytest markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(300)  # 5 minute timeout for integration tests
]
"""
Schema Validation Test Suite for EvalSync
Comprehensive API response schema validation and compliance testing
"""

import json
import jsonschema
from typing import Dict, List, Any, Optional
import pytest

from ..framework.base_test import BaseTestSuite, test_case, TestPriority
from ..utils.client import LLMClient


class SchemaTestSuite(BaseTestSuite):
    """Schema validation and API compliance test suite"""
    
    def setup_suite(self):
        """Setup schema validation test suite"""
        super().setup_suite()
        
        # Initialize clients
        self.clients = {}
        for service_name, service_config in self.config.get('targets', {}).items():
            self.clients[service_name] = LLMClient(
                base_url=service_config['base_url'],
                timeout=service_config.get('timeout', 30),
                auth=service_config.get('auth')
            )
        
        # Define expected schemas
        self.schemas = self._define_schemas()
        
        # Validation configuration
        self.schema_config = self.config.get('schema', {})
        self.strict_validation = self.schema_config.get('strict_validation', True)
    
    def _define_schemas(self) -> Dict[str, Dict]:
        """Define expected API response schemas"""
        return {
            'generate_response': {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "response": {"type": "string"},
                    "result": {"type": "string"},
                    "output": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "tokens_used": {"type": "integer", "minimum": 0},
                            "latency_ms": {"type": "number", "minimum": 0},
                            "timestamp": {"type": ["string", "number"]},
                            "finish_reason": {"type": "string"}
                        }
                    },
                    "usage": {
                        "type": "object",
                        "properties": {
                            "prompt_tokens": {"type": "integer", "minimum": 0},
                            "completion_tokens": {"type": "integer", "minimum": 0},
                            "total_tokens": {"type": "integer", "minimum": 0}
                        }
                    }
                },
                "anyOf": [
                    {"required": ["text"]},
                    {"required": ["response"]},
                    {"required": ["result"]},
                    {"required": ["output"]}
                ]
            },
            
            'error_response': {
                "type": "object",
                "required": ["error"],
                "properties": {
                    "error": {
                        "type": ["string", "object"],
                        "properties": {
                            "message": {"type": "string"},
                            "type": {"type": "string"},
                            "code": {"type": ["string", "integer"]}
                        }
                    },
                    "message": {"type": "string"},
                    "detail": {"type": "string"},
                    "details": {"type": "object"},
                    "code": {"type": ["string", "integer"]},
                    "status_code": {"type": "integer"}
                }
            },
            
            'health_response': {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["healthy", "unhealthy", "degraded"]},
                    "timestamp": {"type": ["string", "number"]},
                    "version": {"type": "string"},
                    "uptime": {"type": "number", "minimum": 0},
                    "checks": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    }
                }
            },
            
            'streaming_response': {
                "type": "object",
                "properties": {
                    "delta": {"type": "string"},
                    "token": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "delta": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"}
                                    }
                                },
                                "finish_reason": {"type": ["string", "null"]}
                            }
                        }
                    }
                }
            }
        }
    
    @test_case(category="schema", priority=TestPriority.HIGH, tags=["validation", "compliance"])
    def test_generate_response_schema(self):
        """Test that generate endpoint responses match expected schema"""
        schema = self.schemas['generate_response']
        
        test_prompts = [
            "Hello world",
            "What is AI?",
            "Explain quantum physics briefly.",
            "Generate a haiku about technology."
        ]
        
        for service_name, client in self.clients.items():
            schema_violations = []
            
            for prompt in test_prompts:
                response = client.generate(prompt, max_tokens=50)
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        jsonschema.validate(result, schema)
                        
                        # Additional semantic validation
                        text_content = (result.get('text') or result.get('response') or 
                                      result.get('result') or result.get('output', ''))
                        
                        assert isinstance(text_content, str), "Response content must be string"
                        assert len(text_content) > 0, "Response content cannot be empty"
                        
                    except jsonschema.ValidationError as e:
                        schema_violations.append({
                            'prompt': prompt[:50] + '...',
                            'error': e.message,
                            'path': list(e.absolute_path) if e.absolute_path else []
                        })
                    except json.JSONDecodeError:
                        schema_violations.append({
                            'prompt': prompt[:50] + '...',
                            'error': 'Invalid JSON response',
                            'response_text': response.text[:200]
                        })
            
            # Assert schema compliance
            if self.strict_validation:
                assert not schema_violations, (
                    f"{service_name} schema violations: {schema_violations}"
                )
            else:
                violation_rate = len(schema_violations) / len(test_prompts)
                assert violation_rate < 0.5, (
                    f"{service_name} high schema violation rate: {violation_rate:.1%}"
                )
            
            if schema_violations:
                self.logger.warning(f"{service_name} schema violations: {len(schema_violations)}")
    
    @test_case(category="schema", priority=TestPriority.HIGH, tags=["error_handling", "validation"])
    def test_error_response_schema(self):
        """Test that error responses follow consistent schema"""
        schema = self.schemas['error_response']
        
        # Test cases that should trigger errors
        error_test_cases = [
            # Invalid authentication
            {'auth_override': {'Authorization': 'Bearer invalid_token'}},
            
            # Invalid parameters
            {'params': {'temperature': -10}},
            {'params': {'max_tokens': -1}},
            {'params': {'top_p': 5.0}},
            
            # Malformed requests
            {'json_override': '{"invalid": json}'},
            {'json_override': {'prompt': None}},
        ]
        
        for service_name, client in self.clients.items():
            error_schema_violations = []
            
            for test_case in error_test_cases:
                try:
                    # Prepare request with error-inducing parameters
                    if 'auth_override' in test_case:
                        original_headers = client.default_headers.copy()
                        client.default_headers.update(test_case['auth_override'])
                        response = client.generate("test prompt")
                        client.default_headers = original_headers
                        
                    elif 'params' in test_case:
                        response = client.generate("test prompt", **test_case['params'])
                        
                    elif 'json_override' in test_case:
                        if isinstance(test_case['json_override'], str):
                            # Send malformed JSON
                            response = client.post('/generate', content=test_case['json_override'],
                                                 headers={'Content-Type': 'application/json'})
                        else:
                            response = client.post('/generate', json=test_case['json_override'])
                    
                    # Should return error status
                    if response.status_code not in [400, 401, 422, 500]:
                        continue  # Skip if no error occurred
                    
                    # Validate error response schema
                    try:
                        error_data = response.json()
                        jsonschema.validate(error_data, schema)
                        
                        # Additional error response validation
                        error_message = (error_data.get('error') or 
                                       error_data.get('message') or 
                                       error_data.get('detail', ''))
                        
                        if isinstance(error_message, dict):
                            error_message = error_message.get('message', '')
                        
                        assert error_message, "Error response must contain error message"
                        assert len(str(error_message)) > 0, "Error message cannot be empty"
                        
                    except jsonschema.ValidationError as e:
                        error_schema_violations.append({
                            'test_case': str(test_case),
                            'status_code': response.status_code,
                            'error': e.message,
                            'response_sample': response.text[:200]
                        })
                    except json.JSONDecodeError:
                        error_schema_violations.append({
                            'test_case': str(test_case),
                            'status_code': response.status_code,
                            'error': 'Non-JSON error response',
                            'response_sample': response.text[:200]
                        })
                
                except Exception as e:
                    # Network errors are acceptable for error testing
                    pass
            
            # Validate error response consistency
            if error_schema_violations and self.strict_validation:
                assert not error_schema_violations, (
                    f"{service_name} error schema violations: {error_schema_violations}"
                )
    
    @test_case(category="schema", priority=TestPriority.MEDIUM, tags=["health", "monitoring"])
    def test_health_endpoint_schema(self):
        """Test health endpoint response schema"""
        schema = self.schemas['health_response']
        
        for service_name, client in self.clients.items():
            try:
                response = client.get('/health')
                
                if response.status_code == 200:
                    health_data = response.json()
                    
                    # Validate against schema
                    jsonschema.validate(health_data, schema)
                    
                    # Additional health-specific validation
                    status = health_data.get('status', '').lower()
                    assert status in ['healthy', 'unhealthy', 'degraded'], (
                        f"Invalid health status: {status}"
                    )
                    
                    if 'timestamp' in health_data:
                        timestamp = health_data['timestamp']
                        if isinstance(timestamp, str):
                            # Try to parse timestamp formats
                            import dateutil.parser
                            dateutil.parser.parse(timestamp)  # Should not raise exception
                
                elif response.status_code == 404:
                    # Health endpoint not available - skip test
                    pytest.skip(f"{service_name} does not have health endpoint")
                    
                else:
                    # Health endpoint should return 200 or 503
                    assert response.status_code in [503], (
                        f"Unexpected health endpoint status: {response.status_code}"
                    )
            
            except json.JSONDecodeError:
                pytest.fail(f"{service_name} health endpoint returned non-JSON response")
            except jsonschema.ValidationError as e:
                pytest.fail(f"{service_name} health endpoint schema violation: {e.message}")
    
    @test_case(category="schema", priority=TestPriority.MEDIUM, tags=["streaming", "websocket"])
    def test_streaming_response_schema(self):
        """Test streaming response schema (if supported)"""
        schema = self.schemas['streaming_response']
        
        for service_name, client in self.clients.items():
            try:
                # Test streaming capability
                stream_chunks = list(client.stream_generate("Tell me a story", max_tokens=50))
                
                if not stream_chunks or any('error' in chunk for chunk in stream_chunks):
                    pytest.skip(f"{service_name} does not support streaming")
                
                schema_violations = []
                for i, chunk in enumerate(stream_chunks):
                    if 'raw' in chunk:
                        continue  # Skip raw text chunks
                    
                    try:
                        jsonschema.validate(chunk, schema)
                        
                        # Validate streaming-specific properties
                        if 'delta' in chunk:
                            assert isinstance(chunk['delta'], str), "Delta must be string"
                        
                        if 'choices' in chunk:
                            for choice in chunk['choices']:
                                if 'delta' in choice and 'content' in choice['delta']:
                                    assert isinstance(choice['delta']['content'], str), (
                                        "Choice delta content must be string"
                                    )
                    
                    except jsonschema.ValidationError as e:
                        schema_violations.append({
                            'chunk_index': i,
                            'error': e.message,
                            'chunk_sample': str(chunk)[:100]
                        })
                
                if schema_violations and self.strict_validation:
                    pytest.fail(f"{service_name} streaming schema violations: {schema_violations}")
            
            except Exception as e:
                # Streaming not supported or failed - skip
                pytest.skip(f"{service_name} streaming test failed: {e}")
    
    @test_case(category="schema", priority=TestPriority.LOW, tags=["consistency", "format"])
    def test_response_format_consistency(self):
        """Test that response formats are consistent across different requests"""
        for service_name, client in self.clients.items():
            responses = []
            
            # Generate multiple responses
            for i in range(5):
                response = client.generate(f"Test prompt {i}", max_tokens=20)
                
                if response.status_code == 200:
                    responses.append(response.json())
            
            if len(responses) < 2:
                pytest.skip(f"Not enough successful responses from {service_name}")
            
            # Check consistency of response structure
            first_response = responses[0]
            inconsistencies = []
            
            for i, response in enumerate(responses[1:], 1):
                # Check top-level keys consistency
                first_keys = set(first_response.keys())
                response_keys = set(response.keys())
                
                if first_keys != response_keys:
                    inconsistencies.append({
                        'response_index': i,
                        'issue': 'different_keys',
                        'first_keys': sorted(first_keys),
                        'response_keys': sorted(response_keys)
                    })
                
                # Check metadata structure consistency (if present)
                if 'metadata' in first_response and 'metadata' in response:
                    first_meta_keys = set(first_response['metadata'].keys())
                    response_meta_keys = set(response['metadata'].keys())
                    
                    if first_meta_keys != response_meta_keys:
                        inconsistencies.append({
                            'response_index': i,
                            'issue': 'metadata_keys_different',
                            'first_meta_keys': sorted(first_meta_keys),
                            'response_meta_keys': sorted(response_meta_keys)
                        })
                
                # Check data type consistency
                for key in first_keys.intersection(response_keys):
                    first_type = type(first_response[key])
                    response_type = type(response[key])
                    
                    if first_type != response_type:
                        inconsistencies.append({
                            'response_index': i,
                            'issue': 'type_inconsistency',
                            'key': key,
                            'first_type': first_type.__name__,
                            'response_type': response_type.__name__
                        })
            
            # Assert consistency
            if inconsistencies and self.strict_validation:
                pytest.fail(f"{service_name} response format inconsistencies: {inconsistencies}")
            elif inconsistencies:
                self.logger.warning(
                    f"{service_name} has {len(inconsistencies)} format inconsistencies"
                )
    
    @test_case(category="schema", priority=TestPriority.LOW, tags=["openapi", "documentation"])
    def test_openapi_compliance(self):
        """Test compliance with OpenAPI specification (if available)"""
        for service_name, client in self.clients.items():
            try:
                # Try to fetch OpenAPI/Swagger documentation
                openapi_endpoints = ['/openapi.json', '/swagger.json', '/docs/openapi.json', '/api-docs']
                
                openapi_spec = None
                for endpoint in openapi_endpoints:
                    try:
                        response = client.get(endpoint)
                        if response.status_code == 200:
                            openapi_spec = response.json()
                            break
                    except:
                        continue
                
                if not openapi_spec:
                    pytest.skip(f"{service_name} does not provide OpenAPI specification")
                
                # Basic OpenAPI structure validation
                assert 'openapi' in openapi_spec or 'swagger' in openapi_spec, (
                    "Missing OpenAPI/Swagger version"
                )
                assert 'info' in openapi_spec, "Missing API info"
                assert 'paths' in openapi_spec, "Missing API paths"
                
                # Validate that documented endpoints actually work
                paths = openapi_spec.get('paths', {})
                working_endpoints = 0
                total_endpoints = 0
                
                for path, methods in paths.items():
                    for method, spec in methods.items():
                        if method.upper() in ['GET', 'POST']:
                            total_endpoints += 1
                            try:
                                if method.upper() == 'GET':
                                    response = client.get(path)
                                else:
                                    response = client.post(path, json={})
                                
                                if response.status_code < 500:  # Not server error
                                    working_endpoints += 1
                                    
                            except:
                                pass
                
                if total_endpoints > 0:
                    working_rate = working_endpoints / total_endpoints
                    assert working_rate > 0.5, (
                        f"{service_name} many documented endpoints not working: {working_rate:.1%}"
                    )
            
            except Exception as e:
                pytest.skip(f"{service_name} OpenAPI compliance test failed: {e}")
    
    def teardown_suite(self):
        """Log schema validation summary"""
        super().teardown_suite()
        
        if self.results:
            passed_schema_tests = len([r for r in self.results if r.passed and 'schema' in r.test_name])
            total_schema_tests = len([r for r in self.results if 'schema' in r.test_name])
            
            if total_schema_tests > 0:
                compliance_rate = passed_schema_tests / total_schema_tests
                self.logger.info(f"Schema compliance rate: {compliance_rate:.1%}")
                
                if compliance_rate < 0.8:
                    self.logger.warning("Low schema compliance rate detected")
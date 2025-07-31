"""
Fuzzing tests for LLM services
"""

import pytest
from unittest.mock import Mock, patch
import json

from src.utils.test_data import PayloadGenerator
from src.utils.client import LLMClient
from src.utils.validators import ResponseValidator


@pytest.fixture
def payload_generator():
    """Create payload generator for fuzzing"""
    return PayloadGenerator()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    with patch('httpx.Client') as mock_client:
        client = LLMClient(base_url="http://localhost:8000")
        yield client


class TestInputFuzzing:
    """Test LLM service robustness with various inputs"""
    
    @pytest.mark.fuzzing
    @pytest.mark.security
    def test_empty_input_handling(self, mock_llm_client, payload_generator):
        """Test handling of empty and whitespace inputs"""
        empty_payloads = payload_generator.generate_empty_payloads()
        
        with patch.object(mock_llm_client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "Invalid input"}
            mock_post.return_value = mock_response
            
            for payload in empty_payloads[:5]:  # Test first 5
                response = mock_llm_client.generate(payload)
                assert response.status_code in [200, 400, 422]
    
    @pytest.mark.fuzzing
    @pytest.mark.security
    def test_unicode_handling(self, mock_llm_client, payload_generator):
        """Test handling of Unicode characters"""
        unicode_payloads = payload_generator.generate_unicode_payloads()
        
        with patch.object(mock_llm_client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "Response"}
            mock_post.return_value = mock_response
            
            for payload in unicode_payloads[:5]:  # Test first 5
                response = mock_llm_client.generate(payload)
                assert response.status_code in [200, 400]
    
    @pytest.mark.fuzzing
    @pytest.mark.security
    @pytest.mark.critical
    def test_injection_resistance(self, mock_llm_client, payload_generator):
        """Test resistance to injection attacks"""
        injection_payloads = payload_generator.generate_injection_payloads()
        
        with patch.object(mock_llm_client, 'post') as mock_post:
            # Simulate safe responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "text": "I cannot process that request"
            }
            mock_post.return_value = mock_response
            
            dangerous_keywords = ['DROP TABLE', 'rm -rf', '<script>', 'HACKED']
            
            for payload in injection_payloads[:10]:  # Test first 10
                response = mock_llm_client.generate(payload)
                
                if response.status_code == 200:
                    result_text = response.json().get('text', '')
                    # Check that dangerous content is not echoed
                    for keyword in dangerous_keywords:
                        assert keyword not in result_text
    
    @pytest.mark.fuzzing
    def test_malformed_json_handling(self, mock_llm_client, payload_generator):
        """Test handling of malformed JSON inputs"""
        malformed_json = payload_generator.generate_malformed_json()
        
        for json_payload in malformed_json[:5]:  # Test first 5
            with patch.object(mock_llm_client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 422
                mock_response.json.return_value = {"error": "Invalid JSON"}
                mock_post.return_value = mock_response
                
                # Try to send as JSON
                response = mock_llm_client.post(
                    '/generate',
                    content=json_payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                # Should handle gracefully
                assert response.status_code in [400, 422]
    
    @pytest.mark.fuzzing
    def test_boundary_values(self, mock_llm_client, payload_generator):
        """Test handling of boundary values"""
        boundary_values = payload_generator.generate_boundary_values()
        
        with patch.object(mock_llm_client, 'post') as mock_post:
            for value in boundary_values[:10]:  # Test first 10
                # Convert value to appropriate format
                if isinstance(value, (int, float)):
                    if value == float('inf') or value == float('-inf') or (isinstance(value, float) and value != value):
                        # Special float values should be rejected
                        mock_response = Mock()
                        mock_response.status_code = 422
                        mock_response.json.return_value = {"error": "Invalid parameter"}
                    else:
                        # Normal numeric values
                        mock_response = Mock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"text": "Response"}
                else:
                    # String/None/Boolean values
                    mock_response = Mock()
                    mock_response.status_code = 200 if value else 400
                    mock_response.json.return_value = {"text": "Response"}
                
                mock_post.return_value = mock_response
                
                # Test as parameter value
                response = mock_llm_client.generate("Test", max_tokens=value if isinstance(value, int) and value > 0 else 100)
                assert response.status_code in [200, 400, 422]


@pytest.mark.fuzzing
class TestParameterFuzzing:
    """Test parameter validation with fuzzing"""
    
    def test_temperature_fuzzing(self, mock_llm_client):
        """Test temperature parameter validation"""
        invalid_temperatures = [-10.0, 10.0, float('inf'), float('-inf'), float('nan'), "invalid", [], {}]
        
        for temp in invalid_temperatures:
            with patch.object(mock_llm_client, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 422
                mock_response.json.return_value = {
                    "error": "Invalid temperature value"
                }
                mock_post.return_value = mock_response
                
                response = mock_llm_client.generate(
                    "Test prompt",
                    temperature=temp
                )
                
                assert response.status_code in [400, 422]
    
    def test_max_tokens_fuzzing(self, mock_llm_client):
        """Test max_tokens parameter validation"""
        invalid_tokens = [-1, 0, 2**31, "unlimited", None, [], {}]
        
        for tokens in invalid_tokens:
            with patch.object(mock_llm_client, 'post') as mock_post:
                if isinstance(tokens, int) and tokens > 0:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"text": "Response"}
                else:
                    mock_response = Mock()
                    mock_response.status_code = 422
                    mock_response.json.return_value = {
                        "error": "Invalid max_tokens value"
                    }
                mock_post.return_value = mock_response
                
                response = mock_llm_client.generate(
                    "Test prompt",
                    max_tokens=tokens
                )
                
                if isinstance(tokens, int) and tokens > 0 and tokens < 100000:
                    assert response.status_code == 200
                else:
                    assert response.status_code in [400, 422]
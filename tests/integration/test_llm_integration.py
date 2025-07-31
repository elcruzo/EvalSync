"""
Integration tests for LLM services
"""

import pytest
import httpx
from unittest.mock import Mock, patch
import json

from src.utils.client import LLMClient, APIClient
from src.utils.validators import ResponseValidator, SchemaValidator


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    with patch('httpx.Client') as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "This is a test response",
            "metadata": {
                "model": "test-model",
                "tokens_used": 50
            }
        }
        mock_client.return_value.request.return_value = mock_response
        yield mock_client


@pytest.fixture
def llm_client():
    """Create LLM client for testing"""
    return LLMClient(base_url="http://localhost:8000")


@pytest.fixture
def response_validator():
    """Create response validator"""
    return ResponseValidator()


class TestBasicInference:
    """Test basic LLM inference capabilities"""
    
    @pytest.mark.integration
    @pytest.mark.high_priority
    def test_simple_generation(self, llm_client, mock_llm_service):
        """Test simple text generation"""
        with patch.object(llm_client, 'client', mock_llm_service.return_value):
            response = llm_client.generate("Test prompt")
            
            assert response.status_code == 200
            result = response.json()
            assert "text" in result
            assert len(result["text"]) > 0
    
    @pytest.mark.integration
    def test_generation_with_parameters(self, llm_client, mock_llm_service):
        """Test generation with custom parameters"""
        with patch.object(llm_client, 'client', mock_llm_service.return_value):
            response = llm_client.generate(
                "Test prompt",
                temperature=0.5,
                max_tokens=100,
                top_p=0.9
            )
            
            assert response.status_code == 200
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_generation(self, llm_client):
        """Test async generation capability"""
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "Async response"}
            
            mock_async_client.return_value.request.return_value = mock_response
            
            # Patch the async client
            with patch.object(llm_client, 'async_client', mock_async_client.return_value):
                response = await llm_client.generate_async("Test async prompt")
                assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in LLM integration"""
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    def test_empty_prompt_handling(self, llm_client):
        """Test handling of empty prompts"""
        with patch.object(llm_client, 'post') as mock_post:
            # Simulate API accepting empty prompt
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "Empty prompt"}
            mock_post.return_value = mock_response
            
            response = llm_client.generate("")
            assert response.status_code in [200, 400, 422]
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    def test_invalid_parameters(self, llm_client):
        """Test handling of invalid parameters"""
        with patch.object(llm_client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 422
            mock_response.json.return_value = {
                "error": "Invalid parameter",
                "details": {"temperature": "must be between 0 and 1"}
            }
            mock_post.return_value = mock_response
            
            response = llm_client.generate(
                "Test prompt",
                temperature=2.0  # Invalid temperature
            )
            
            assert response.status_code in [400, 422]
            error_data = response.json()
            assert "error" in error_data
    
    @pytest.mark.integration
    @pytest.mark.error_handling
    def test_timeout_handling(self, llm_client):
        """Test timeout handling"""
        with patch.object(llm_client, 'post') as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")
            
            with pytest.raises(httpx.TimeoutException):
                llm_client.generate("Test prompt")


class TestResponseValidation:
    """Test response validation"""
    
    @pytest.mark.integration
    @pytest.mark.schema
    def test_response_schema_validation(self, response_validator):
        """Test schema validation of responses"""
        schema = {
            "type": "object",
            "required": ["text", "metadata"],
            "properties": {
                "text": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "required": ["model", "tokens_used"],
                    "properties": {
                        "model": {"type": "string"},
                        "tokens_used": {"type": "integer", "minimum": 0}
                    }
                }
            }
        }
        
        valid_response = Mock()
        valid_response.status_code = 200
        valid_response.json.return_value = {
            "text": "Test response",
            "metadata": {
                "model": "gpt-3.5",
                "tokens_used": 50
            }
        }
        
        is_valid = response_validator.validate_response(
            valid_response,
            expected_status=200,
            schema=schema
        )
        
        assert is_valid is True
    
    @pytest.mark.integration
    @pytest.mark.schema
    def test_invalid_response_detection(self, response_validator):
        """Test detection of invalid responses"""
        schema = {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"}
            }
        }
        
        invalid_response = Mock()
        invalid_response.status_code = 200
        invalid_response.json.return_value = {
            "wrong_field": "This is wrong"
        }
        
        is_valid = response_validator.validate_response(
            invalid_response,
            expected_status=200,
            schema=schema
        )
        
        assert is_valid is False
        errors = response_validator.get_errors()
        assert len(errors) > 0


class TestRAGIntegration:
    """Test RAG-specific functionality"""
    
    @pytest.mark.integration
    @pytest.mark.rag
    def test_context_injection(self, llm_client):
        """Test that context is properly used in generation"""
        context = "The capital of France is Paris."
        prompt = "What is the capital of France?"
        
        with patch.object(llm_client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "text": "Based on the context, the capital of France is Paris.",
                "metadata": {"used_context": True}
            }
            mock_post.return_value = mock_response
            
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            response = llm_client.generate(full_prompt)
            
            assert response.status_code == 200
            result = response.json()
            assert "paris" in result["text"].lower()


@pytest.mark.integration
@pytest.mark.smoke
class TestSmokeTests:
    """Quick smoke tests for basic functionality"""
    
    def test_client_initialization(self):
        """Test that clients can be initialized"""
        client = LLMClient(base_url="http://localhost:8000")
        assert client is not None
        assert client.base_url == "http://localhost:8000"
    
    def test_validator_initialization(self):
        """Test that validators can be initialized"""
        validator = ResponseValidator()
        assert validator is not None
        
        schema_validator = SchemaValidator()
        assert schema_validator is not None
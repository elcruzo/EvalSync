"""
HTTP Clients for API Testing
Specialized clients for LLM and API service testing
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
import httpx
import json
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class APIClient:
    """Base HTTP client for API testing"""
    
    def __init__(self, 
                 base_url: str,
                 timeout: int = 30,
                 auth: Optional[Dict[str, str]] = None,
                 headers: Optional[Dict[str, str]] = None):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            auth: Authentication configuration
            headers: Default headers to include
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.default_headers = headers or {}
        
        # Setup authentication
        if auth:
            self._setup_auth(auth)
        
        # Initialize httpx client
        self.client = httpx.Client(
            timeout=timeout,
            headers=self.default_headers
        )
        
        # Async client for concurrent requests
        self.async_client = httpx.AsyncClient(
            timeout=timeout,
            headers=self.default_headers
        )
        
        # Request tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
    
    def _setup_auth(self, auth: Dict[str, str]):
        """Setup authentication headers"""
        auth_type = auth.get('type', '').lower()
        
        if auth_type == 'bearer':
            self.default_headers['Authorization'] = f"Bearer {auth['token']}"
        elif auth_type == 'api_key':
            header_name = auth.get('header', 'X-API-Key')
            self.default_headers[header_name] = auth['key']
        elif auth_type == 'basic':
            import base64
            credentials = f"{auth['username']}:{auth['password']}"
            encoded = base64.b64encode(credentials.encode()).decode()
            self.default_headers['Authorization'] = f"Basic {encoded}"
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str,
                     headers: Optional[Dict] = None,
                     **kwargs) -> httpx.Response:
        """Make HTTP request with tracking"""
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        # Merge headers
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)
        
        start_time = time.time()
        
        try:
            response = self.client.request(
                method=method,
                url=url,
                headers=request_headers,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            # Track metrics
            self.request_count += 1
            self.total_response_time += response_time
            
            logger.debug(f"{method} {url} -> {response.status_code} ({response_time:.3f}s)")
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"{method} {url} failed: {e}")
            raise e
    
    async def _make_async_request(self,
                                 method: str,
                                 endpoint: str,
                                 headers: Optional[Dict] = None,
                                 **kwargs) -> httpx.Response:
        """Make async HTTP request"""
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)
        
        start_time = time.time()
        
        try:
            response = await self.async_client.request(
                method=method,
                url=url,
                headers=request_headers,
                **kwargs
            )
            
            response_time = time.time() - start_time
            self.request_count += 1
            self.total_response_time += response_time
            
            logger.debug(f"ASYNC {method} {url} -> {response.status_code} ({response_time:.3f}s)")
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"ASYNC {method} {url} failed: {e}")
            raise e
    
    def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request"""
        return self._make_request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request"""
        return self._make_request('POST', endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PUT request"""
        return self._make_request('PUT', endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make DELETE request"""
        return self._make_request('DELETE', endpoint, **kwargs)
    
    async def get_async(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make async GET request"""
        return await self._make_async_request('GET', endpoint, **kwargs)
    
    async def post_async(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make async POST request"""
        return await self._make_async_request('POST', endpoint, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics"""
        avg_response_time = (self.total_response_time / self.request_count 
                           if self.request_count > 0 else 0)
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0,
            'average_response_time_ms': avg_response_time * 1000,
            'total_response_time_ms': self.total_response_time * 1000
        }
    
    def close(self):
        """Close HTTP clients"""
        self.client.close()
        asyncio.create_task(self.async_client.aclose())


class LLMClient(APIClient):
    """Specialized client for LLM API testing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Common LLM endpoints
        self.endpoints = {
            'generate': '/generate',
            'chat': '/chat', 
            'complete': '/complete',
            'query': '/query',
            'health': '/health',
            'metrics': '/metrics'
        }
    
    def generate(self, 
                prompt: str,
                temperature: float = 0.7,
                max_tokens: int = 100,
                top_p: float = 0.9,
                top_k: Optional[int] = None,
                stop: Optional[List[str]] = None,
                **kwargs) -> httpx.Response:
        """Generate text using the LLM"""
        # Try different common request formats
        request_bodies = [
            # OpenAI-style format
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stop": stop,
                **kwargs
            },
            # Anthropic-style format
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            },
            # HuggingFace-style format
            {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    "do_sample": True,
                    **kwargs
                }
            },
            # Custom format
            {
                "query": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **kwargs
            }
        ]
        
        # Try different endpoints
        endpoints_to_try = ['/generate', '/chat', '/complete', '/query', '/v1/completions', '/api/v1/generate']
        
        last_response = None
        for endpoint in endpoints_to_try:
            for request_body in request_bodies:
                try:
                    response = self.post(
                        endpoint,
                        json=request_body,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code in [200, 201]:
                        return response
                    
                    last_response = response
                    
                except Exception as e:
                    logger.debug(f"Failed to call {endpoint} with format: {e}")
                    continue
        
        # Return the last response if none succeeded
        if last_response:
            return last_response
        
        # If all failed, try basic endpoint
        return self.post('/generate', json={"prompt": prompt})
    
    async def generate_async(self, prompt: str, **kwargs) -> httpx.Response:
        """Async version of generate"""
        request_body = {
            "prompt": prompt,
            **kwargs
        }
        
        return await self.post_async(
            '/generate',
            json=request_body,
            headers={'Content-Type': 'application/json'}
        )
    
    def chat(self, 
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 100,
             **kwargs) -> httpx.Response:
        """Chat completion request"""
        request_body = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        return self.post(
            '/chat',
            json=request_body,
            headers={'Content-Type': 'application/json'}
        )
    
    def retrieve(self, 
                query: str, 
                top_k: int = 5,
                **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents (for RAG systems)"""
        request_body = {
            "query": query,
            "top_k": top_k,
            **kwargs
        }
        
        try:
            response = self.post(
                '/retrieve',
                json=request_body,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    return result
                elif 'documents' in result:
                    return result['documents']
                elif 'results' in result:
                    return result['results']
            
            return []
            
        except Exception as e:
            logger.warning(f"Retrieve failed: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            response = self.get('/health')
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time_ms': self.get_metrics()['average_response_time_ms'],
                    'details': response.json() if response.content else {}
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'response': response.text[:200]
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def stream_generate(self, prompt: str, **kwargs):
        """Generate streaming response (if supported)"""
        request_body = {
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        try:
            with self.client.stream(
                'POST',
                urljoin(self.base_url, '/generate'),
                json=request_body,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                for line in response.iter_lines():
                    if line:
                        try:
                            # Parse SSE format
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data == '[DONE]':
                                    break
                                yield json.loads(data)
                            else:
                                yield {"raw": line}
                        except json.JSONDecodeError:
                            yield {"raw": line}
                            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield {"error": str(e)}


class MultiServiceClient:
    """Client for testing multiple services simultaneously"""
    
    def __init__(self, service_configs: Dict[str, Dict[str, Any]]):
        """Initialize clients for multiple services"""
        self.clients = {}
        
        for service_name, config in service_configs.items():
            self.clients[service_name] = LLMClient(
                base_url=config['base_url'],
                timeout=config.get('timeout', 30),
                auth=config.get('auth')
            )
    
    def broadcast_request(self, 
                         method: str,
                         **kwargs) -> Dict[str, httpx.Response]:
        """Send the same request to all services"""
        results = {}
        
        for service_name, client in self.clients.items():
            try:
                method_func = getattr(client, method)
                response = method_func(**kwargs)
                results[service_name] = response
                
            except Exception as e:
                logger.error(f"Request to {service_name} failed: {e}")
                results[service_name] = e
        
        return results
    
    async def broadcast_request_async(self, 
                                    method: str,
                                    **kwargs) -> Dict[str, httpx.Response]:
        """Send the same request to all services asynchronously"""
        async def make_request(service_name, client):
            try:
                method_func = getattr(client, f"{method}_async", None)
                if method_func:
                    return await method_func(**kwargs)
                else:
                    # Fallback to sync method
                    return getattr(client, method)(**kwargs)
            except Exception as e:
                logger.error(f"Async request to {service_name} failed: {e}")
                return e
        
        tasks = [
            make_request(service_name, client)
            for service_name, client in self.clients.items()
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        return dict(zip(self.clients.keys(), responses))
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all clients"""
        return {
            service_name: client.get_metrics()
            for service_name, client in self.clients.items()
        }
    
    def close_all(self):
        """Close all client connections"""
        for client in self.clients.values():
            client.close()
"""
Validators for API Response and Schema Validation
"""

import json
import jsonschema
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Validates API responses against expected patterns"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_errors = []
    
    def validate_response(self, 
                         response: Any, 
                         expected_status: Optional[int] = None,
                         expected_fields: Optional[List[str]] = None,
                         schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate an API response
        
        Args:
            response: Response object to validate
            expected_status: Expected HTTP status code
            expected_fields: List of fields that must be present
            schema: JSON schema to validate against
            
        Returns:
            True if valid, False otherwise
        """
        self.validation_errors.clear()
        is_valid = True
        
        # Validate status code
        if expected_status and hasattr(response, 'status_code'):
            if response.status_code != expected_status:
                self.validation_errors.append(
                    f"Expected status {expected_status}, got {response.status_code}"
                )
                is_valid = False
        
        # Parse response data
        try:
            if hasattr(response, 'json'):
                data = response.json()
            else:
                data = response
        except (json.JSONDecodeError, ValueError) as e:
            self.validation_errors.append(f"Invalid JSON response: {e}")
            return False
        
        # Validate required fields
        if expected_fields:
            missing_fields = []
            for field in expected_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                self.validation_errors.append(
                    f"Missing required fields: {missing_fields}"
                )
                is_valid = False
        
        # Validate against schema
        if schema:
            try:
                jsonschema.validate(data, schema)
            except jsonschema.ValidationError as e:
                self.validation_errors.append(f"Schema validation failed: {e.message}")
                is_valid = False
        
        return is_valid
    
    def validate_text_response(self, 
                              response_text: str,
                              min_length: Optional[int] = None,
                              max_length: Optional[int] = None,
                              contains: Optional[List[str]] = None,
                              not_contains: Optional[List[str]] = None) -> bool:
        """
        Validate text content in responses
        
        Args:
            response_text: Text to validate
            min_length: Minimum text length
            max_length: Maximum text length
            contains: Strings that must be present
            not_contains: Strings that must not be present
            
        Returns:
            True if valid, False otherwise
        """
        self.validation_errors.clear()
        is_valid = True
        
        # Length validation
        if min_length and len(response_text) < min_length:
            self.validation_errors.append(
                f"Response too short: {len(response_text)} < {min_length}"
            )
            is_valid = False
        
        if max_length and len(response_text) > max_length:
            self.validation_errors.append(
                f"Response too long: {len(response_text)} > {max_length}"
            )
            is_valid = False
        
        # Content validation
        if contains:
            for required_text in contains:
                if required_text not in response_text:
                    self.validation_errors.append(
                        f"Missing required text: '{required_text}'"
                    )
                    is_valid = False
        
        if not_contains:
            for forbidden_text in not_contains:
                if forbidden_text in response_text:
                    self.validation_errors.append(
                        f"Contains forbidden text: '{forbidden_text}'"
                    )
                    is_valid = False
        
        return is_valid
    
    def get_errors(self) -> List[str]:
        """Get list of validation errors from last validation"""
        return self.validation_errors.copy()


class SchemaValidator:
    """Validates data against JSON schemas with enhanced features"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_results = []
    
    def validate(self, 
                data: Union[Dict, List, str], 
                schema: Dict[str, Any],
                raise_on_error: bool = False) -> bool:
        """
        Validate data against a JSON schema
        
        Args:
            data: Data to validate (dict, list, or JSON string)
            schema: JSON schema to validate against
            raise_on_error: Whether to raise exception on validation error
            
        Returns:
            True if valid, False otherwise
        """
        # Parse JSON string if necessary
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
                if raise_on_error:
                    raise ValueError(error_msg)
                logger.error(error_msg)
                return False
        
        # Perform validation
        try:
            jsonschema.validate(data, schema)
            return True
        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            if raise_on_error:
                raise e
            logger.warning(error_msg)
            self.validation_results.append({
                'valid': False,
                'error': error_msg,
                'path': list(e.path)
            })
            return False
    
    def validate_multiple(self, 
                         data_items: List[Any],
                         schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate multiple data items against the same schema
        
        Args:
            data_items: List of data items to validate
            schema: JSON schema to validate against
            
        Returns:
            Dictionary with validation statistics
        """
        results = {
            'total': len(data_items),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        for i, item in enumerate(data_items):
            if self.validate(item, schema):
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'index': i,
                    'errors': self.validation_results[-1] if self.validation_results else None
                })
        
        return results
    
    def compare_schemas(self, 
                       schema1: Dict[str, Any],
                       schema2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two schemas for compatibility
        
        Args:
            schema1: First schema
            schema2: Second schema
            
        Returns:
            Dictionary with comparison results
        """
        differences = {
            'identical': schema1 == schema2,
            'property_differences': [],
            'type_differences': [],
            'required_differences': []
        }
        
        # Compare properties
        props1 = schema1.get('properties', {})
        props2 = schema2.get('properties', {})
        
        # Find property differences
        all_props = set(props1.keys()) | set(props2.keys())
        for prop in all_props:
            if prop not in props1:
                differences['property_differences'].append(f"Missing in schema1: {prop}")
            elif prop not in props2:
                differences['property_differences'].append(f"Missing in schema2: {prop}")
            elif props1[prop] != props2[prop]:
                differences['type_differences'].append(f"Different definition for: {prop}")
        
        # Compare required fields
        req1 = set(schema1.get('required', []))
        req2 = set(schema2.get('required', []))
        
        if req1 != req2:
            differences['required_differences'] = {
                'only_in_schema1': list(req1 - req2),
                'only_in_schema2': list(req2 - req1)
            }
        
        return differences
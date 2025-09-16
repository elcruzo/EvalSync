# Changelog

All notable changes to EvalSync will be documented in this file.

## [1.0.1] - 2025-09-16

### Fixed
- Fixed import issues with validators and test_data modules
- Improved timeout handling in HTTP clients to prevent hanging tests
- Added connection limits to prevent resource exhaustion
- Added proper error handling for missing dependencies

### Changed
- Enhanced client initialization with better timeout configuration
- Updated httpx client to use structured timeout settings
- Added fallback imports for better backwards compatibility

### Security
- Added connection limits to prevent DoS scenarios
- Improved error handling to prevent information leakage

## [1.0.0] - 2025-07-31

### Added
- Initial implementation of comprehensive testing framework
- Core test framework with base suite architecture
- Integration and fuzzing test suites
- Validators and test data generators
- Multi-service client for parallel testing
- Pytest configuration with comprehensive markers
- Automated test runner scripts

## [0.1.0] - 2023-11-08

### Added
- Initial project setup
- Basic repository structure
- README with project overview
- Requirements file with dependencies
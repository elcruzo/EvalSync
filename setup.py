"""
Setup configuration for EvalSync
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="evalsync",
    version="1.0.0",
    author="EvalSync Team",
    author_email="team@evalsync.dev",
    description="Comprehensive automated test framework for LLM integration testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evalsync/evalsync",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.1",
        "httpx>=0.25.2",
        "jsonschema>=4.20.0",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
        "faker>=20.1.0",
        "hypothesis>=6.88.1",
    ],
    extras_require={
        "dev": [
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.5.0",
        ],
        "all": [
            "locust>=2.17.0",
            "allure-pytest>=2.13.2",
            "pytest-html>=4.1.1",
            "pytest-benchmark>=4.0.0",
            "memory-profiler>=0.61.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evalsync=src.cli:main",
            "evalsync-runner=src.framework.test_runner:main",
        ],
    },
    include_package_data=True,
    package_data={
        "evalsync": [
            "config/*.yaml",
            "config/*.json",
            "test_data/*",
        ],
    },
    zip_safe=False,
)
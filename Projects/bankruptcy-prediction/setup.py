"""
============================================================
Project Packaging Configuration
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This setup script allows the project to be installed as a
Python package.

Benefits
--------
• Enables `pip install -e .`
• Makes project modules importable globally
• Simplifies CI/CD pipelines
• Supports Docker deployments
"""

from setuptools import setup, find_packages


setup(
    name="bankruptcy-prevention-mlops",
    version="0.1.0",

    author="Aruri Gowtham",
    author_email="arurigowthamraj@gmail.com",

    description="End-to-end MLOps pipeline for bankruptcy risk prediction",
    long_description="Production-ready machine learning pipeline with MLflow tracking and Streamlit deployment",
    long_description_content_type="text/plain",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.9",

    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "streamlit",
        "plotly",
        "pyyaml",
        "joblib"
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    include_package_data=True,
)
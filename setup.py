from setuptools import find_namespace_packages, setup


setup(
    name="embedding-finetuning",
    version="0.1.0",
    description="Embedding fine-tuning project",
    packages=find_namespace_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "sentence-transformers>=2.3.0",
        "datasets>=2.14.0",
        "beir>=1.0.1",
        "pyyaml>=6.0",
        "faiss-cpu>=1.7.4",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.7.0",
    ],
)

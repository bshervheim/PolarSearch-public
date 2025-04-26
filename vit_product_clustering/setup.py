from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vit-product-clustering",
    version="0.1.0",
    author="Bradley Shervheim",
    author_email="brad@polarsearch.io",
    description="Vision Transformer for Semantic Product Image Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polarSearch/polarsearch-vit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.29.0",
        "Pillow>=9.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.1.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
)

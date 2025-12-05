"""
Elite Set-Piece Analytics - Setup Configuration
================================================

A comprehensive sports analytics platform for set-piece analysis,
first receiver prediction, and tactical intelligence.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.2.0",
    "xgboost>=1.7.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
    "kloppy>=3.8.0",
    "statsbombpy>=1.9.0",
    "mplsoccer>=1.1.0",
    "streamlit>=1.25.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
]

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
    ],
    "deep_learning": [
        "tensorflow>=2.12.0",
        "torch>=2.0.0",
    ],
}

# All extras combined
EXTRAS_REQUIRE["all"] = list(
    set(dep for deps in EXTRAS_REQUIRE.values() for dep in deps)
)

setup(
    name="elite-set-piece-analytics",
    version="0.1.0",
    author="Elite Set-Piece Analytics Team",
    author_email="team@elite-setpiece.com",
    description="A comprehensive sports analytics platform for set-piece analysis and tactical intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics",
    project_urls={
        "Bug Tracker": "https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics/issues",
        "Documentation": "https://github.com/filopatirbakhoum06-collab/elite-set-piece-analytics/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "setpiece-train=scripts.train_models:main",
            "setpiece-dashboard=dashboard.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

from setuptools import setup, find_packages

setup(
    name="stock_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tensorflow",
        "yfinance",
        "transformers",
        "gradio",
        "dvc",
        "mlflow",
        "pytest",
    ],
    author="Abhinav Pundir",
    author_email="abhinav.pundir@ucdenver.edu",
    description="Hybrid stock price prediction using LSTM and LLM",
    keywords="stock, prediction, machine learning, deep learning, lstm, llm",
    url="https://github.com/abhinavvpundirr/stock-prediction-hybrid-model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)


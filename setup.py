from setuptools import setup, find_packages

setup(
    name="hotel_reservation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "google-cloud-storage",
        "scikit-learn",
        "pyyaml",
        "imbalanced-learn",
        "lightgbm",
        "mlflow",
        "flask"
    ],
    author="Sudip Maji",
    author_email="xxx",
    description="Hotel Reservation Prediction MLOps Project on GCP",
    keywords="mlops,hotel,reservation,prediction,gcp",
    python_requires=">=3.9",
)

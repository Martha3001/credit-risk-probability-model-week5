# Credit Risk Probability Model

## Overview
This project develops a **Credit Scoring Model** for Bati Bank to assess customer creditworthiness in a buy-now-pay-later (BNPL) service. The model predicts default risk using transactional data and RFM (Recency, Frequency, Monetary) metrics.

## Credit Scoring Business Understanding

### Basel II and Model Interpretability
The Basel II Accord emphasizes rigorous risk measurement and capital adequacy, requiring financial institutions to justify their credit risk models to regulators. This drives the need for **interpretable and well-documented models** that clearly explain how predictions are made, ensuring transparency, auditability, and regulatory compliance.

### Proxy Variables and Business Risk
In the absence of a direct "default" label, we must construct a **proxy variable**—often based on delinquency thresholds or payment behavior—to approximate default. While necessary for model training, this introduces risks: the proxy may not perfectly reflect true default behavior, potentially leading to **misclassification, biased risk estimates**, and flawed business decisions that affect credit approval and portfolio management.

### Trade-offs: Simplicity vs. Performance
- **Simple models** (e.g., Logistic Regression with Weight of Evidence) offer high interpretability, ease of documentation, and regulatory friendliness.
- **Complex models** (e.g., Gradient Boosting) often deliver superior predictive performance but are harder to explain and validate.

In regulated environments, the trade-off centers on balancing **predictive power with transparency**. Institutions must weigh the benefits of performance against the costs of reduced interpretability, especially when facing audits or needing to justify decisions to stakeholders.

## Key Features
1. **Data Engineering**
   - RFM-based proxy target (`is_high_risk`).
   - Automated feature pipelines.
2. **Modeling**
   - Random Forest (99.7% accuracy, ROC-AUC 0.999).
   - Hyperparameter tuning via `GridSearchCV`.
3. **MLOps**
   - Model versioning with MLflow.
   - FastAPI endpoint for real-time predictions.
   - CI/CD with GitHub Actions (linting, unit tests).

## Project Structure

credit-risk-probability-model-week5/
├── .github/workflows/ # CI/CD (linting, testing)
│ └── ci.yml
├── data/ # Raw/processed data (ignored in Git)
├── notebooks/ # EDA and prototyping
│ ├── eda.ipynb
│ ├── model.ipynb
│ ├── processing.ipynb
│ └── response.ipynb
├── src/ # Production scripts
│ ├── data_processing.py # Feature engineering pipeline
│ ├── train.py # Model training and tuning
│ └── api/ # FastAPI deployment
│      ├── main.py
│      └── pydantic_models.py
├── tests/ # Unit tests
│ └── test_data_processing.py
├── Dockerfile # Containerization
├── docker-compose.yml
├── requirements.txt # Dependencies
└── README.md

## Setup
1. Clone: `git clone https://github.com/Martha3001/credit-risk-probability-model-week5.git`
2. Create venv: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

## Usage
Run the FastAPI server locally: `uvicorn src.api.main:app --reload`

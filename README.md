## Credit Scoring Business Understanding

### Basel II and Model Interpretability
The Basel II Accord emphasizes rigorous risk measurement and capital adequacy, requiring financial institutions to justify their credit risk models to regulators. This drives the need for **interpretable and well-documented models** that clearly explain how predictions are made, ensuring transparency, auditability, and regulatory compliance.

### Proxy Variables and Business Risk
In the absence of a direct "default" label, we must construct a **proxy variable**—often based on delinquency thresholds or payment behavior—to approximate default. While necessary for model training, this introduces risks: the proxy may not perfectly reflect true default behavior, potentially leading to **misclassification, biased risk estimates**, and flawed business decisions that affect credit approval and portfolio management.

### Trade-offs: Simplicity vs. Performance
- **Simple models** (e.g., Logistic Regression with Weight of Evidence) offer high interpretability, ease of documentation, and regulatory friendliness.
- **Complex models** (e.g., Gradient Boosting) often deliver superior predictive performance but are harder to explain and validate.

In regulated environments, the trade-off centers on balancing **predictive power with transparency**. Institutions must weigh the benefits of performance against the costs of reduced interpretability, especially when facing audits or needing to justify decisions to stakeholders.

## Setup
1. Clone: `git clone https://github.com/Martha3001/credit-risk-probability-model-week5.git`
2. Create venv: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

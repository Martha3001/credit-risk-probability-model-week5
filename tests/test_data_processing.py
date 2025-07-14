import pandas as pd
import numpy as np
from src.data_processing import (
    AggregateFeatures, ExtractDateFeatures, CategoricalEncoder, 
    NumericalScaler, assign_high_risk_label, calculate_rfm
)

def test_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300]
    })
    agg = AggregateFeatures().fit_transform(df)
    assert 'total_transaction_amount' in agg.columns
    assert agg.loc[agg['CustomerId'] == 1, 'total_transaction_amount'].iloc[0] == 300
    assert agg.loc[agg['CustomerId'] == 2, 'total_transaction_amount'].iloc[0] == 300

def test_extract_date_features():
    df = pd.DataFrame({'TransactionStartTime': ['2024-01-01 10:00:00']})
    features = ExtractDateFeatures().fit_transform(df.copy())
    assert 'transaction_hour' in features.columns
    assert features['transaction_hour'].iloc[0] == 10
    assert 'transaction_day' in features.columns
    assert 'transaction_month' in features.columns
    assert 'transaction_year' in features.columns

def test_categorical_encoder():
    df = pd.DataFrame({'A': ['x', 'y'], 'B': ['cat', 'dog']})
    enc = CategoricalEncoder(onehot_columns=['A'], label_columns=['B'])
    enc.fit(df)
    transformed = enc.transform(df)
    assert any(col.startswith('A_') for col in transformed.columns)
    assert 'B_label' in transformed.columns

def test_numerical_scaler():
    df = pd.DataFrame({'num': [1, 2, 3]})
    scaler = NumericalScaler(method='standard', columns=['num'])
    scaler.fit(df)
    scaled = scaler.transform(df.copy())
    np.testing.assert_almost_equal(scaled['num'].mean(), 0, decimal=6)

def test_calculate_rfm():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'TransactionId': [1, 2, 3],
        'Amount': [100, 200, 300]
    })
    rfm = calculate_rfm(df)
    assert set(['CustomerId', 'Recency', 'Frequency', 'Monetary']).issubset(rfm.columns)

def test_assign_high_risk_label():
    rfm = pd.DataFrame({
        'CustomerId': [1, 2, 3],
        'Recency': [1, 2, 3],
        'Frequency': [10, 5, 1],
        'Monetary': [1000, 500, 100]
    })
    rfm = assign_high_risk_label(rfm, n_clusters=2)
    assert 'is_high_risk' in rfm.columns
    assert set(rfm['is_high_risk']).issubset({0, 1})
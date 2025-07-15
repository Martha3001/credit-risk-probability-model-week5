import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, LabelEncoder,
                                   MinMaxScaler, StandardScaler)
from sklearn.cluster import KMeans


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Aggregate features per customer: sum, mean, count,
    std of transaction amounts.
    Assumes 'customer_id' and 'amount' columns exist.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby('CustomerId')['Amount'].agg([
            ('total_transaction_amount', 'sum'),
            ('average_transaction_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_transaction_amount', 'std')
        ]).reset_index()
        # Fill missing std_transaction_amount with 0
        agg['std_transaction_amount'] = agg['std_transaction_amount'].fillna(0)
        # Merge aggregated features back to the original DataFrame
        X_merged = X.merge(agg, on='CustomerId', how='left')
        return X_merged


class ExtractDateFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, year from a datetime column
    'transaction_date'.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year
        X = X.drop(columns=['TransactionStartTime'])
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables using One-Hot or Label Encoding.
    By default, uses One-Hot for nominal and Label for ordinal.
    """
    def __init__(self, onehot_columns=None, label_columns=None):
        self.onehot_columns = onehot_columns
        self.label_columns = label_columns
        self.encoders = {}

    def fit(self, X, y=None):
        if self.onehot_columns:
            self.encoders['onehot'] = OneHotEncoder(sparse_output=False,
                                                    handle_unknown='ignore')
            self.encoders['onehot'].fit(X[self.onehot_columns])
        if self.label_columns:
            self.encoders['label'] = {}
            for col in self.label_columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders['label'][col] = le
        return self

    def transform(self, X):
        # One-Hot Encoding
        if self.onehot_columns:
            onehot = self.encoders['onehot'].transform(X[self.onehot_columns])
            onehot_df = pd.DataFrame(onehot, columns=self.encoders['onehot']
                                     .get_feature_names_out(
                                        self.onehot_columns),
                                     index=X.index)
            X = pd.concat([X.drop(columns=self.onehot_columns), onehot_df],
                          axis=1)
        # Label Encoding
        if self.label_columns:
            for col in self.label_columns:
                le = self.encoders['label'][col]
                X[col + '_label'] = le.transform(X[col].astype(str))
                X = X.drop(columns=[col])
        return X


class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Scales numerical features using normalization (MinMax) or
    standardization (StandardScaler).
    """
    def __init__(self, method='standard', columns=None):
        self.method = method
        self.columns = columns
        self.scaler = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(
                include=[np.number]).columns.tolist()
        if self.method == 'normalize':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X


def build_feature_pipeline():
    """
    Returns a pipeline that extracts date features and
    aggregates per customer.
    """
    onehot_columns = ['ProviderId', 'ChannelId', 'ProductCategory']
    label_columns = ['ProductId']  # too many unique values
    scaling_method = 'standard'
    numerical_features = [
        'Amount', 'Value',
        'total_transaction_amount', 'average_transaction_amount',
        'transaction_count', 'std_transaction_amount'
    ]
    return Pipeline([
        ('extract_date_features', ExtractDateFeatures()),
        ('aggregate_features', AggregateFeatures()),
        ('categorical_encoding', CategoricalEncoder(
            onehot_columns=onehot_columns, label_columns=label_columns
            )),
        ('numerical_scaler', NumericalScaler(
            method=scaling_method, columns=numerical_features
            ))
    ])


def process_features(df):
    """
    Main entry point: applies the feature pipeline to the dataframe.
    """
    pipeline = build_feature_pipeline()
    features = pipeline.fit_transform(df)
    return features


def assign_high_risk_label(rfm, n_clusters=3, random_state=42):
    """
    Scale RFM features, perform KMeans clustering, and assign high-risk label.
    Returns rfm DataFrame with Cluster and is_high_risk columns.
    """
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    cluster_stats = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats.sort_values(
        ['Frequency', 'Monetary']).index[0]
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm


def calculate_rfm(df, snapshot_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each CustomerId.
    snapshot_date: datetime to calculate recency from.
    If None, use max(TransactionStartTime)+1 day.
    Returns a DataFrame with columns: CustomerId, Recency, Frequency, Monetary
    """
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(
            df['TransactionStartTime']).max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date
                                           - pd.to_datetime(x).max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

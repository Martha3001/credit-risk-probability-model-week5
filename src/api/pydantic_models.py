from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    Amount: float
    Value: float
    PricingStrategy: int
    FraudResult: int
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    total_transaction_amount: float
    average_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float
    ProviderId_1: float
    ProviderId_2: float
    ProviderId_3: float
    ProviderId_4: float
    ProviderId_5: float
    ProviderId_6: float
    ChannelId_1: float
    ChannelId_2: float
    ChannelId_3: float
    ChannelId_5: float
    ProductCategory_airtime: float
    ProductCategory_data_bundles: float
    ProductCategory_financial_services: float
    ProductCategory_movies: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_transport: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ProductId_label: int


class PredictionResponse(BaseModel):
    risk_probability: float

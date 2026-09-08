"""
Script to generate realistic benchmark datasets for Autonomous Data Scientist:
1. data/churn.csv (Binary Classification, ~1000 rows, mixed types, realistic churn drivers)
2. data/housing.csv (Regression, ~1000 rows, continuous target, spatial & property attributes)
"""
import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# --- 1. Classification: Customer Churn ---
n_churn = 1000
age = np.random.randint(18, 75, size=n_churn)
tenure_months = np.random.randint(1, 72, size=n_churn)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_churn, p=[0.55, 0.25, 0.20])
internet = np.random.choice(["DSL", "Fiber optic", "No"], size=n_churn, p=[0.35, 0.45, 0.20])
tech_support = np.random.choice(["Yes", "No", "No internet service"], size=n_churn, p=[0.30, 0.50, 0.20])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=n_churn,
    p=[0.34, 0.22, 0.22, 0.22]
)
monthly_charges = np.where(
    internet == "Fiber optic",
    np.random.uniform(70, 115, size=n_churn),
    np.where(internet == "DSL", np.random.uniform(40, 70, size=n_churn), np.random.uniform(18, 30, size=n_churn))
).round(2)
total_charges = (monthly_charges * tenure_months + np.random.normal(0, 50, size=n_churn)).clip(18).round(2)

# Churn probability logit
logit = (
    -1.5
    + (0.015 * (monthly_charges - 60))
    - (0.04 * tenure_months)
    + (0.8 * (contract == "Month-to-month"))
    - (0.7 * (contract == "Two year"))
    + (0.6 * (internet == "Fiber optic"))
    - (0.5 * (tech_support == "Yes"))
    + (0.4 * (payment_method == "Electronic check"))
)
prob = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(n_churn) < prob).astype(int)

df_churn = pd.DataFrame({
    "customer_id": [f"CUST-{i:04d}" for i in range(1, n_churn + 1)],
    "age": age,
    "tenure_months": tenure_months,
    "contract_type": contract,
    "internet_service": internet,
    "tech_support": tech_support,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn": churn
})

# Introduce realistic nulls and a duplicate or two for data quality testing
df_churn.loc[np.random.choice(n_churn, 25, replace=False), "total_charges"] = np.nan
df_churn.loc[np.random.choice(n_churn, 15, replace=False), "tech_support"] = np.nan
df_churn.to_csv("data/churn.csv", index=False)
print(f"Generated data/churn.csv: {df_churn.shape}, churn rate: {churn.mean():.2%}")

# --- 2. Regression: Housing Prices ---
n_house = 1000
median_income = np.random.gamma(shape=3.0, scale=1.5, size=n_house).round(2)
house_age = np.random.randint(1, 52, size=n_house)
total_rooms = np.random.randint(200, 5000, size=n_house)
total_bedrooms = (total_rooms * np.random.uniform(0.18, 0.25, size=n_house)).astype(int)
population = (total_rooms * np.random.uniform(0.4, 0.8, size=n_house)).astype(int)
ocean_proximity = np.random.choice(["NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN", "ISLAND"], size=n_house, p=[0.20, 0.35, 0.30, 0.14, 0.01])

# House value function
value = (
    50000
    + (median_income * 35000)
    + (house_age * 500)
    + (total_rooms * 10)
    + np.where(ocean_proximity == "INLAND", -35000, 45000)
    + np.random.normal(0, 15000, size=n_house)
).clip(40000, 500000).round(-2)

df_house = pd.DataFrame({
    "median_income": median_income,
    "house_age": house_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "ocean_proximity": ocean_proximity,
    "median_house_value": value
})
# Introduce slight missingness
df_house.loc[np.random.choice(n_house, 18, replace=False), "total_bedrooms"] = np.nan
df_house.to_csv("data/housing.csv", index=False)
print(f"Generated data/housing.csv: {df_house.shape}")

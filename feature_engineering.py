import pandas as pd
import numpy as np

df = pd.read_csv("retail_price.csv")

#1. Fix date column
df['month_year'] = pd.to_datetime(df['month_year'], format='%d-%m-%Y')
df = df.sort_values(['product_id', 'month_year']).reset_index(drop=True)

#2. Competitor features
df['avg_comp_price'] = df[['comp_1', 'comp_2', 'comp_3']].mean(axis=1)
df['price_vs_comp'] = df['unit_price'] - df['avg_comp_price']
df['comp_price_ratio'] = df['unit_price'] / df['avg_comp_price']

#3. Lag & rolling demand features (per product)
df['demand_lag1'] = df.groupby('product_id')['qty'].shift(1)
df['demand_lag2'] = df.groupby('product_id')['qty'].shift(2)
df['demand_rolling3'] = df.groupby('product_id')['qty'].transform(
    lambda x: x.shift(1).rolling(3).mean()
)

#4. Price change features
df['price_change'] = df['unit_price'] - df['lag_price']
df['price_pct_change'] = df['price_change'] / df['lag_price']

#5. Drop rows where lag features are NaN
df = df.dropna(subset=['demand_lag1', 'demand_lag2', 'demand_rolling3'])

print("Shape after feature engineering:", df.shape)
print("\nNew features added:")
print(df[['product_id', 'unit_price', 'qty', 'demand_lag1', 
          'demand_rolling3', 'price_vs_comp', 'comp_price_ratio',
          'price_change']].head(10))

#Save for next phase
df.to_csv("retail_price_featured.csv", index=False)
print("\nSaved to retail_price_featured.csv")
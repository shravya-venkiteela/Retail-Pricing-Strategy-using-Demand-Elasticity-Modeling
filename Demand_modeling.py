import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings('ignore')

df = pd.read_csv("retail_price_featured.csv")

#Features to use
features = [
    'unit_price', 'freight_price', 'product_score',
    'comp_price_ratio', 'price_vs_comp', 'price_change', 'price_pct_change',
    'demand_lag1', 'demand_lag2', 'demand_rolling3',
    'month', 'weekend', 'holiday',
    'product_weight_g', 'volume'
]

#Encode category
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['product_category_name'])
features.append('category_encoded')

X = df[features]
y = df['qty']

#Model 1: Linear Regression (baseline)
lr = LinearRegression()
lr_scores = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Linear Regression MAE: {-lr_scores.mean():.2f} (+/- {lr_scores.std():.2f})")

#Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Random Forest MAE:     {-rf_scores.mean():.2f} (+/- {rf_scores.std():.2f})")

#Model 3: XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_scores = cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"XGBoost MAE:           {-xgb_scores.mean():.2f} (+/- {xgb_scores.std():.2f})")

#Feature Importance from best model
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head())
rf.fit(X, y)
joblib.dump(rf, 'demand_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
df.to_csv("retail_price_featured.csv", index=False)
print("\nModel saved to demand_model.pkl")
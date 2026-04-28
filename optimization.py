import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("retail_price_featured.csv")
model = joblib.load("demand_model.pkl")
le = joblib.load("label_encoder.pkl")

features = [
    'unit_price', 'freight_price', 'product_score',
    'comp_price_ratio', 'price_vs_comp', 'price_change', 'price_pct_change',
    'demand_lag1', 'demand_lag2', 'demand_rolling3',
    'month', 'weekend', 'holiday',
    'product_weight_g', 'volume', 'category_encoded'
]

results = []

for product_id in df['product_id'].unique():
    prod_df = df[df['product_id'] == product_id].copy()
    latest = prod_df.sort_values('month_year').iloc[-1].copy()
    
    current_price = latest['unit_price']
    avg_comp = latest['avg_comp_price']
    
    #Business constraints
    min_price = current_price * 0.80   # max 20% decrease
    max_price = current_price * 1.20   # max 20% increase
    price_points = np.linspace(min_price, max_price, 50)
    
    best_revenue = -np.inf
    best_price = current_price
    best_demand = None
    
    for price in price_points:
        row = latest.copy()
        row['unit_price'] = price
        row['price_vs_comp'] = price - avg_comp
        row['comp_price_ratio'] = price / avg_comp if avg_comp > 0 else 1
        row['price_change'] = price - current_price
        row['price_pct_change'] = (price - current_price) / current_price
        
        X = pd.DataFrame([row[features]])
        pred_demand = max(0, model.predict(X)[0])
        revenue = price * pred_demand
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_price = price
            best_demand = pred_demand
    
    #Current baseline
    X_current = pd.DataFrame([latest[features]])
    current_demand = max(0, model.predict(X_current)[0])
    current_revenue = current_price * current_demand
    
    revenue_lift = ((best_revenue - current_revenue) / current_revenue * 100 
                    if current_revenue > 0 else 0)
    
    results.append({
        'product_id': product_id,
        'category': latest['product_category_name'],
        'current_price': round(current_price, 2),
        'optimal_price': round(best_price, 2),
        'price_change_pct': round((best_price - current_price) / current_price * 100, 1),
        'current_revenue': round(current_revenue, 2),
        'optimal_revenue': round(best_revenue, 2),
        'revenue_lift_pct': round(revenue_lift, 1)
    })

results_df = pd.DataFrame(results).sort_values('revenue_lift_pct', ascending=False)
print(results_df.to_string(index=False))
print(f"\nAverage projected revenue lift: {results_df['revenue_lift_pct'].mean():.1f}%")
print(f"Median projected revenue lift:  {results_df['revenue_lift_pct'].median():.1f}%")

#Plot top 10 opportunities
top10 = results_df.head(10)
plt.figure(figsize=(10, 6))
x = range(len(top10))
width = 0.35
plt.bar([i - width/2 for i in x], top10['current_revenue'], width, label='Current Revenue')
plt.bar([i + width/2 for i in x], top10['optimal_revenue'], width, label='Optimized Revenue')
plt.xticks(list(x), top10['product_id'], rotation=45, ha='right')
plt.title('Top 10 Revenue Optimization Opportunities')
plt.ylabel('Projected Revenue')
plt.legend()
plt.tight_layout()
plt.savefig('optimization_results.png')
plt.show()

results_df.to_csv('optimization_results.csv', index=False)
print("\nSaved to optimization_results.csv")
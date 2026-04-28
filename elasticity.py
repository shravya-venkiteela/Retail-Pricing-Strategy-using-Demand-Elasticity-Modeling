import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("retail_price_featured.csv")

#Log-log model per category
results = []

for category in df['product_category_name'].unique():
    cat_df = df[df['product_category_name'] == category].copy()
    
    #Remove zeros to avoid log errors
    cat_df = cat_df[(cat_df['qty'] > 0) & (cat_df['unit_price'] > 0)]
    
    log_price = np.log(cat_df['unit_price'])
    log_qty = np.log(cat_df['qty'])
    
    #OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)
    
    results.append({
        'category': category,
        'elasticity': round(slope, 3),
        'r_squared': round(r_value**2, 3),
        'p_value': round(p_value, 4),
        'n_obs': len(cat_df),
        'avg_price': round(cat_df['unit_price'].mean(), 2),
        'avg_qty': round(cat_df['qty'].mean(), 2)
    })

elasticity_df = pd.DataFrame(results).sort_values('elasticity')
print(elasticity_df.to_string(index=False))

#Visualize
plt.figure(figsize=(10, 6))
colors = ['red' if e < -1 else 'orange' if e < 0 else 'green' 
          for e in elasticity_df['elasticity']]
bars = plt.barh(elasticity_df['category'], elasticity_df['elasticity'], color=colors)
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(x=-1, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
plt.title('Price Elasticity by Category\n(Red = Elastic, Orange = Inelastic, Green = Unusual)')
plt.xlabel('Price Elasticity of Demand')
plt.tight_layout()
plt.savefig('elasticity_by_category.png')
plt.show()

#Save
elasticity_df.to_csv('elasticity_results.csv', index=False)
print("\nSaved to elasticity_results.csv")
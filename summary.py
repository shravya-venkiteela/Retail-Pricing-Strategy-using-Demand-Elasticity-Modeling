import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

results_df = pd.read_csv('optimization_results.csv')
elasticity_df = pd.read_csv('elasticity_results.csv')
results_df['revenue_lift_pct'] = results_df['revenue_lift_pct'].clip(lower=0)

#Summary stats
print("=" * 50)
print("RETAIL PRICING OPTIMIZATION — FINAL SUMMARY")
print("=" * 50)
print(f"Products analyzed:         {len(results_df)}")
print(f"Products to price DOWN:    {(results_df['price_change_pct'] < 0).sum()}")
print(f"Products to price UP:      {(results_df['price_change_pct'] > 0).sum()}")
print(f"Median revenue lift:       {results_df['revenue_lift_pct'].median():.1f}%")
print(f"Average revenue lift:      {results_df['revenue_lift_pct'].mean():.1f}%")
print(f"\nBy category:")
cat_summary = results_df.groupby('category')['revenue_lift_pct'].median().sort_values(ascending=False)
print(cat_summary.round(1).to_string())

#Figure with 3 panels
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)

#Panel 1: Revenue lift distribution
ax1 = fig.add_subplot(gs[0, 0])
results_df['revenue_lift_pct'].clip(upper=400).hist(bins=20, ax=ax1, color='steelblue', edgecolor='white')
ax1.axvline(results_df['revenue_lift_pct'].median(), color='red', linestyle='--', label=f"Median: {results_df['revenue_lift_pct'].median():.1f}%")
ax1.set_title('Distribution of Revenue Lift %')
ax1.set_xlabel('Revenue Lift %')
ax1.set_ylabel('Number of Products')
ax1.legend()

#Panel 2: Current vs Optimal price by category
ax2 = fig.add_subplot(gs[0, 1])
cat_prices = results_df.groupby('category').agg(
    current=('current_price', 'mean'),
    optimal=('optimal_price', 'mean')
)
x = range(len(cat_prices))
ax2.bar([i - 0.2 for i in x], cat_prices['current'], 0.4, label='Current', color='steelblue')
ax2.bar([i + 0.2 for i in x], cat_prices['optimal'], 0.4, label='Optimal', color='orange')
ax2.set_xticks(list(x))
ax2.set_xticklabels(cat_prices.index, rotation=45, ha='right')
ax2.set_title('Avg Current vs Optimal Price by Category')
ax2.set_ylabel('Price ($)')
ax2.legend()

#Panel 3: Elasticity vs revenue lift
ax3 = fig.add_subplot(gs[1, 0])
merged = results_df.merge(elasticity_df[['category','elasticity']], on='category')
for cat in merged['category'].unique():
    cat_data = merged[merged['category'] == cat]
    ax3.scatter(cat_data['elasticity'], cat_data['revenue_lift_pct'].clip(upper=400),
                label=cat, alpha=0.7)
ax3.set_title('Elasticity vs Revenue Lift')
ax3.set_xlabel('Price Elasticity')
ax3.set_ylabel('Revenue Lift %')
ax3.legend(fontsize=7, bbox_to_anchor=(1.05, 1))

#Panel 4: Top 10 biggest opportunities
ax4 = fig.add_subplot(gs[1, 1])
top10 = results_df.nlargest(10, 'revenue_lift_pct')
bars = ax4.barh(top10['product_id'], top10['revenue_lift_pct'], color='steelblue')
ax4.set_title('Top 10 Products by Revenue Lift %')
ax4.set_xlabel('Revenue Lift %')
ax4.axvline(x=results_df['revenue_lift_pct'].median(), color='red',
            linestyle='--', alpha=0.7, label='Median')
ax4.legend()

plt.suptitle('Retail Price Optimization — Project Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved project_summary.png")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('retail_price.csv');
#1. How many unique products and categories?
print("Unique products:", df['product_id'].nunique())
print("Unique categories:", df['product_category_name'].nunique())
print("\nCategory breakdown:")
print(df['product_category_name'].value_counts())

#2. Price vs Demand scatter
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='unit_price', y='qty', hue='product_category_name')
plt.title('Price vs Demand by Category')
plt.xlabel('Unit Price')
plt.ylabel('Quantity Sold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('price_vs_demand.png')
plt.show()

#3. Distribution of unit prices
plt.figure(figsize=(8,4))
sns.histplot(df['unit_price'], bins=30, kde=True)
plt.title('Distribution of Unit Prices')
plt.savefig('price_distribution.png')
plt.show()

#4. How does the price compare to competitors?
df['avg_comp_price'] = df[['comp_1', 'comp_2', 'comp_3']].mean(axis=1)
df['price_vs_comp'] = df['unit_price'] - df['avg_comp_price']
print("\nWe are cheaper than competitors (negative = we're cheaper):")
print(df['price_vs_comp'].describe())

#Average price and demand by category
category_summary = df.groupby('product_category_name').agg(
    avg_price=('unit_price', 'mean'),
    avg_qty=('qty', 'mean'),
    avg_comp_price=('avg_comp_price', 'mean'),
    total_revenue=('total_price', 'sum')
).round(2).sort_values('avg_price', ascending=False)

print(category_summary)

#Price vs competitor price by category
plt.figure(figsize=(10, 5))
x = range(len(category_summary))
width = 0.35
plt.bar([i - width/2 for i in x], category_summary['avg_price'], width, label='Our Price')
plt.bar([i + width/2 for i in x], category_summary['avg_comp_price'], width, label='Avg Competitor Price')
plt.xticks(list(x), category_summary.index, rotation=45, ha='right')
plt.title('Our Price vs Competitor Price by Category')
plt.ylabel('Average Price')
plt.legend()
plt.tight_layout()
plt.savefig('price_vs_competitor.png')
plt.show()
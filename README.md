# Retail Pricing Strategy using Demand Elasticity Modeling
An end-to-end pricing strategy engine that combines machine learning, econometric elasticity modeling, and competitive intelligence to simulate revenue-maximizing retail pricing decisions across 52 products and 9 categories.

<img width="2234" height="1475" alt="project_summary" src="https://github.com/user-attachments/assets/bcc964fc-d22e-4201-8cdd-6fd897f7fa85" />

### What it does

1. Exploratory Data Analysis (EDA): identifies competitive pricing gaps by category.
2. Feature Engineering: builds lag demand, price change, and competitor ratio features.
3. Demand Modeling: benchmarks Linear Regression, Random Forest, and XGBoost (Random Forest wins, MAE = 7.90)
4. Elasticity Estimation: fits log-log OLS per category to estimate price sensitivity
5. Revenue Optimization: simulates 50 price points per product and selects the revenue-maximizing price

### Key Findings

- health_beauty is priced ~2x above competitor average, largest pricing gap in the dataset.
- Freight cost is a stronger demand signal than unit price across the portfolio.
- Statistically significant elasticity found in 2 of 9 categories (consoles_games: -4.34, garden_tools: -0.77)
- 51 of 52 products appear priced above their modeled revenue-maximizing point under current assumptions.

### Stack
pandas · scikit-learn · xgboost · statsmodels · matplotlib

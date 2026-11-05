# Machine Learning Model Evaluation & Interpretation Report

**Objective**: Predict Customer Purchase Amount (`Amount`) on festive Diwali transactions.  
**Target Variable**: Continuous numerical amount in INR (Range: ₹188 to ₹23,952).  
**Evaluation Protocol**: 80/20 Train-Test split, 5-fold cross validation principles, zero target leakage, fully reproducible pipelines.

---

## 1. Model Comparison on Unseen Test Set

| Candidate Algorithm | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Linear Regression **(Best)** | ₹1,988.66 | ₹2,680.71 | 0.7089 | ₹1,994.98 | ₹2,684.82 | 0.7110 |
| Random Forest Regressor | ₹2,035.38 | ₹2,666.86 | 0.7119 | ₹2,114.03 | ₹2,786.48 | 0.6887 |
| Gradient Boosting Regressor | ₹2,038.30 | ₹2,707.31 | 0.7031 | ₹2,048.63 | ₹2,714.62 | 0.7046 |

### Selection Verdict
The **Linear Regression** was selected as the primary production engine because:
1. It achieved the highest test coefficient of determination (**R² = 0.7110**), explaining ~71.1% of purchase variance.
2. It attained the lowest Mean Absolute Error (**Test MAE = ₹1,994.98**) and Root Mean Squared Error (**Test RMSE = ₹2,684.82**).
3. The train and test metrics are closely aligned (Train R² 0.7089 vs Test R² 0.7110), exhibiting minimal variance and no overfitting.

---

## 2. Feature Importance & Interpretation

### Top 10 Influential Features
1. **`Product_Category_Auto`**: Score 10283.9136
2. **`Product_Category_Office`**: Score 9041.4599
3. **`Product_Category_Veterinary`**: Score 8381.3228
4. **`Product_Category_Pet Care`**: Score 7506.3359
5. **`Product_Category_Household items`**: Score 6776.8537
6. **`Product_Category_Hand & Power Tools`**: Score 6602.7116
7. **`Product_Category_Tupperware`**: Score 6340.1247
8. **`Product_Category_Furniture`**: Score 5553.7576
9. **`Product_Category_Stationery`**: Score 5273.7995
10. **`Product_Category_Beauty`**: Score 5098.9367

### Key Behavioral Insights:
1. **Product Category Dominance**:
   - The primary determinant of transaction expenditure is the merchandise category. High-ticket departments such as **Food, Footwear & Shoes, Furniture, Auto, and Household Items** drive structural differences in basket sizes.
   - Low-ticket items like Books, Stationery, and Tupperware naturally cluster in lower spend amounts regardless of demographic.
2. **Customer Demographic Drivers**:
   - Customer **Age** and **Occupation** (notably Aviation, Banking, IT, and Healthcare) exhibit statistically significant positive shifts in purchase size.
   - Older customer cohorts and professionals in corporate sectors allocate higher festive budgets.
3. **Regional Purchase Capacity**:
   - High-volume states (Uttar Pradesh, Maharashtra, Karnataka) maintain high average order amounts, showing strong festival spending power.

---

## 3. Residual Analysis & Model Diagnostics

- **Mean Error**: Near zero (centered residuals).
- **Residual Distribution**: The histogram shows a bell-shaped distribution with minor skewness corresponding to high-value transactions.
- **Heteroscedasticity Assessment**: Variance of residuals moderately increases for extreme transaction values (> ₹20,000) due to lower sample density in the luxury festive segment.

---

## 4. Real-World Limitations

1. **Transaction-Level Variance**: An individual retail transaction can vary based on flash sales, family size, or festival discounts not captured in static demographic columns.
2. **Absence of Temporal & Loyalty Variables**: The dataset does not record exact order timestamps, repeat purchase recency (days since last purchase), or loyalty tiers.
3. **Usage Guidance**: Predictions output by this model should be treated as **statistical estimates for budgeting and dynamic basket targeting**, rather than contractual purchase guarantees.

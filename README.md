# 🏠 House Price Predictor

Predicts California house prices using **Linear Regression** and **Ridge Regression** with full EDA, feature analysis, and visualizations.

## Tech Stack
`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## Features
- Exploratory Data Analysis with correlation heatmap
- StandardScaler preprocessing
- Linear Regression vs Ridge Regression comparison
- Actual vs Predicted scatter plot
- Feature coefficient analysis

## Results

| Model             | RMSE   | MAE    | R² Score |
|-------------------|--------|--------|----------|
| Linear Regression | ~0.726 | ~0.531 | ~0.576   |
| Ridge Regression  | ~0.726 | ~0.531 | ~0.576   |

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python house_price_predictor.py
```

## Output
- `correlation_heatmap.png` — feature correlation matrix
- `actual_vs_predicted.png` — model prediction quality

## Dataset
California Housing dataset from `sklearn.datasets` — no download needed.

---
**Author:** Divya Nimbalkar | [GitHub](https://github.com/divya-09nimbalkar) | [LinkedIn](https://www.linkedin.com/in/divya-nimbalkar09/)

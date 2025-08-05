# 🛒 Sales Forecasting with Facebook Prophet – Rossmann Store Sales

This project showcases **time series forecasting** for **Rossmann retail stores** using **Facebook Prophet**, a powerful open-source forecasting tool. The main goal is to **predict future sales** for selected stores using historical data, and evaluate performance using **store-wise RMSE**.

---

## 📦 Dataset

**Source**: [Rossmann Store Sales Dataset (Kaggle)](https://www.kaggle.com/c/rossmann-store-sales/data)  
**Files Used**:
- `train.csv`: Historical daily sales data
- `store.csv`: Additional store-level information

Each row represents a single store on a specific date, with features such as:
- `Sales`, `Customers`, `Date`, `Promo`
- `StoreType`, `Assortment`, `CompetitionDistance`, etc.

---

## 🔧 Tools & Libraries Used

- Python 3.x
- [Facebook Prophet](https://facebook.github.io/prophet/)
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

## 📌 Project Workflow

1. **Data Merging & Cleaning**
   - Combined `train.csv` with `store.csv`
   - Removed closed stores and entries with zero sales
   - Handled missing values (e.g., competition data, promo intervals)

2. **Forecasting per Store**
   - Selected individual stores (Store 1, Store 2, etc.)
   - Trained Prophet models on daily sales data
   - Forecasted sales for next 30 days

3. **Evaluation**
   - Calculated **RMSE** between actual and predicted sales
   - Compared store-wise performance

---

## 📊 Sample RMSE Results

| Store | RMSE        |
|-------|-------------|
| 1     | ✅ 698.04   |
| 6     | 1446.94     |
| 7     | ❌ 2935.31  |

📌 **Store 1** had the best forecast performance, while **Store 7** showed the highest deviation.

---

## 📈 Visuals & Forecast

- Prophet’s forecast components (trend + weekly + yearly)
- 30-day future sales forecast
- Store-level plots showing historical + predicted sales

---

## 🧠 Key Insight

> This project demonstrates how **Prophet can effectively forecast sales patterns** for individual stores using minimal tuning. It's a great example of applying time series modeling to real retail data for **demand prediction** and **business planning**.

---

## 🚀 Run it Yourself

```bash
# Install dependencies
pip install prophet pandas scikit-learn matplotlib seaborn

# Run in Jupyter Notebook or Google Colab

# 🚀 Engine Failure Prediction System

This project predicts the Remaining Useful Life (RUL) of turbofan engines using NASA's CMAPSS dataset, and provides a real-time engine health monitoring dashboard.

---

## 📚 Project Structure

- `data/`: Raw training, test, and RUL files
- `notebooks/`: Data processing, model training notebook
- `src/`: Saved model and scalers
- `app.py`: Streamlit dashboard application
- `final_report.md`: Phase 1 detailed write-up
- `README.md`: Project overview

---

## 📈 Phase 1: RUL Prediction System

- Data preprocessing and RUL labeling
- Feature normalization and degradation analysis
- Model training with Gradient Boosting Regressor
- Confidence interval estimation for predictions
- Evaluation using MAE, RMSE, and R² Score

---

## 📊 Phase 2: Real-Time Engine Health Dashboard (Streamlit)

- Engine Health Overview: Current cycle, predicted RUL
- Live Sensor Trends: Sensor reading trends across cycles
- Maintenance Planning: Prioritized engines based on risk
- Risk Distribution Pie Charts

---

## 🛠 Requirements

Install necessary packages:
```bash
pip install -r requirements.txt

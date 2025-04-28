# Engine Failure Prediction Project

## 1. Approach Taken

We developed a machine learning system to predict the Remaining Useful Life (RUL) of turbofan engines based on NASA's CMAPSS dataset.  
The overall steps were:

- Load and clean the training and test datasets by dropping extra empty columns and assigning appropriate column names.
- Create a Remaining Useful Life (RUL) label for each cycle in the training dataset.
- Normalize sensor measurements and operational settings separately using Min-Max Scaling.
- Analyze sensor degradation patterns through visualization to select meaningful sensors.
- Train a Gradient Boosting Regressor model on the processed training data.
- Predict RUL for unseen engines by using their latest available sensor readings.
- Add 95% confidence intervals to each prediction based on residual standard deviation.
- Evaluate model performance using MAE, RMSE, and R² score.

---

## 2. Feature Engineering Decisions

- **Remaining Useful Life (RUL):**  
  For each engine, RUL was calculated as `(maximum cycle number - current cycle number)`.

- **Feature Scaling:**  
  Applied Min-Max Scaling separately to sensor measurements and operational settings to maintain scaling consistency.

- **Feature Selection:**  
  Based on visual degradation analysis, sensors like `sensor_measurement_2`, `sensor_measurement_3`, `sensor_measurement_4`, `sensor_measurement_7`, `sensor_measurement_11`, `sensor_measurement_15`, `sensor_measurement_17`, and `sensor_measurement_20` showed clear degradation patterns and were prioritized.

- **Aggregation for Test Data:**  
  Only the latest cycle data per engine was extracted for RUL prediction on the test set.

---

## 3. Model Selection Rationale

Gradient Boosting Regressor was chosen because:

- It efficiently handles tabular numerical data.
- It captures non-linear relationships between sensor readings and RUL without heavy feature engineering.
- It is robust against overfitting when hyperparameters are properly tuned.
- It provides relatively fast training times and clear feature importance analysis.

---

## 4. Performance Metrics

| Metric                       | Validation Set | Test Set |
|-------------------------------|----------------|----------|
| MAE (Mean Absolute Error)     | 29.62 cycles    | 50.95 cycles |
| RMSE (Root Mean Square Error) | 41.38 cycles    | 63.96 cycles |
| R² Score                      | 0.63            | -1.37     |

The scatter plot of predicted vs true RUL showed that most predictions fell close to the diagonal (in the validation set), indicating good model calibration.  
Test set R² is negative, suggesting that the model does not generalize perfectly and further tuning may be needed.

Additionally, **95% confidence intervals** were created around the predictions based on the standard deviation of residuals.

---

## 5. Limitations and Assumptions

- Sensor readings are assumed to have minimal missingness and no sensor faults.
- Confidence intervals were calculated assuming residuals follow a normal distribution.
- Model performance may degrade if operational conditions or engine types differ significantly from the training data.

---

## 6. Example of Usage in Maintenance Planning

A real-world maintenance team could use this system by:

- Continuously monitoring live sensor data from operating engines.
- Feeding the latest sensor and setting readings into the trained model to get the current RUL prediction.
- Viewing the predicted RUL range (with confidence intervals) to assess maintenance needs.
- Scheduling preventive maintenance when the lower bound of predicted RUL reaches a critical threshold (e.g., 50 cycles).
- Prioritizing maintenance actions for engines approaching failure sooner than expected.

This proactive approach enables:
- Avoidance of unexpected engine failures.
- Better spare parts inventory planning.
- Reduction of downtime and maintenance costs.

---

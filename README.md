# Used Car Price Prediction — Kaggle Competition

## Project Description
- This project is for the **Predictive Analytics (BUSA8001)** unit at Macquarie University.
- The team competed in a Kaggle in-class competition to build a regression model that predicts the price of used cars based on vehicle attributes, dealer information, and market signals.
- The model is evaluated using **Mean Absolute Percentage Error (MAPE)**.

**Team Name:** BUSA8001_That's not my name

## Team Members
| Role | Name | Task |
|------|------|------|
| Team Leader & Member 3 | **Quoc Phong (Leo) Nguyen** | Task 3 — Model Building & Competition |
| Member 1 | Ha My Dang | Task 1 — Problem Description & EDA |
| Member 2 | Luong Phuong Anh Pham | Task 2 — Data Cleaning & Feature Engineering |

## Introduction
The used car market suffers from information asymmetry — buyers and sellers often lack reliable pricing benchmarks. This project builds a machine learning pipeline to accurately forecast used car prices, with real-world applications for consumers, dealerships, fleet managers, and digital auto platforms.

The dataset contains **37 variables** covering vehicle specifications, dealer information, and market signals across 8,000 training observations.

## Tools & Libraries Used
- **Python** (pandas, numpy)
- **scikit-learn** — model training, cross-validation, hyperparameter tuning
- **LightGBM** — alternative gradient boosting model (Appendix)
- **matplotlib / seaborn / plotly** — data visualisation
- **Jupyter Notebook** — development environment
- **Kaggle** — competition platform and leaderboard

## Project Components

### Task 1 — Problem Description & Initial Data Analysis *(Ha My Dang)*
- Defined the business problem and real-world applications of used car price prediction
- Described the evaluation metric (MAPE) and its advantages and limitations
- Performed initial data analysis: data types, shape, and missing value overview
- Categorised 37 variables into 18 numeric, 2 ordinal, and 16 nominal features
- Conducted univariate analysis across vehicle specs, manufacturers, and aesthetics

### Task 2 — Data Cleaning, Missing Values & Feature Engineering *(Luong Phuong Anh Pham)*
- **Numerical cleaning:** Extracted numeric values from compound text columns
  (e.g., `back_legroom`, `power`, `torque`)
- **Feature engineering:** Created 11+ new features by decomposing compound columns
  (`power_hp`, `power_rpm`, `torque_lbft`, `torque_rpm`, `cylarr_type`, `gear_num`, etc.)
- **Additional features:** `age`, `listed_year/month/date`, `brand_avg_price` (target encoding)
- **Missing value imputation:** Custom `Imputer` class using hierarchical fallback chains
  (mode/mean/median by `make_name → model_name → year → global`)
- **Categorical encoding:** Boolean encoding, top-5 + "other" grouping, One-Hot Encoding
- **Additional preprocessing:** Outlier capping (99.8th percentile), log transformation
  (`log_mileage`, `age_cubed`), and 15+ interaction/ratio features

### Task 3 — Model Building, Tuning & Competition *(Leo Nguyen — Team Leader)*
- **EDA:** Correlation analysis and scatter plots to identify linear and non-linear relationships between features and the target variable `price`
- **Model selection:** Evaluated 3 regression models:
  | Model | Validation MAPE | Validation R² |
  |-------|----------------|---------------|
  | Linear Regression | 15.32% | 0.828 |
  | Decision Tree Regressor | 16.50% | 0.828 |
  | **Random Forest Regressor** | **11.60%** | **0.913** |
- **Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation
  (best: `n_estimators=500`, `max_depth=25`, `max_features='sqrt'`)
- **Feature selection:** Top 67 features by importance → validation MAPE improved to **11.50%**
- **Final submission:** Retrained on full dataset → **Kaggle public score: 0.09236 (9.24%)**
- **Alternative model:** LightGBM achieved **6.97% MAPE** on Kaggle (included in Appendix)
- **Iterative approach:** 128 Kaggle submissions to systematically refine model performance

## Key Findings
- Random Forest significantly outperformed Linear Regression and Decision Tree by capturing non-linear interactions between features.
- Feature engineering (interaction terms, polynomial features, target encoding) was critical to reducing prediction error.
- Training on the full dataset (vs. train-validation split) improved Kaggle score from ~11.5% to 9.24%, confirming strong model generalisation.
- LightGBM (gradient boosting) demonstrated superior performance at 6.97% MAPE, highlighting the value of exploring advanced ensemble methods.

## Future Enhancements
1. Implement stacking/blending ensemble of Random Forest and LightGBM.
2. Apply more advanced hyperparameter search (e.g., Bayesian optimisation).
3. Explore additional target encoding strategies for high-cardinality categorical features.
4. Investigate geographic clustering using latitude/longitude for regional price signals.
5. Deploy the model as a web-based used car price estimator.

## Contributing
This is a group assignment completed as part of university coursework at Macquarie University.

## License
This project is part of a university assignment and should be used for educational purposes only.

## Acknowledgments
- Dataset and competition hosted on **Kaggle** as part of BUSA8001
- Feature engineering strategies inspired by domain knowledge of the automotive market
- Gradient boosting exploration based on LightGBM documentation

## 📜 Kaggle Competition Reference Letter

[📄 View Reference Letter (PDF)](Kaggle_Group_competition_reference_letter.pdf)
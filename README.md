# AutoValue AI – Used Car Price Prediction

An end-to-end Machine Learning project that estimates the resale value of used cars based on vehicle specifications, ownership details, and condition.

## Overview

AutoValue AI uses a trained Support Vector Regression (SVR) model to predict used car prices. The project covers the complete ML workflow, from exploratory data analysis and preprocessing to model evaluation, serialization, and an interactive Streamlit application.

## Features

- Predict estimated used car prices.
- Accept 11 vehicle input features.
- Encode categorical variables using One-Hot Encoding.
- Scale numerical features using StandardScaler.
- Compare multiple regression algorithms.
- Tune model hyperparameters using RandomizedSearchCV.
- Display prediction history during the current session.
- Explore model performance through an About Model page.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Workflow

1. Import libraries and load the dataset.
2. Inspect data types, missing values, and duplicates.
3. Perform univariate, bivariate, and multivariate analysis.
4. Handle missing values.
5. Encode categorical features.
6. Separate independent and dependent variables.
7. Split data into training and testing sets.
8. Scale numerical features.
9. Train and tune regression models.
10. Compare model performance.
11. Select the best-performing model.
12. Generate predictions.
13. Save the trained model and preprocessing objects using Joblib.
14. Develop an interactive Streamlit application.

## Dataset

- Records: 10,000
- Input features: 11
- Target variable: `price_usd`
- Manufacturing year range: 1995–2023

### Input Features

`make_year`, `mileage_kmpl`, `engine_cc`, `fuel_type`, `owner_count`, `brand`, `transmission`, `color`, `service_history`, `accidents_reported`, `insurance_valid`

## Model Performance

The selected model was Tuned Support Vector Regression (SVR).

| Metric | Test Result |
|---|---:|
| R² Score | 0.8766 |
| MAE | $790.80 |
| RMSE | $990.78 |

The model achieved an R² score of approximately 87.66% on the held-out test dataset.

## Project Structure

```text
AutoValue-AI/
├── app/
│   └── app.py
├── dataset/
│   └── used_car_price_dataset_extended.csv
├── models/
│   └── used_car_price_model.pkl
├── notebooks/
│   └── 01_data_understanding.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AutoValue-AI.git
cd AutoValue-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

## Important Limitations

- The dataset covers manufacturing years from 1995 to 2023. Predictions for newer vehicles are outside the training range and may be less reliable.
- The predicted price is an estimate, not a verified market valuation.
- Model performance is measured on this dataset and does not guarantee accuracy in real-world used-car markets.
- Dataset provenance and representativeness should be verified before using the model commercially.

## Future Improvements

- Integrate real-world used car listings.
- Add location-based price comparisons.
- Improve model validation and generalization.
- Store prediction history in a database.
- Deploy the application publicly.

## Author

**Saksham Kedare**

GitHub: [Add your GitHub profile link]

Portfolio: https://portfolio-website-six-eta-11.vercel.app/
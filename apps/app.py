
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AutoValue AI",
    page_icon="🚗",
    layout="wide"
)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "used_car_price_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_data = load_model()

    model = model_data["model"]
    encoder = model_data["encoder"]
    scaler = model_data["scaler"]

    numeric_features = model_data["numeric_features"]
    categorical_features = model_data["categorical_features"]
    encoded_columns = model_data["encoded_columns"]
    feature_columns = model_data["feature_columns"]

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0B0D12;
    color: #F5F7FA;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Headings */
h1, h2, h3 {
    color: #F5F7FA !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #11141B;
    border-right: 1px solid #252A35;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #8AB4FF;
    color: #080B12;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.65rem 1rem;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #B3D0FF;
    color: #080B12;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #171B24;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #151A24;
    border: 1px solid #283247;
    border-radius: 12px;
    padding: 20px;
}

/* Dividers */
hr {
    border-color: #252A35;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

with st.sidebar:

    st.markdown(
        "# AUTO<span style='color:#8AB4FF'>VALUE</span>",
        unsafe_allow_html=True
    )

    st.caption("INTELLIGENT VEHICLE VALUATION")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Price Prediction", "About Model"],
        key="navigation"
    )

    st.divider()

    st.caption("MODEL INFORMATION")
    st.markdown("**Algorithm:** Tuned SVR")
    st.markdown("**R² Score:** 87.66%")
    st.markdown("**Dataset:** 10,000 records")

    st.divider()

    st.caption("AUTO VALUE AI · 2026")


# --------------------------------
# DASHBOARD PAGE
# --------------------------------

if page == "Dashboard":

    st.caption("OVERVIEW")

    st.title("Welcome to AutoValue AI")

    st.markdown(
        "Your intelligent workspace for estimating "
        "used vehicle resale values."
    )

    st.divider()

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.caption("DATASET RECORDS")
            st.metric("Total Vehicles", "10,000")

    with col2:
        with st.container(border=True):
            st.caption("MODEL PERFORMANCE")
            st.metric("Test R² Score", "87.66%")

    with col3:
        with st.container(border=True):
            st.caption("AVERAGE PREDICTION ERROR")
            st.metric("Mean Absolute Error", "$790.80")

    st.divider()

    # Project overview
    st.subheader("Project Overview")

    st.write(
        "AutoValue AI uses a trained Support Vector "
        "Regression model to estimate used car prices "
        "based on vehicle specifications and condition."
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):

            st.markdown("### Price Prediction")

            st.write(
                "Enter vehicle details and generate "
                "an estimated resale price."
            )

            st.button(
                "Go to Price Prediction",
                key="dashboard_prediction",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    {"navigation": "Price Prediction"}
                )
            )

    with col2:
        with st.container(border=True):

            st.markdown("### Model Information")

            st.write(
                "Explore the algorithm, dataset, "
                "evaluation metrics and model details."
            )

            st.button(
                "Explore Model",
                key="dashboard_model",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    {"navigation": "About Model"}
                )
            )

    st.divider()

    st.subheader("How It Works")

    st.markdown("""
    1. **Input:** Enter the vehicle's specifications.
    2. **Preprocessing:** Encode categorical features and scale numerical features.
    3. **Prediction:** The trained SVR model estimates the vehicle price.
    4. **Result:** View the estimated valuation and prediction history.
    """)

    st.info(
        "The model was evaluated on a held-out test dataset. "
        "Its predictions are estimates, not guaranteed market prices."
    )

    st.stop()


# --------------------------------
# ABOUT MODEL PAGE
# --------------------------------

if page == "About Model":

    st.title("About AutoValue AI")

    st.markdown(
        "Understand how our machine learning model "
        "estimates used car prices."
    )

    st.divider()

    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "SVR")

    with col2:
        st.metric("R² Score", "87.66%")

    with col3:
        st.metric("Dataset Size", "10,000")

    st.divider()

    st.subheader("Model Performance")

    st.markdown("""
    | Evaluation Metric | Result |
    |---|---:|
    | R² Score | 0.8766 |
    | Mean Absolute Error | $790.80 |
    | Root Mean Squared Error | $990.78 |
    """)

    st.subheader("Dataset Information")

    st.write("""
    The model was trained using a used-car dataset
    containing 10,000 records and 11 input features.

    Features include:
    - Manufacturing year
    - Mileage
    - Engine capacity
    - Fuel type
    - Brand
    - Transmission
    - Previous owners
    - Service history
    - Accident history
    - Insurance validity
    - Car colour
    """)

    st.warning(
        "The dataset contains records from 1995 to 2023. "
        "Predictions outside this range may be less reliable."
    )

    st.caption(
        "AutoValue AI | Machine Learning Project"
    )

    st.stop()


# --------------------------------
# PRICE PREDICTION PAGE
# --------------------------------

submitted = False

st.title("AutoValue AI")

st.markdown(
    "Discover the estimated value of your vehicle "
    "through machine learning."
)

st.divider()

# --------------------------------
# VEHICLE INFORMATION FORM
# --------------------------------

st.markdown("## Vehicle Information")

st.caption(
    "Enter the details of your vehicle to estimate its resale value."
)

with st.form("car_prediction_form"):

    # General details
    st.markdown("### General Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox(
            "Car Brand",
            [
                "BMW", "Chevrolet", "Ford", "Honda",
                "Hyundai", "Kia", "Nissan", "Tesla",
                "Toyota", "Volkswagen"
            ]
        )

    with col2:
        make_year = st.number_input(
            "Manufacturing Year",
            min_value=1995,
            max_value=2026,
            value=2018,
            step=1
        )

    with col3:
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "Electric"]
        )

    st.divider()

    # Technical specifications
    st.markdown("### Technical Specifications")

    col1, col2, col3 = st.columns(3)

    with col1:
        engine_cc = st.number_input(
            "Engine Capacity (CC)",
            min_value=800,
            max_value=5000,
            value=1500,
            step=100
        )

    with col2:
        mileage_kmpl = st.number_input(
            "Mileage (KM/L)",
            min_value=5.0,
            max_value=35.0,
            value=18.5,
            step=0.5
        )

    with col3:
        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

    st.divider()

    # Ownership and condition
    st.markdown("### Ownership & Condition")

    col1, col2, col3 = st.columns(3)

    with col1:
        owner_count = st.selectbox(
            "Previous Owners",
            [1, 2, 3, 4, 5]
        )

    with col2:
        accidents_reported = st.selectbox(
            "Accidents Reported",
            [0, 1, 2, 3, 4, 5]
        )

    with col3:
        service_history = st.selectbox(
            "Service History",
            ["Full", "Partial", "Unknown"]
        )

    col1, col2 = st.columns(2)

    with col1:
        color = st.selectbox(
            "Car Color",
            ["White", "Black", "Blue", "Red", "Gray", "Silver"]
        )

    with col2:
        insurance_valid = st.selectbox(
            "Insurance Valid",
            ["Yes", "No"]
        )

    st.divider()

    # Submit button must be INSIDE the form
    submitted = st.form_submit_button(
        "Predict Car Price",
        use_container_width=True
    )


# --------------------------------
# PRICE PREDICTION PROCESSING
# --------------------------------

if submitted:

    # Create input DataFrame
    new_car = pd.DataFrame({
        "make_year": [make_year],
        "mileage_kmpl": [mileage_kmpl],
        "engine_cc": [engine_cc],
        "fuel_type": [fuel_type],
        "owner_count": [owner_count],
        "brand": [brand],
        "transmission": [transmission],
        "color": [color],
        "service_history": [service_history],
        "accidents_reported": [accidents_reported],
        "insurance_valid": [insurance_valid]
    })

    # Warn about extrapolation beyond training years
    if make_year > 2023:

        st.warning(
            "This vehicle's manufacturing year is outside "
            "the training dataset (1995–2023). "
            "The prediction may be less reliable."
        )

    # Encode categorical features
    encoded_data = encoder.transform(
        new_car[categorical_features]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=new_car.index
    )

    # Scale numerical features
    scaled_data = scaler.transform(
        new_car[numeric_features]
    )

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=numeric_features,
        index=new_car.index
    )

    # Combine processed features
    processed_data = pd.concat(
        [scaled_df, encoded_df],
        axis=1
    )

    processed_data = processed_data[feature_columns]

    # Generate prediction
    predicted_price = model.predict(processed_data)[0]

    # Save prediction to session history
    st.session_state.prediction_history.append({
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Brand": brand,
        "Year": make_year,
        "Fuel": fuel_type,
        "Mileage": mileage_kmpl,
        "Predicted Price (USD)": round(predicted_price, 2)
    })

    # --------------------------------
    # PREDICTION RESULT
    # --------------------------------

    st.divider()

    st.markdown("## Vehicle Valuation")

    with st.container(border=True):

        st.caption("ESTIMATED RESALE VALUE")

        st.markdown(
            f"<h1 style='color:#8AB4FF; font-size:46px;'>"
            f"${predicted_price:,.2f}</h1>",
            unsafe_allow_html=True
        )

        st.caption("Machine learning generated estimate · USD")

    # Vehicle summary
    st.markdown("### Vehicle Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Brand", brand)
        st.metric("Manufacturing Year", make_year)

    with col2:
        st.metric("Engine Capacity", f"{engine_cc} CC")
        st.metric("Mileage", f"{mileage_kmpl} KM/L")

    with col3:
        st.metric("Transmission", transmission)
        st.metric("Previous Owners", owner_count)

    st.caption(
        "This estimate is based on the trained SVR model and "
        "historical dataset patterns. It is not a verified "
        "market valuation or guaranteed selling price."
    )


# --------------------------------
# PREDICTION HISTORY
# --------------------------------

st.divider()

st.markdown("## Recent Predictions")

st.caption(
    "Your previous vehicle valuation results in this session."
)

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions yet. Generate your first vehicle valuation!"
    )

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and feature columns
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("purchase_intention_model.pkl")

model, feature_columns = load_model()

st.set_page_config(
    page_title="Online Purchase Intention Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Purchase Intention Prediction System")
st.write("Predict whether a visitor is likely to make an online purchase based on browsing behavior.")

st.divider()

# -----------------------------
# Browsing Behaviour
# -----------------------------
st.header("Browsing Activity")

col1, col2 = st.columns(2)

with col1:
    administrative = st.slider("Administrative Pages", 0, 20, 2)
    admin_duration = st.slider("Administrative Duration", 0, 1000, 30)
    informational = st.slider("Informational Pages", 0, 20, 1)
    info_duration = st.slider("Informational Duration", 0, 1000, 20)

with col2:
    product_pages = st.slider("Product Pages", 0, 100, 10)
    product_duration = st.slider("Product Duration", 0, 5000, 500)
    bounce_rate = st.slider("Bounce Rate", 0.0, 1.0, 0.2, 0.01)
    exit_rate = st.slider("Exit Rate", 0.0, 1.0, 0.2, 0.01)
    page_value = st.slider("Page Value", 0.0, 1000.0, 50.0)
    special_day = st.slider("Special Day Score", 0.0, 1.0, 0.0)

st.divider()

# -----------------------------
# System Information
# -----------------------------
st.header("Technical Information")

col3, col4 = st.columns(2)

with col3:
    operating_system = st.selectbox("Operating System", [1,2,3,4,5,6,7,8])
    browser = st.selectbox("Browser", [1,2,3,4,5,6,7,8,9,10,11,12,13])
    region = st.selectbox("Region", [1,2,3,4,5,6,7,8,9])

with col4:
    traffic_type = st.selectbox("Traffic Type", list(range(1,21)))

st.divider()

# -----------------------------
# Visitor Info
# -----------------------------
st.header("Visitor Information")

visitor_type = st.selectbox("Visitor Type", ["New Visitor", "Returning Visitor"])
weekend = st.selectbox("Weekend Visit", ["No", "Yes"])

month = st.selectbox(
    "Month",
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
)

visitor_returning = 1 if visitor_type == "Returning Visitor" else 0
weekend_value = 1 if weekend == "Yes" else 0

# Month One-hot encoding
months = [
    "Month_Feb","Month_Mar","Month_Apr","Month_May","Month_Jun",
    "Month_Jul","Month_Aug","Month_Sep","Month_Oct","Month_Nov","Month_Dec"
]

month_values = {m:0 for m in months}

if month != "Jan":
    month_values[f"Month_{month}"] = 1

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Purchase Intention", use_container_width=True):

    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": admin_duration,
        "Informational": informational,
        "Informational_Duration": info_duration,
        "ProductRelated": product_pages,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": page_value,
        "SpecialDay": special_day,
        "OperatingSystems": operating_system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType_Returning_Visitor": visitor_returning,
        "Weekend": weekend_value,
        **month_values
    }])

    # Ensure column order matches training
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    st.write(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.metric("Purchase Probability", f"{probability*100:.2f}%")
    st.progress(float(probability))

    if prediction == 1:
        st.success("Customer is likely to PURCHASE.")
        st.info("Recommendation: Offer bundle deals or premium products.")
    else:
        st.error("Customer is UNLIKELY to purchase.")
        st.info("Recommendation: Provide discount coupons or promotional offers.")

    st.subheader("Top Factors Influencing the Model")

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    top_features = importance.head(5)

    st.bar_chart(top_features.set_index("Feature"))
```python
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("purchase_intention_model.pkl")

model, feature_columns = load_model()

st.set_page_config(
    page_title="Online Purchase Intention Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Online Purchase Intention Prediction System")
st.caption("Predict whether a website visitor is likely to make a purchase based on browsing behaviour.")

st.divider()

# -----------------------------
# Browsing Activity
# -----------------------------
st.header("Browsing Activity")

administrative = st.number_input("Administrative Pages", min_value=0, value=2)
admin_duration = st.number_input("Administrative Duration", min_value=0.0, value=30.0)

informational = st.number_input("Informational Pages", min_value=0, value=1)
info_duration = st.number_input("Informational Duration", min_value=0.0, value=20.0)

product_pages = st.number_input("Product Pages", min_value=0, value=10)
product_duration = st.number_input("Product Duration", min_value=0.0, value=500.0)

bounce_rate = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.2)
exit_rate = st.number_input("Exit Rate", min_value=0.0, max_value=1.0, value=0.2)

page_value = st.number_input("Page Value", min_value=0.0, value=50.0)
special_day = st.number_input("Special Day Score", min_value=0.0, max_value=1.0, value=0.0)

st.divider()

# -----------------------------
# Visitor Information
# -----------------------------
st.header("Visitor Information")

visitor_type = st.selectbox(
    "Visitor Type",
    ["New Visitor", "Returning Visitor"]
)

weekend = st.selectbox(
    "Weekend Visit",
    ["No", "Yes"]
)

month = st.selectbox(
    "Month",
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
)

visitor_returning = 1 if visitor_type == "Returning Visitor" else 0
weekend_value = 1 if weekend == "Yes" else 0

# -----------------------------
# Default Technical Info
# -----------------------------
operating_system = 2   # Windows
browser = 2            # Chrome
region = 1             # Asia
traffic_type = 2       # Default value

# -----------------------------
# Month Encoding
# -----------------------------
months = [
    "Month_Feb","Month_Mar","Month_Apr","Month_May","Month_Jun",
    "Month_Jul","Month_Aug","Month_Sep","Month_Oct","Month_Nov","Month_Dec"
]

month_values = {m:0 for m in months}

if month != "Jan":
    month_values[f"Month_{month}"] = 1

st.divider()

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Predict Purchase Intention", use_container_width=True)

with col2:
    if st.button("Reset Inputs", use_container_width=True):
        st.rerun()

# -----------------------------
# Prediction
# -----------------------------
if predict_button:

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

    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.subheader("Prediction Result")

    st.metric("Purchase Probability", f"{probability*100:.2f}%")
    st.progress(float(probability))

    if prediction == 1:
        st.success("Customer is likely to PURCHASE.")
        st.info("Recommendation: Offer bundle deals or premium products.")
    else:
        st.error("Customer is UNLIKELY to purchase.")
        st.info("Recommendation: Provide discount coupons or promotional offers.")

    # -----------------------------
    # Customer Behavior Summary
    # -----------------------------
    st.subheader("Customer Behavior Summary")

    summary_points = []

    if product_pages > 30:
        summary_points.append("Customer viewed many product pages.")

    if product_duration > 1500:
        summary_points.append("Customer spent significant time browsing products.")

    if bounce_rate < 0.05:
        summary_points.append("Low bounce rate indicates strong engagement.")

    if visitor_returning == 1:
        summary_points.append("Customer is a returning visitor.")

    if page_value > 200:
        summary_points.append("High page value suggests strong purchase intent.")

    if not summary_points:
        summary_points.append("Browsing behavior indicates limited purchase engagement.")

    for point in summary_points:
        st.write(f"• {point}")

    # -----------------------------
    # Business Insight Panel
    # -----------------------------
    st.subheader("Business Insight")

    if prediction == 1:
        st.info(
            "This visitor shows strong indicators of purchase intent. "
            "Businesses should consider offering product bundles, premium items, "
            "or cross-selling recommendations to maximize revenue."
        )
    else:
        st.warning(
            "This visitor currently shows low purchase intent. "
            "Offering discount coupons, limited-time promotions, "
            "or personalized recommendations may help encourage a purchase."
        )

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("Top Influential Factors")

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance.head(5).set_index("Feature"))
```

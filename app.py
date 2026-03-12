from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("purchase_intention_model.pkl")

# Recommendation logic
def recommend_action(prediction):
    
    if prediction == 1:
        return "Customer likely to purchase. Recommend bundle deals or premium offers."
    else:
        return "Customer unlikely to purchase. Offer discount coupons or promotions."


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Convert input to dataframe
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    recommendation = recommend_action(prediction)

    return jsonify({
        "prediction": int(prediction),
        "recommendation": recommendation
    })


if __name__ == "__main__":
    app.run(debug=True)
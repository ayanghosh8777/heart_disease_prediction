import pickle
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application


# Load model and scaler
best_model = pickle.load(
    open("models/new_best_model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_disease():

    if request.method == "POST":

        age = float(request.form.get("age"))
        sex = float(request.form.get("sex"))
        cp = float(request.form.get("cp"))
        trestbps = float(request.form.get("trestbps"))
        chol = float(request.form.get("chol"))
        fbs = float(request.form.get("fbs"))
        restecg = float(request.form.get("restecg"))
        thalach = float(request.form.get("thalach"))
        exang = float(request.form.get("exang"))
        oldpeak = float(request.form.get("oldpeak"))
        slope = float(request.form.get("slope"))
        ca = float(request.form.get("ca"))
        thal = float(request.form.get("thal"))

        # Create feature array
        features = np.array([
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        prediction = best_model.predict(features_scaled)

        probability = best_model.predict_proba(features_scaled)

        risk_percentage = probability[0][1] * 100

        if prediction[0] == 1:
                result = "Heart Disease Detected"
        else:
                result = "No Heart Disease Detected"
        if risk_percentage < 30:
              risk_level = "Low Risk"
        elif risk_percentage < 70:
              risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
        # Return result
        return render_template(
            "home.html",
             prediction=result,
             risk_percentage=risk_percentage,
             risk_level=risk_level
        )


    # If GET request
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
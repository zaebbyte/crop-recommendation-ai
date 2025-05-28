from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("crop_model.pkl")
le_crop = joblib.load("crop_label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        soil = int(request.form['soil_type'])
        state = int(request.form['state'])
        season = int(request.form['season'])
        rainfall = float(request.form['rainfall'])
        ground_water = float(request.form['ground_water'])
        temperature = float(request.form['temperature'])

        features = np.array([[soil, state, season, rainfall, ground_water, temperature]])
        prediction = model.predict(features)[0]
        crop_name = le_crop.inverse_transform([prediction])[0]

        return render_template("index.html", prediction_text=f"Recommended Crop: {crop_name}")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

demand_model = joblib.load('demand_forecast_model.joblib')
communication_model = joblib.load('customer_communication_model.joblib')
route_model = joblib.load('route_optimization_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/demand', methods=['POST'])
def predict_demand():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = demand_model.predict(df)
    return jsonify({'predicted_demand': prediction[0]})

@app.route('/predict/communication', methods=['POST'])
def predict_communication():
    data = request.json
    text = data['text']
    prediction = communication_model.predict([text])
    return jsonify({'predicted_category': prediction[0]})

@app.route('/predict/route', methods=['POST'])
def predict_route():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = route_model.predict(df)
    return jsonify({'predicted_delivery_time': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
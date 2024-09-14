import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_demand_forecast_model():
    np.random.seed(42)
    data = {
        'historical_sales': np.random.normal(500, 100, 1000),
        'promotional_effect': np.random.uniform(0, 1, 1000),
        'economic_indicator': np.random.normal(100, 10, 1000),
        'demand': np.random.normal(600, 200, 1000)
    }
    df = pd.DataFrame(data)
    X = df[['historical_sales', 'promotional_effect', 'economic_indicator']]
    Y = df['demand']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)
    joblib.dump(model, 'demand_forecast_model.joblib')
    logging.info("Demand Forecast model saved successfully.")

def create_customer_communication_model():
    data = {
        'text': [
            'Where is my order?',
            'I want to return my product.',
            'Can you update me on the delivery status?',
            'What are the shipping options?',
            'I would like to change my order.'
        ],
        'label': [
            'tracking', 'returns', 'tracking', 'shipping', 'orders'
        ]
    }
    df = pd.DataFrame(data)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['text'], df['label'])
    joblib.dump(model, 'customer_communication_model.joblib')
    logging.info("Customer Communication model saved successfully.")

def create_route_optimization_model():
    np.random.seed(42)
    data = {
        'distance_km': np.random.uniform(1, 50, 1000),
        'traffic_density': np.random.uniform(0, 10, 1000),
        'weather_impact': np.random.uniform(0, 1, 1000),
        'delivery_time': np.random.uniform(15, 90, 1000)
    }
    df = pd.DataFrame(data)
    X = df[['distance_km', 'traffic_density', 'weather_impact']]
    Y = df['delivery_time']
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)
    joblib.dump(model, 'route_optimization_model.joblib')
    logging.info("Route Optimization model saved successfully.")

if __name__ == "__main__":
    create_demand_forecast_model()
    create_customer_communication_model()
    create_route_optimization_model()
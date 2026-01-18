import os
import time
import logging
import sys
import pandas as pd
import paho.mqtt.client as mqtt
import requests
import json
import csv
import numpy as np
from dotenv import load_dotenv

# --- Configuration & Environment Variables ---
load_dotenv()

# MQTT Broker Configuration
MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS", "192.168.1.246")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "31883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "admin")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "password")

# MQTT Topic Configuration
MQTT_TOPIC_FEATURES = os.getenv("MQTT_TOPIC_FEATURES", "windguard/features")
MQTT_TOPIC_FAULT = os.getenv("MQTT_TOPIC_FAULT", "windguard/fault/prediction")
MQTT_TOPIC_MONITORING = os.getenv("MQTT_TOPIC_MONITORING", "windguard/monitoring")

# Prediction Service Configuration
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://127.0.0.1:5000/predict")

# Data Source Configuration
DATA_PATH = os.getenv("DATA_PATH", "iiot-combined-dataset.csv")

# Constants for Logic
NON_FEATURE_COLUMNS = [
    'Fault', 
    'DateTime_x', 
    'Time', 
    'Error', 
    'WEC: ava. windspeed', 
    'WEC: ava. available P from wind',
    'WEC: ava. available P technical reasons',
    'WEC: ava. Available P force majeure reasons',
    'WEC: ava. Available P force external reasons',
    'WEC: max. windspeed', 
    'WEC: min. windspeed', 
    'WEC: Operating Hours', 
    'WEC: Production kWh',
    'WEC: Production minutes', 
    'DateTime_y'
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- MQTT Functions ---

def connect_mqtt(broker_address, broker_port, username=None, password=None):
    """Connects to the MQTT broker."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="data_pipeline_dynamic_features")
    
    if username and password:
        logging.info("Setting MQTT username and password for authentication.")
        client.username_pw_set(username, password)

    try:
        logging.info(f"Connecting to MQTT broker at {broker_address}:{broker_port}...")
        client.connect(broker_address, broker_port, 60)
        client.loop_start()
        logging.info("Successfully connected to MQTT broker.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        return None

# --- Data Processing Functions ---

def get_model_prediction(api_url, api_payload):
    """
    Sends a data row to the KServe REST API and returns the fault label.
    """
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, data=json.dumps(api_payload), headers=headers)
        if response.status_code == 200:
            result = response.json()
            
            predictions = result.get('predictions')
            if isinstance(predictions, list) and len(predictions) > 0:
                return str(predictions[0])
            
            prediction = result.get('prediction') or result.get('label')
            return str(prediction) if prediction is not None else None
        else:
            logging.warning(f"Prediction API returned status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error calling prediction API: {e}")
        return None

def sanitize_topic(name):
    """
    Cleans up column headers to create valid MQTT topic segments.
    """
    return name.replace(':', '').replace(' ', '_').replace('.', '').strip().lower()

def process_data_file(client, file_path, api_url):
    """
    Reads a data CSV, dynamically identifies features from headers,
    publishes them to MQTT, and handles classification/monitoring.
    """
    logging.info(f"Processing data file: {file_path}")
    
    try:
        # Read headers first to determine features
        full_df_iterator = pd.read_csv(file_path, chunksize=1)
        
        for chunk in full_df_iterator:
            if chunk.empty:
                continue
                
            row = chunk.iloc[0]
            all_columns = chunk.columns.tolist()
            
            # --- 1. Publish individual features to MQTT ---
            # Dynamically identify features: any column NOT in the exclusion list
            for col in all_columns:
                if col not in NON_FEATURE_COLUMNS:
                    topic = f"{MQTT_TOPIC_FEATURES}/{sanitize_topic(col)}"
                    value = row[col]
                    client.publish(topic, payload=str(value), qos=1)

            # --- 2. Prepare Payload for Prediction ---
            # Drop excluded columns and non-numeric data for the ML model
            model_input_row = row.drop(labels=NON_FEATURE_COLUMNS, errors='ignore')
            #numeric_features = model_input_row.select_dtypes(include=[np.number])
            #numeric_features = numeric_features.fillna(0)
            
            # CRITICAL FIX: Cast numpy types (int64/float64) to standard Python floats 
            # to avoid JSON serialization errors (int64 is not JSON serializable).
            python_numeric_list = [float(x) for x in model_input_row.tolist()]
            
            model_payload = {
                "instances": [python_numeric_list]
            }

            label = get_model_prediction(api_url, model_payload)
            
            # --- 3. Publish Fault Classification ---
            if label is not None:
                client.publish(MQTT_TOPIC_FAULT, payload=label, qos=1)
                logging.info(f"Model Prediction: {label}")
                
                if 'Fault' in row:
                    logging.info(f"Actual Fault in CSV: {row['Fault']}")
            else:
                logging.warning("No valid label received from prediction service.")

            # --- 4. Publish Power Monitoring Data (Individual Topics) ---
            if 'WEC: ava. windspeed' in row:
                wind_topic = f"{MQTT_TOPIC_MONITORING}/windspeed"
                client.publish(wind_topic, payload=str(row['WEC: ava. windspeed']), qos=1)
            if 'WEC: Production kWh' in row:
                prod_topic = f"{MQTT_TOPIC_MONITORING}/production_kwh"
                client.publish(prod_topic, payload=str(row['WEC: Production kWh']), qos=1)

            # Optional: Add a small delay to simulate real-time stream and not overwhelm broker
            # in the real-world scenario there are at least 10 minutes between data points
            time.sleep(10)

    except Exception as e:
        logging.error(f"Error processing data file {file_path}: {e}")

# --- Main Execution ---

def main():
    """Main function to run the data pipeline."""
    # Connect to MQTT using configuration from the top of the file
    client = connect_mqtt(MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT, MQTT_USERNAME, MQTT_PASSWORD)
    
    if not client:
        logging.error("Failed to connect to MQTT. Exiting.")
        return

    try:
        # Check if the configured data path is a directory or a single file
        if os.path.isdir(DATA_PATH):
            data_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
            for data_file in sorted(data_files):
                process_data_file(client, os.path.join(DATA_PATH, data_file), SCORING_API_URL)
        else:
            process_data_file(client, DATA_PATH, SCORING_API_URL)
            
    except Exception as e:
        logging.error(f"An error occurred during file processing: {e}")
    finally:
        logging.info("Processing complete. Disconnecting.")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
import os
import sys
import logging
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import time
import requests
import json
import csv
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get config from environment variables
MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS", "192.168.1.135")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME","admin")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD","password")

# using local KServe custom predictor endpoint on Microshift
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://onnx-model-predictor/v1/models/onnx-model:predict")
FEATURES_FILE_PATH = os.getenv("FEATURES_FILE_PATH", "/home/luca/RH26/dataset/Wind_Farm_A/comma_feature_description.csv")
DATA_PATH = os.getenv("DATA_PATH", "/home/luca/RH26/dataset/Wind_Farm_A/datasets/comma_0.csv")  # Can be a file or a folder
CLIENT_ID = f"data_pipeline_{os.getpid()}"

# MQTT Topics
FEATURE_TOPIC_PREFIX = "features"
ANOMALY_TOPIC = "anomaly/score"

# Map from features.csv 'statistics_type' to data.csv suffix
STATS_ABBREVIATION_MAP = {
    "average": "avg",
    "maximum": "max",
    "minimum": "min",
    "std_dev": "std"
}

# --- Main Application ---

#def on_connect(client, userdata, flags, rc, properties=None):
#    """Callback for when the client connects to the MQTT broker."""
#    if rc == 0:
#        logging.info(f"Connected successfully to MQTT Broker at {MQTT_BROKER_ADDRESS}")
#    else:
#        logging.error(f"Failed to connect to MQTT broker, return code {rc}")

def connect_mqtt(broker_address, broker_port, username=None, password=None):
    """Connects to the MQTT broker."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    
    if username and password:
        logging.info(f"Setting MQTT username and password for authentication.")
        client.username_pw_set(username, password)
    else:
        logging.info("Connecting to MQTT without authentication.")

    try:
        logging.info(f"Connecting to MQTT broker at {broker_address}:{broker_port}...")
        client.connect(broker_address, broker_port, 60)
        client.loop_start()  # Start background thread for processing
        logging.info("Successfully connected to MQTT broker.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        return None

def load_features(filepath: str) -> list[tuple[str, str]]:
    """
    Loads feature map from the CSV file as described by the user.
    - Column 1: "base_feature" (e.g., 'sensor_name') - The key to find the value in the data CSV.
    - Column 2: "specifiers" (e.g., 'statistics_type') - A single value or comma-separated list
                  of values to be appended to the base_feature.
    
    Returns a list of tuples: [(base_feature, specifier), ...]
    Example: [('sensor_0', 'average'), ('wind_speed_3', 'maximum'), ('wind_speed_3', 'minimum'), ...]
    """
    feature_mappings = []
    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            # Use csv.reader to handle quotes properly (e.g., "max,min,avg")
            reader = csv.reader(f)
            
            # Skip header
            try:
                header = next(reader)
                logging.info(f"Features file header: {header}")
            except StopIteration:
                logging.warning(f"Features file is empty: {filepath}")
                return []

            # Process each feature row
            for row in reader:
                if not row or len(row) < 2:
                    logging.warning(f"Skipping malformed row in features file: {row}")
                    continue
                
                base_feature = row[0].strip()
                # Split the second column by comma
                specifiers = [s.strip() for s in row[1].split(',')]
                
                if not base_feature or not specifiers:
                    logging.warning(f"Skipping row with empty base_feature or specifiers: {row}")
                    continue
                
                # Create a mapping for each specifier
                for spec in specifiers:
                    if spec:
                        feature_mappings.append((base_feature, spec))
                        
        logging.info(f"Loaded {len(feature_mappings)} feature mappings. First 5: {feature_mappings[:5]}")
        return feature_mappings
        
    except FileNotFoundError:
        logging.error(f"Features file not found at: {filepath}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading features file: {e}. Exiting.")
        sys.exit(1)

def get_anomaly_score(endpoint_url: str, row_data: dict) -> float:
    """
    Sends a row of data to the scoring API and returns the anomaly score.
    
    Args:
        endpoint_url (str): The URL of the REST API endpoint.
        row_data (dict): The data row (as a dictionary) to be sent as JSON payload.

    Returns:
        float | None: The anomaly score if successful, None otherwise.
    """
    try:
        # Convert pandas/numpy types to native Python types for JSON serialization
        native_payload = {k: v.item() if hasattr(v, 'item') else v for k, v in row_data.items()}
        
        response = requests.post(endpoint_url, json={"instances":native_payload}, timeout=5)
        #response = requests.post(endpoint_url, json={"instances":str(row_data.values)}, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        score_data = response.json()
        if "score" in score_data:
            return float(score_data["score"])
        else:
            logging.warning(f"Scoring API response did not contain 'score' field: {score_data}")
            return None
            
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error calling scoring API: {http_err} - Response: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error calling scoring API at {endpoint_url}: {conn_err}")
    except requests.exceptions.Timeout:
        logging.error(f"Timeout calling scoring API at {endpoint_url}")
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Error decoding API response or casting score: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred calling scoring API: {e}")
        
    return None

def process_data_file(filepath: str, features: list[tuple[str, str]], client: mqtt.Client, score_url: str):
    """Reads a data CSV file row by row and processes it."""
    logging.info(f"Processing data file: {filepath}")
    
    try:
        # Use pandas chunking to read the file row by row without loading all into memory
        for chunk in pd.read_csv(filepath, chunksize=1):
            # Each 'chunk' is a DataFrame with 1 row
            if chunk.empty:
                continue
                
            row = chunk.iloc[0]
            row_dict = row.to_dict()
            
            # --- 1. Publish features to individual MQTT topics ---
            # 'features' is a list of (base_feature, specifier) tuples
            # (e.g., ('sensor_0', 'average'), ('wind_speed_3', 'maximum'), ...)
            for base_feature, specifier in features:
                
                # Get the suffix for the data file (e.g., 'average' -> 'avg')
                stat_suffix = STATS_ABBREVIATION_MAP.get(specifier)
                
                if not stat_suffix:
                    logging.warning(f"No stat abbreviation found for '{specifier}'. Skipping.")
                    continue
                
                # Construct the column name to find in the data file
                # (e.g., "sensor_0_avg", "wind_speed_3_max")
                column_name = f"{base_feature}_{stat_suffix}"
                
                if column_name in row or column_name[:-4] in row:  # Handle possible missing '_std' suffix
                    if column_name in row: 
                        value = row[column_name]
                    else:
                        value = row[column_name[:-4]]  # for missing cases
                                        
                    # Create the MQTT topic name using the *full* specifier
                    # (e.g., "sensor_0_average", "wind_speed_3_maximum")
                    topic_name = f"{base_feature}_{specifier}"
                    topic = f"{FEATURE_TOPIC_PREFIX}/{topic_name}"
                    
                    # Convert numpy types to standard Python types for MQTT
                    if hasattr(value, 'item'):
                        payload = value.item()
                    else:
                        payload = value
                    
                    # Publish to MQTT
                    id = row.get("id", "unknown")
                    result = client.publish(topic, str(payload))
                    logging.warning(f"Publishing data point {id} for sensor '{topic_name}' to topic {topic} with value: {payload}")
                    if result.rc != mqtt.MQTT_ERR_SUCCESS:
                        logging.warning(f"Failed to publish to topic {topic}")
                else:
                    logging.warning(f"Constructed column name '{column_name}' (from features.csv) not found in data row.")

            # --- 2. Get anomaly score from REST API ---
            numeric_features = chunk.select_dtypes(include=[np.number])
            model_payload = {
                "instances": [numeric_features.iloc[0].tolist()]
            }
            score = get_anomaly_score(score_url, model_payload)

            # --- 3. Publish anomaly score to MQTT ---
            if score is not None:
                # Add timestamp or row identifier to the payload if needed
                score_payload = json.dumps({
                    "timestamp": time.time(),
                    "score": score,
                    "source_file": os.path.basename(filepath)
                })
                result = client.publish(ANOMALY_TOPIC, score_payload)
                logging.warning(f"Fault scoring for {id} to anomaly topic {ANOMALY_TOPIC}")
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                     logging.warning(f"Failed to publish to anomaly topic {ANOMALY_TOPIC}")
            else:
                logging.warning(f"No score received for row. Skipping anomaly publish.")

            # Optional: Add a small delay to simulate real-time stream and not overwhelm broker
            time.sleep(10)

    except FileNotFoundError:
        logging.error(f"Data file not found: {filepath}")
    except pd.errors.EmptyDataError:
        logging.warning(f"Data file is empty: {filepath}")
    except Exception as e:
        logging.error(f"An error occurred processing data file {filepath}: {e}")

def main():
    """Main function to orchestrate the data pipeline."""
    logging.info("Starting data pipeline...")
    
    # 1. Load feature definitions
    features = load_features(FEATURES_FILE_PATH)
    if not features:
        logging.error("No features loaded. Check features.csv file. Exiting.")
        return

    # 2. Connect to MQTT Broker
    mqtt_client = connect_mqtt(MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT, MQTT_USERNAME, MQTT_PASSWORD)
    if not mqtt_client:
        logging.error("Failed to connect to MQTT. Exiting.")
        return


    # 3. Find and process data file(s)
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data path not found: {DATA_PATH}. Exiting.")
        return

    file_list = []
    if os.path.isdir(DATA_PATH):
        logging.info(f"Processing all CSV files in directory: {DATA_PATH}")
        for filename in os.listdir(DATA_PATH):
            if filename.lower().endswith(".csv"):
                file_list.append(os.path.join(DATA_PATH, filename))
    elif os.path.isfile(DATA_PATH) and DATA_PATH.lower().endswith(".csv"):
        logging.info(f"Processing single data file: {DATA_PATH}")
        file_list.append(DATA_PATH)
    else:
        logging.error(f"Data path {DATA_PATH} is not a valid CSV file or directory. Exiting.")
        return

    if not file_list:
        logging.warning(f"No CSV files found in {DATA_PATH}. Exiting.")
        return

    # Process each file
    for filepath in file_list:
        process_data_file(filepath, features, mqtt_client, SCORING_API_URL)

    logging.info("All data files processed. Shutting down.")
    
    # Clean up
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    logging.info("MQTT client disconnected. Pipeline finished.")

if __name__ == "__main__":
    main()



import os
import pandas as pd
import numpy as np
import onnxruntime as ort
import logging
import sys

# --- Configuration ---
DATA_FILE_PATH = os.path.join("data", "/home/luca/RH26/dataset/Wind_Farm_A/datasets/comma_0.csv")
ONNX_MODEL_PATH = "isolation_forest.onnx"
ID_COLUMNS = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def run_inference():
    """
    Loads the ONNX model and the sample data, runs inference,
    and prints the results.
    """

    # --- 1. Load ONNX Model ---
    logging.info(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        logging.info(f"Model loaded. Input: '{input_name}', Outputs: {output_names}")
    except FileNotFoundError:
        logging.error(f"Model file not found at: {ONNX_MODEL_PATH}")
        logging.error("Please run 'python train_model.py' first.")
        return
    except Exception as e:
        logging.error(f"Error loading ONNX model: {e}")
        return

    # --- 2. Load Data ---
    logging.info(f"Loading data from {DATA_FILE_PATH}...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {DATA_FILE_PATH}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # --- 3. Preprocess Data (Must match train_model.py) ---
    logging.info("Preprocessing data...")
    try:
        features_df = df.drop(columns=ID_COLUMNS, errors='ignore')
    except KeyError as e:
        logging.warning(f"Could not drop column: {e}. It might be missing.")
        features_df = df.copy()

    numeric_features_df = features_df.select_dtypes(include=['number'])
    
    if numeric_features_df.empty:
        logging.error("No numeric features found for inference.")
        return

    # Handle missing values
    if numeric_features_df.isnull().sum().any():
        logging.info("Handling missing values by filling with column median...")
        numeric_features_df = numeric_features_df.fillna(numeric_features_df.median())

    # --- 4. Run Inference ---
    try:
        # Convert data to float32 numpy array, as required by the ONNX model
        input_data = numeric_features_df.to_numpy().astype(np.float32)
        
        logging.info(f"Running inference on {input_data.shape[0]} rows...")
        
        # Run the model
        # The output is a list of [labels, scores]
        outputs = session.run(output_names, {input_name: input_data})
        
        # We are interested in the scores, which is the second output
        # (skl2onnx for IsolationForest returns [labels, scores])
        labels = outputs[0]
        anomaly_scores = outputs[1]
        
        logging.info("Inference complete.")
        
        # --- 5. Display Results ---
        logging.info("--- Anomaly Scores (first 10 rows) ---")
        for i in range(min(10, len(anomaly_scores))):
            # A negative score from Isolation Forest typically indicates an anomaly
            logging.info(f"Row {i}: Label = {labels[i]}, Score = {anomaly_scores[i]}")
            
        anomaly_count = (labels == -1).sum()
        logging.info("--- Summary ---")
        logging.info(f"Total rows processed: {len(labels)}")
        logging.info(f"Anomalies detected (label == -1): {anomaly_count}")

    except Exception as e:
        logging.error(f"Error during inference: {e}")

if __name__ == "__main__":
    run_inference()

import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
import sys
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- Configuration ---
DATA_FILE_PATH = os.path.join("data", "/home/luca/RH26/dataset/Wind_Farm_A/datasets/comma_3.csv")
MODEL_OUTPUT_PATH = "/home/luca/RH26/isolation_forest.onnx"
ID_COLUMNS = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def train_model():
    """
    Loads the sample data, trains an Isolation Forest model,
    and saves it to a file.
    """
    
    # --- 1. Load Data ---
    logging.info(f"Loading data from {DATA_FILE_PATH}...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {DATA_FILE_PATH}")
        logging.error("Please make sure 'data/sample_data.csv' exists.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # --- 2. Preprocess Data ---
    logging.info("Preprocessing data...")
    
    # Select only the feature columns for training
    # We drop ID, timestamp, and categorical columns
    try:
        features_df = df.drop(columns=ID_COLUMNS, errors='ignore')
    except KeyError as e:
        logging.warning(f"Could not drop column: {e}. It might be missing.")
        features_df = df.copy()

    # Ensure all remaining columns are numeric
    numeric_features_df = features_df.select_dtypes(include=['number'])
    
    if numeric_features_df.empty:
        logging.error("No numeric features found for training. Check your data file and ID_COLUMNS.")
        return
        
    logging.info(f"Training on {len(numeric_features_df.columns)} numeric features and {numeric_features_df.shape[0]} samples.")

    # Handle missing values (e.g., fill with median)
    # Isolation Forest doesn't support NaNs
    if numeric_features_df.isnull().sum().any():
        logging.info("Handling missing values by filling with column median...")
        numeric_features_df = numeric_features_df.fillna(numeric_features_df.median())
    
    # --- 3. Train Model ---
    logging.info("Training Isolation Forest model...")
    
    # 'contamination' is the expected proportion of anomalies.
    # 'auto' is a good default. You can also set it to a specific value, e.g., 0.05
    model = IsolationForest(
        n_estimators=100,
        contamination='auto',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(numeric_features_df)
    
    logging.info("Model training complete.")

    # --- 4. Save Model as ONNX ---
    logging.info("Converting model to ONNX format...")
    try:
        # Define the input type for the ONNX model
        # We need to tell it the shape (batch size, num_features) and type (float)
        n_features = numeric_features_df.shape[1]
        logging.info(f"ONNX model will have {n_features} input features.")
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert the model
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'':15,'ai.onnx.ml': 3})
        
        # Save the model
        with open(MODEL_OUTPUT_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        logging.info(f"Model successfully saved to {MODEL_OUTPUT_PATH}")
        
    except Exception as e:
        logging.error(f"Error saving model as ONNX: {e}")

if __name__ == "__main__":
    train_model()


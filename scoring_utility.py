from flask import Flask, request, jsonify
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def get_score():
    """
    A mock scoring endpoint.
    It receives data, logs it, and returns a random anomaly score.
    """
    try:
        data = request.json
        
        if data is None:
            logging.warning("Received request with no JSON payload.")
            return jsonify({"error": "No JSON payload provided"}), 400

        # Log the received data
        app.logger.info(f"Received data for scoring: {data}")

        # --- Mock Scoring Logic ---
        # In a real app, you would load a model and score the data.
        # Here, we just return a random "anomaly score".
        
        # Example: check for a specific feature
        temp = data.get("temperature", 70)
        pressure = data.get("pressure", 101)
        
        # Simple rule for "anomaly"
        if temp > 90 or pressure < 95:
            score = random.uniform(0.8, 1.0) # High anomaly score
        else:
            score = random.uniform(0.0, 0.3) # Low anomaly score
        
        score = round(score, 4)
        
        app.logger.info(f"Returning score: {score}")

        # Return the score in the expected format
        return jsonify({"score": score})

    except Exception as e:
        app.logger.error(f"Error in /score endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' makes it accessible from other containers/machines
    app.run(debug=True, host='0.0.0.0', port=5000)

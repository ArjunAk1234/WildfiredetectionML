import os
import sys
from flask import Flask, request, jsonify
import io
import tempfile
import joblib
import pandas as pd
import numpy as np

# Import your existing classes
from app import WildfireMLProject, WildfirePredictor

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            file.save(tmp_file.name)
            training_file_path = tmp_file.name
        
        # Run your existing training pipeline
        project = WildfireMLProject()
        project.run_full_project(training_file_path)
        
        # Clean up temp file
        os.unlink(training_file_path)
        
        return jsonify({"status": "success", "message": "Model trained successfully"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict wildfire risk for uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
            
        area_name = request.form.get('area_name', 'Unknown Area')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            file.save(tmp_file.name)
            prediction_file_path = tmp_file.name
        
        # Check if model exists
        if not os.path.exists("wildfire_pipeline.joblib"):
            return jsonify({"status": "error", "message": "Model not found. Please train the model first."}), 404
        
        # Use your existing predictor
        predictor = WildfirePredictor(model_path="wildfire_pipeline.joblib")
        
        if predictor.classification_model is None:
            return jsonify({"status": "error", "message": "Failed to load model"}), 500
        
        # Capture the prediction output
        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer
        
        try:
            result = predictor.predict_area_risk(prediction_file_path, area_name)
            prediction_output = output_buffer.getvalue()
        finally:
            sys.stdout = original_stdout
        
        # Clean up temp file
        os.unlink(prediction_file_path)
        
        return jsonify({
            "status": "success", 
            "area_name": area_name,
            "prediction_output": prediction_output,
            "result": result if isinstance(result, dict) else None
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Wildfire ML API is running"})

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Wildfire Detection ML API",
        "endpoints": {
            "train": "POST /train - Upload CSV to train model",
            "predict": "POST /predict - Upload CSV to get predictions",
            "health": "GET /health - Health check"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

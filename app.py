from flask import Flask, request, jsonify
from main import WildfireMLProject  # Import your class
import os

app = Flask(__name__)
project = WildfireMLProject()

@app.route('/train', methods=['POST'])
def train_model():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        csv_content = file.read().decode('utf-8')
        result = project.train_from_api(csv_content)
        return jsonify({'message': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        area_name = request.form.get('area_name', 'Uploaded Area')
        
        csv_content = file.read().decode('utf-8')
        result = project.predict_from_api(csv_content, area_name)
        return jsonify({'prediction': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

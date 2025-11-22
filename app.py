from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model
try:
    with open('predictive_maintenance_models.pkl', 'rb') as f:
        models = pickle.load(f)
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/seemore')
def seemore_page():
    return render_template('seemore.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/input')
def input_page():
    return render_template('input.html')


@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if models is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Get input data from request
        data = request.json
        
        # Extract input features
        input_data = {
            'temperature': float(data.get('temperature', 0)),
            'rotation_speed': float(data.get('rotation_speed', 0)),
            'torque': float(data.get('torque', 0)),
            'tool_wear': float(data.get('tool_wear', 0)),
            'power': float(data.get('power', 0)),
            'vibration': float(data.get('vibration', 0)),
            'pressure': float(data.get('pressure', 0)),
            'current': float(data.get('current', 0)),
            'voltage': float(data.get('voltage', 0)),
            'ambient_temp': float(data.get('ambient_temp', 0)),
            'humidity': float(data.get('humidity', 0)),
            'operational_hours': float(data.get('operational_hours', 0)),
            'cycles_completed': float(data.get('cycles_completed', 0)),
            'quality_score': float(data.get('quality_score', 0)),
            'efficiency': float(data.get('efficiency', 0))
        }
        
        # Calculate derived features
        input_data['temp_diff'] = input_data['temperature'] - input_data['ambient_temp']
        input_data['power_to_torque_ratio'] = input_data['power'] / (input_data['torque'] + 1)
        input_data['speed_torque_product'] = input_data['rotation_speed'] * input_data['torque']
        input_data['wear_rate'] = input_data['tool_wear'] / (input_data['operational_hours'] + 1)
        input_data['thermal_stress'] = input_data['temperature'] * input_data['vibration']
        input_data['electrical_stress'] = input_data['current'] * input_data['voltage']
        input_data['mechanical_stress'] = input_data['rotation_speed'] * input_data['torque'] * input_data['vibration']
        
        # Get models and scalers
        failure_model = models['failure_detection_model']
        type_model = models['failure_type_model']
        rul_model = models['rul_model']
        health_model = models['health_score_model']
        scaler_bin = models['scaler_binary']
        scaler_multi = models['scaler_multi']
        scaler_rul = models['scaler_rul']
        scaler_health = models['scaler_health']
        label_enc = models['label_encoder']
        feat_cols = models['feature_columns']
        health_feats = models['health_features']
        
        # Prepare input
        input_df = pd.DataFrame([input_data])
        input_scaled_bin = scaler_bin.transform(input_df[feat_cols])
        
        # Predict failure
        failure_pred = failure_model.predict(input_scaled_bin)[0]
        failure_proba = failure_model.predict_proba(input_scaled_bin)[0][1] if hasattr(failure_model, 'predict_proba') else failure_pred
        
        # Enhance features for RUL
        input_rul = input_df[feat_cols].copy()
        input_rul['tool_wear_squared'] = input_rul['tool_wear'] ** 2
        input_rul['temp_vibration'] = input_rul['temperature'] * input_rul['vibration']
        input_scaled_rul = scaler_rul.transform(input_rul)
        rul_pred = max(0, rul_model.predict(input_scaled_rul)[0])
        
        # Predict failure type if failure likely
        failure_type = "No Failure"
        if failure_proba > 0.5:
            input_scaled_multi = scaler_multi.transform(input_df[feat_cols])
            type_pred = type_model.predict(input_scaled_multi)[0]
            failure_type = label_enc.inverse_transform([type_pred])[0]
        
        # Predict health score
        input_scaled_health = scaler_health.transform(input_df[health_feats])
        health_score = np.clip(health_model.predict(input_scaled_health)[0], 0, 100)
        
        # Maintenance recommendations
        if rul_pred < 100:
            maintenance_window = "IMMEDIATE - Schedule maintenance within 24 hours"
            priority = "CRITICAL"
        elif rul_pred < 300:
            maintenance_window = "URGENT - Schedule maintenance within 1 week"
            priority = "HIGH"
        elif rul_pred < 500:
            maintenance_window = "SOON - Schedule maintenance within 2 weeks"
            priority = "MEDIUM"
        else:
            maintenance_window = "NORMAL - Schedule during next planned maintenance"
            priority = "LOW"
        
        preventive_cost = 500 * 83
        failure_cost = 5000 * 83
        downtime_cost = 2000 * 83
        
        if failure_proba > 0.5:
            potential_savings = failure_cost + downtime_cost - preventive_cost
        else:
            potential_savings = 0
        
        # Root cause analysis
        root_causes = []
        if input_data.get('temperature', 0) > 320:
            root_causes.append("High temperature detected")
        if input_data.get('vibration', 0) > 1.0:
            root_causes.append("Excessive vibration")
        if input_data.get('tool_wear', 0) > 180:
            root_causes.append("High tool wear")
        if input_data.get('operational_hours', 0) > 7000:
            root_causes.append("Extended operational hours")
        if input_data.get('current', 0) > 60:
            root_causes.append("High electrical current")
        if input_data.get('pressure', 0) > 110:
            root_causes.append("High system pressure")
        
        # Alerts
        alerts = []
        if failure_proba > 0.7:
            alerts.append(f"⚠️ FAILURE ALERT: {failure_type} predicted (confidence: {failure_proba:.1%})")
        if rul_pred < 100:
            alerts.append(f"🚨 CRITICAL: RUL is only {rul_pred:.0f} hours")
        if health_score < 40:
            alerts.append(f"⚠️ LOW HEALTH: Equipment health at {health_score:.1f}%")
        if input_data.get('temperature', 0) > 325:
            alerts.append("🌡️ HIGH TEMPERATURE WARNING")
        if input_data.get('vibration', 0) > 1.5:
            alerts.append("📳 EXCESSIVE VIBRATION ALERT")
        
        result = {
            'failure_predicted': bool(failure_pred),
            'failure_probability': float(failure_proba * 100),
            'failure_type': failure_type,
            'rul_hours': float(rul_pred),
            'rul_days': float(rul_pred / 24),
            'health_score': float(health_score),
            'maintenance_window': maintenance_window,
            'priority': priority,
            'potential_savings': float(potential_savings),
            'root_causes': root_causes if root_causes else ["No critical issues detected"],
            'alerts': alerts if alerts else ["✓ System operating normally"],
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
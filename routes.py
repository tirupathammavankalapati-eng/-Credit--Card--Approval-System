from flask import render_template, request, jsonify, session, redirect, url_for
from app import app, db
from app.models import PredictionLog
from app.ml_pipeline import predict_from_form, train_and_save_model
from app.utils import plot_confusion_matrix, get_recent_predictions
from app.config import Config
import uuid
import json

@app.route('/')
def index():
    """Render the main prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction form submission."""
    # Validate required fields
    required = ['gender', 'income_type', 'annual_income', 'employment_duration', 'education_level']
    for field in required:
        if field not in request.form or not request.form[field]:
            return render_template('index.html', error=f"Missing required field: {field}")

    try:
        form_data = {
            'gender': request.form['gender'],
            'income_type': request.form['income_type'],
            'annual_income': float(request.form['annual_income']),
            'employment_duration': int(request.form['employment_duration']),
            'education_level': request.form['education_level']
        }
    except ValueError as e:
        return render_template('index.html', error=f"Invalid numeric value: {str(e)}")

    # Get prediction
    try:
        pred, proba = predict_from_form(form_data, threshold=Config.PREDICTION_THRESHOLD)
    except FileNotFoundError:
        # Automatically train if model missing
        from app.ml_pipeline import generate_synthetic_data
        import os
        os.makedirs(os.path.dirname(Config.DATA_PATH), exist_ok=True)
        if not os.path.exists(Config.DATA_PATH):
            df = generate_synthetic_data()
            df.to_csv(Config.DATA_PATH, index=False)
        train_and_save_model(Config.DATA_PATH, Config.MODEL_PATH, retrain=True)
        pred, proba = predict_from_form(form_data, threshold=Config.PREDICTION_THRESHOLD)

    # Log prediction to database
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id

    log_entry = PredictionLog(
        session_id=session_id,
        input_features=json.dumps(form_data),
        prediction=pred,
        probability=proba
    )
    db.session.add(log_entry)
    db.session.commit()

    result_text = "Approved" if pred == 1 else "Rejected"
    return render_template('result.html',
                           prediction=result_text,
                           probability=round(proba * 100, 2),
                           details=form_data)

@app.route('/dashboard')
def dashboard():
    """Admin dashboard showing model metrics and history."""
    # Get all logs
    logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(100).all()
    log_data = [log.to_dict() for log in logs]

    # Generate confusion matrix if we have feedback data
    feedback_logs = PredictionLog.query.filter(PredictionLog.actual_feedback.isnot(None)).all()
    cm_img = None
    if feedback_logs:
        y_true = [log.actual_feedback for log in feedback_logs]
        y_pred = [log.prediction for log in feedback_logs]
        cm_img = plot_confusion_matrix(y_true, y_pred)

    # Model metadata
    from app.models import ModelMetadata
    meta = ModelMetadata.query.order_by(ModelMetadata.trained_date.desc()).first()

    return render_template('dashboard.html',
                           logs=log_data,
                           cm_img=cm_img,
                           meta=meta)

@app.route('/retrain', methods=['POST'])
def retrain():
    """Admin endpoint to trigger model retraining."""
    try:
        train_and_save_model(Config.DATA_PATH, Config.MODEL_PATH, retrain=True)
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint for Kubernetes."""
    return jsonify({'status': 'healthy'}), 200

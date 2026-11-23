from app import db
from datetime import datetime
import json

class PredictionLog(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False)
    input_features = db.Column(db.Text, nullable=False)  # JSON string
    prediction = db.Column(db.Integer, nullable=False)   # 1 = Approved, 0 = Rejected
    probability = db.Column(db.Float, nullable=False)
    actual_feedback = db.Column(db.Integer, nullable=True)  # For future retraining
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'input_features': json.loads(self.input_features),
            'prediction': 'Approved' if self.prediction == 1 else 'Rejected',
            'probability': round(self.probability, 4),
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

class ModelMetadata(db.Model):
    __tablename__ = 'model_metadata'
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(20), nullable=False)
    accuracy = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    trained_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_path = db.Column(db.String(200))

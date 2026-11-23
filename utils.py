import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering on server
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, labels=['Rejected', 'Approved']):
    """Generate a confusion matrix plot and return as base64 encoded image."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Save to bytes and encode
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

def get_recent_predictions(limit=50):
    """Fetch recent predictions from DB and return as DataFrame for plotting."""
    from app import db
    from app.models import PredictionLog
    logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    if not logs:
        return pd.DataFrame()
    data = {
        'timestamp': [log.timestamp for log in logs],
        'prediction': [log.prediction for log in logs],
        'probability': [log.probability for log in logs]
    }
    return pd.DataFrame(data)


import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')

# Create instance folder if it doesn't exist
os.makedirs(INSTANCE_DIR, exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'

    # Use the correct database location on your computer
    DATABASE_PATH = os.path.join(INSTANCE_DIR, 'approval.db')

    SQLALCHEMY_DATABASE_URI = (
        'sqlite:///' + DATABASE_PATH.replace('\\', '/')
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODEL_PATH = os.environ.get('MODEL_PATH') or 'models/model_pipeline.pkl'

    DATA_PATH = os.environ.get('DATA_PATH') or 'data/credit_card_dataset.csv'

    PREDICTION_THRESHOLD = float(os.environ.get('THRESHOLD', 0.5))

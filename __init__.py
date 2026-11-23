from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.config import Config

db = SQLAlchemy()
migrate = Migrate()

# Create the app instance globally
app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
migrate.init_app(app, db)

# Import routes and models (they will use the global `app` and `db`)
from app import routes
from app import models

# Create tables
with app.app_context():
    db.create_all()
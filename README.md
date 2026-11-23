# 💳 Credit Card Approval Prediction System

The Credit Card Approval System is a Flask ML app that automates underwriting. Built with XGBoost, Scikit-learn, and SQLite, it trains models, finds the best fit, and delivers real-time predictions via a Bootstrap UI. Features include an admin dashboard, REST API, and Docker.

---

## 🚀 Features

- **Instant Predictions** – Submit applicant details and get an approval decision with a confidence score in under 500ms.
- **Multiple Classifiers** – Compares Logistic Regression, Decision Tree, Random Forest, and XGBoost, automatically selects the best based on F1-score.
- **Hyperparameter Tuning** – Uses `GridSearchCV` to fine-tune the winning model for maximum performance.
- **Admin Dashboard** – View model metrics (accuracy, F1), confusion matrix, recent prediction logs, and trigger retraining with a single click.
- **Synthetic Data Generation** – No need to source a dataset; the system generates realistic synthetic records to get you started instantly.
- **Docker Ready** – Containerized deployment with volume mounts for persistent data and models.
- **REST API** – `/predict` endpoint for programmatic access, plus a health check endpoint.

---

## 🏗️ Architecture Overview

The application follows a **Model-View-Controller (MVC)** pattern, clearly separating the ML logic from the web interface.

```mermaid
flowchart TD
    subgraph Client_Layer
        Browser[Web Browser / Client]
    end

    subgraph Application_Layer [Flask Application]
        Router[Flask Router]
        Templates[Jinja2 Views]
        Preprocessor[Data Preprocessing Module]
        ModelEngine[Prediction Engine]
        DB_Connector[Database Connector]
    end

    subgraph Data_Layer
        SQLite[(SQLite Database)]
        FileSystem[File System (.pkl Models)]
    end

    Browser -->|HTTP Requests| Router
    Router --> Templates
    Router --> Preprocessor
    Preprocessor --> ModelEngine
    ModelEngine -->|Loads Serialized Pipeline| FileSystem
    ModelEngine -->|Returns Prediction| Router
    Router -->|Logs Activity| DB_Connector
    DB_Connector --> SQLite

```

---

## 🧰 Technology Stack

| Tier | Technology | Purpose |
| --- | --- | --- |
| **Front-End** | HTML5, CSS3, Bootstrap 5 | Responsive, accessible user interface |
| **Back-End** | Python 3.12, Flask 2.3 | Lightweight web framework for routing and business logic |
| **Data Processing** | Pandas, NumPy, Scikit-learn | Data cleaning, scaling, encoding, and pipeline creation |
| **Machine Learning** | XGBoost, Scikit-learn (Logistic Regression, Random Forest, Decision Tree) | Training, evaluation, and inference |
| **Model Serialization** | Joblib | Persisting trained pipelines to `.pkl` files |
| **Database** | SQLite (development), PostgreSQL (production ready) | Storing prediction logs and model metadata |
| **Containerization** | Docker | Reproducible environments and simplified deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment on push |

---

## 📁 Project Structure

```
credit-card-approval-system/
│
├── app/
│   ├── __init__.py        # Flask app initialization & global db instance
│   ├── config.py          # Configuration (reads from .env)
│   ├── models.py          # SQLAlchemy ORM models (PredictionLog, ModelMetadata)
│   ├── routes.py          # All Flask routes (/, /predict, /dashboard, /retrain, /health)
│   ├── ml_pipeline.py     # Core ML logic: preprocessing, training, inference
│   ├── utils.py           # Helper functions (confusion matrix plotting, etc.)
│   └── templates/         # HTML templates (base, index, result, dashboard)
│
├── scripts/
│   └── train_model.py     # Standalone script to train & save the model
│
├── data/                  # (auto-created) – synthetic dataset stored here
├── models/                # (auto-created) – serialized model pipeline (.pkl)
├── instance/              # (manual/auto-created) – SQLite database folder
│
├── .env                   # Environment variables (secret key, threshold)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── .dockerignore          # Files excluded from the Docker build
├── run.py                 # Entry point to start the Flask app
└── README.md              # This file

```

---

## ⚙️ Core Concepts Explained

### 1. Data Preprocessing Pipeline

We build a `ColumnTransformer` that applies different transformations to numeric and categorical features. This pipeline is **fitted on the training set** and then **saved alongside the model** to ensure the same transformations are applied during inference.

```python
# Snippet from ml_pipeline.py
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Annual_Income', 'Employment_Duration']),
        ('cat', categorical_transformer, ['Gender', 'Income_Type', 'Education_Level'])
    ])

```

### 2. Model Selection & Hyperparameter Tuning

The system trains four classifiers and evaluates them using cross-validation. The best-performing model (by F1-score) is then fine-tuned with `GridSearchCV` to achieve optimal performance.

```python
# Snippet from ml_pipeline.py
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}
# ... training and selection logic ...

```

### 3. Real-time Inference

When a user submits the form, the data is validated, transformed, and fed to the loaded pipeline. The pipeline returns both the binary prediction and the probability, which are displayed on the results page.

```python
# Snippet from routes.py
pred, proba = predict_from_form(form_data, threshold=Config.PREDICTION_THRESHOLD)

```

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.12** or later (recommended: use a virtual environment)
* **pip** and **virtualenv** (or Conda)
* **Docker** (optional, for containerized deployment)

### Option A: Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd credit-card-approval-system

```


2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

```


3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt

```


4. **Setup local workspace directories**
```bash
mkdir -p instance data models

```


5. **Train the model**
```bash
python scripts/train_model.py

```


6. **Run the application**
```bash
python run.py

```


7. **Open your browser** at `http://localhost:5000`

### Option B: Docker

1. **Build the Docker image**
```bash
docker build -t credit-approval-app .

```


2. **Run the container** (mount volumes to persist data, databases, and models)
```bash
docker run -p 5000:5000 \
  -v $(pwd)/instance:/app/instance \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  credit-approval-app

```



---

## ⚙️ Configuration (.env)

Create a `.env` file in the project root with the following variables:

```env
# Flask secret key (change to a secure random string)
SECRET_KEY=dev-secret-key-change-in-production

# Custom database override url (Optional: Defaults to absolute local SQLite instance path if left blank)
DATABASE_URL=

# Prediction threshold (0.5 = balanced)
THRESHOLD=0.5

```

To generate a secure `SECRET_KEY`, run:

```bash
python -c "import secrets; print(secrets.token_hex(32))"

```

---

## 🧪 Testing the System

### Web Interface

* **Home** – `/` : Submit an application.
* **Result** – After submission, the prediction and confidence score are displayed.
* **Dashboard** – `/dashboard` : View model metrics and recent logs, and trigger retraining.

### API Endpoints

```bash
curl -X POST http://localhost:5000/predict \
  -d "gender=Male&income_type=Salaried&annual_income=85000&employment_duration=10&education_level=Graduate"

```

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```

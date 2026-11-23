import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Global pipeline instance (cached for inference)
_pipeline = None

def get_feature_columns():
    """Define feature names as expected by the model."""
    numeric = ['Annual_Income', 'Employment_Duration']
    categorical = ['Gender', 'Income_Type', 'Education_Level']
    return numeric, categorical

def build_preprocessor():
    """Create a ColumnTransformer for numeric and categorical data."""
    numeric_features, categorical_features = get_feature_columns()

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
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def create_full_pipeline(classifier):
    """Return a complete pipeline with preprocessor and classifier."""
    return Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', classifier)
    ])

def train_and_save_model(data_path, model_save_path, retrain=False):
    """
    Train multiple classifiers, select the best based on F1-score, and save the pipeline.
    If retrain is True, force re-training regardless of existing model.
    """
    if not retrain and os.path.exists(model_save_path):
        print(f"[ML] Model already exists at {model_save_path}. Use retrain=True to force.")
        return load_pipeline(model_save_path)

    print("[ML] Loading dataset...")
    df = pd.read_csv(data_path)

    # Ensure columns exist; if not, generate synthetic data for demonstration
    required_cols = ['Gender', 'Income_Type', 'Annual_Income', 'Employment_Duration', 'Education_Level', 'Approved']
    if not all(col in df.columns for col in required_cols):
        print("[ML] Required columns missing. Generating synthetic dataset for demo...")
        df = generate_synthetic_data()
        df.to_csv(data_path, index=False)

    X = df[['Gender', 'Income_Type', 'Annual_Income', 'Employment_Duration', 'Education_Level']]
    y = df['Approved']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define classifiers to evaluate
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    best_f1 = -1
    best_clf_name = None
    best_pipeline = None
    results = {}

    print("[ML] Training and evaluating models...")
    for name, clf in classifiers.items():
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'f1_score': f1}
        print(f"  {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_clf_name = name
            best_pipeline = pipeline

    print(f"[ML] Best model: {best_clf_name} with F1={best_f1:.4f}")

    # Hyperparameter tuning on the best model (XGBoost example)
    if best_clf_name == 'XGBoost':
        print("[ML] Performing hyperparameter tuning on XGBoost...")
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.1, 0.3]
        }
        grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        print(f"  Best params: {grid_search.best_params_}")

    # Evaluate final pipeline on test set
    y_pred_final = best_pipeline.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final)
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"[ML] Final Model - Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Save the pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_pipeline, model_save_path)
    print(f"[ML] Pipeline saved to {model_save_path}")

    # Save metadata to DB (if database is available)
    try:
        from app import db
        from app.models import ModelMetadata
        meta = ModelMetadata(
            version='v1.0',
            accuracy=final_acc,
            f1_score=final_f1,
            file_path=model_save_path
        )
        db.session.add(meta)
        db.session.commit()
    except Exception as e:
        print(f"[ML] Could not save metadata to DB: {e}")

    return best_pipeline

def generate_synthetic_data(n_samples=5000):
    """Generate a realistic synthetic dataset for demonstration."""
    np.random.seed(42)
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Income_Type': np.random.choice(['Salaried', 'Self-Employed', 'Business', 'Unemployed'], n_samples, p=[0.5, 0.2, 0.2, 0.1]),
        'Annual_Income': np.random.randint(20000, 250000, n_samples),
        'Employment_Duration': np.random.randint(0, 40, n_samples),
        'Education_Level': np.random.choice(['High School', 'Graduate', 'Post-Graduate', 'Professional'], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
    }
    df = pd.DataFrame(data)

    # Heuristic to create target label (Approved)
    # Higher income, longer employment, and graduate education increase chances
    score = (
        (df['Annual_Income'] / 100000) * 0.4 +
        (df['Employment_Duration'] / 40) * 0.3 +
        (df['Education_Level'].map({'High School': 1, 'Graduate': 2, 'Post-Graduate': 3, 'Professional': 4}) / 4) * 0.3
    )
    prob = 1 / (1 + np.exp(-(score * 3 - 1.5)))  # sigmoid
    df['Approved'] = (np.random.random(n_samples) < prob).astype(int)
    return df

def load_pipeline(model_path):
    """Load the serialized pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
    return joblib.load(model_path)

def predict_from_form(form_data, model_path=None, threshold=0.5):
    """
    Accept a dictionary of form data, preprocess, and return prediction & probability.
    """
    global _pipeline
    if _pipeline is None:
        from app.config import Config
        _pipeline = load_pipeline(model_path or Config.MODEL_PATH)

    # Convert form data to DataFrame
    df = pd.DataFrame([{
        'Gender': form_data.get('gender'),
        'Income_Type': form_data.get('income_type'),
        'Annual_Income': float(form_data.get('annual_income')),
        'Employment_Duration': int(form_data.get('employment_duration')),
        'Education_Level': form_data.get('education_level')
    }])

    # Predict
    proba = _pipeline.predict_proba(df)[0][1]  # probability of class 1 (Approved)
    pred = 1 if proba >= threshold else 0

    return pred, proba

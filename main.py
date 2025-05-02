import logging
import pickle
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import torch

# Setup logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logger.info(f"Using device: {device}")


def load_data(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def preprocess_data(df, target):
    X, y = df.drop(columns=[target]), df[target]

    # Memory optimization: Convert float64 to float32
    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype('float32')

    # Handle categorical columns efficiently
    num_cols = X.select_dtypes(include=['int64', 'float32', 'float64', 'int32']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # For UNSW_NB15 dataset - limit categorical encoding
    # Only include categorical columns with reasonable cardinality
    filtered_cat_cols = []
    for col in cat_cols:
        if X[col].nunique() < 100:  # Only encode categories with fewer than 100 unique values
            filtered_cat_cols.append(col)
        else:
            logger.info(f"Skipping high-cardinality column: {col}")
            X = X.drop(columns=[col])

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), filtered_cat_cols)
    ], remainder='drop')

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Transform data
    X_tr_p = preprocessor.fit_transform(X_tr)
    X_te_p = preprocessor.transform(X_te)

    # Apply SMOTE using original approach without sampling
    logger.info(f"Applying SMOTE to training data of shape {X_tr_p.shape}")
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr_p, y_tr)
    logger.info(f"After SMOTE: shape {X_res.shape}")

    return X_res, X_te_p, y_res, y_te, preprocessor


def train_and_evaluate(model, X_tr, y_tr, X_te, y_te, name):
    logger.info(f"Training {name}")
    start = time.time()

    if name == 'XGBoost':
        logger.info(" - XGBoost on GPU" if cuda_available else " - XGBoost on CPU")
        # Fix: Use XGBoost with proper parameters
        eval_set = [(X_tr, y_tr), (X_te, y_te)]
        model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)

        # Force synchronize GPU (for XGBoost GPU mode)
        if cuda_available:
            torch.cuda.synchronize()  # Ensure GPU operations are complete
    else:
        model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    duration = time.time() - start

    report = classification_report(y_te, preds, output_dict=True)
    cm = confusion_matrix(y_te, preds)

    try:
        probs = model.predict_proba(X_te)
        auc = roc_auc_score(y_te, probs, multi_class='ovr')
    except Exception as e:
        logger.warning(f"Could not calculate AUC for {name}: {e}")
        auc = 'N/A'

    logger.info(f"{name} done in {duration:.2f}s, acc={report['accuracy']:.4f}, AUC={auc}")
    save_confusion_matrix(cm, name, np.unique(y_te))

    return {
        'Model': name,
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1': report['weighted avg']['f1-score'],
        'AUC': auc,
        'Training Time': duration
    }


def save_confusion_matrix(cm, name, classes):
    Path('visualizations').mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f"visualizations/{name}_cm.png")
    plt.close()


def save_model(model, name, preprocessor):
    Path('models').mkdir(exist_ok=True)

    try:
        # Handle XGBoost specially to ensure proper saving
        if name == 'XGBoost':
            logger.info(f"Saving XGBoost model using specialized approach")
            model.save_model(f"models/{name}.json")  # Save in XGBoost binary format
            with open(f"models/{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        else:
            with open(f"models/{name}.pkl", 'wb') as f:
                pickle.dump(model, f)

        # Save preprocessor separately
        with open(f"models/{name}_preprocessor.pkl", 'wb') as f:
            pickle.dump(preprocessor, f)

        logger.info(f"Saved {name} and its preprocessor")
    except Exception as e:
        logger.error(f"Error saving {name} model: {e}")
        # Fallback saving method
        try:
            if name == 'XGBoost':
                model.save_model(f"models/{name}_fallback.json")
            else:
                with open(f"models/{name}_fallback.pkl", 'wb') as f:
                    pickle.dump(model, f)
            logger.info(f"Saved {name} using fallback method")
        except Exception as e2:
            logger.error(f"Fallback save also failed for {name}: {e2}")


def save_results(res, mode='a'):
    df = pd.DataFrame([res])
    header = not Path('model_metrics.csv').exists() or mode == 'w'
    df.to_csv('model_metrics.csv', mode=mode, header=header, index=False)
    logger.info(f"Appended results for {res['Model']}")


def main(path, target):
    df = load_data(path)

    # UNSW_NB15 specific preprocessing
    logger.info("Applying UNSW_NB15 specific preprocessing")
    # Drop columns that might cause training issues or are not useful
    columns_to_drop = []
    if 'id' in df.columns:
        columns_to_drop.append('id')
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Process data
    X_tr, X_te, y_tr, y_te, pre = preprocess_data(df, target)

    # Handle labels
    le = LabelEncoder().fit(y_tr)
    y_tr_encoded, y_te_encoded = le.transform(y_tr), le.transform(y_te)
    num_classes = len(le.classes_)
    logger.info(f"Target classes: {num_classes}")

    # Configure models with optimized parameters for UNSW_NB15
    models = {
        'XGBoost': xgb.XGBClassifier(
            tree_method='hist',
            device='cuda:0' if cuda_available else 'cpu',
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=40,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=40,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=200,
            C=0.1,
            n_jobs=-1,
            random_state=42,
            solver='saga'  # Faster for large datasets
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=40,
            learning_rate=0.1,
            max_depth=4,
            n_iter_no_change=3,
            validation_fraction=0.1,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=40,
            learning_rate=0.1,
            random_state=42
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1,
            algorithm='kd_tree'  # More efficient for large datasets
        ),
        'NaiveBayes': GaussianNB()
    }

    # Start with empty results file
    save_results({'Model': '', 'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'AUC': 0, 'Training Time': 0},
                 mode='w')

    # Train and evaluate models
    for name, mdl in tqdm(models.items(), desc="Training"):
        try:
            res = train_and_evaluate(mdl, X_tr, y_tr_encoded, X_te, y_te_encoded, name)
            save_model(mdl, name, pre)
            save_results(res, mode='a')
        except Exception as e:
            logger.error(f"Error {name}: {e}")
            save_results(
                {'Model': name, 'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'AUC': 'N/A', 'Training Time': 0},
                mode='a')

    # Create comparison plot
    df_res = pd.read_csv('model_metrics.csv')
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Accuracy', data=df_res[df_res.Model != ''])
    plt.xticks(rotation=45)
    plt.title('Model Accuracy Comparison - UNSW_NB15')
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')

    # Also plot training time
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Training Time', data=df_res[df_res.Model != ''])
    plt.xticks(rotation=45)
    plt.title('Model Training Time Comparison - UNSW_NB15')
    plt.tight_layout()
    plt.savefig('visualizations/training_time_comparison.png')


if __name__ == "__main__":
    # Update this path to your UNSW_NB15 dataset location
    main('C:/Users/jagad/aiproj/data.csv','type')
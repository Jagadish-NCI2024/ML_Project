import logging
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Setup logging
logging.basicConfig(
    filename='dl_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()

# Check for GPU
try:
    physical_devices = keras.backend.tensorflow.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        keras.backend.tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info(f"Using GPU: {physical_devices}")
    else:
        logger.info("No GPU found, using CPU.")
except Exception as e:
    logger.warning(f"Error checking GPU: {e}")
    logger.info("Using CPU.")

def load_data(path):
    """Load the dataset from the specified path."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    return df

def preprocess_data(df, target):
    """Preprocess the data for deep learning."""
    logger.info("Preprocessing data for deep learning")
    
    # Separate features and target
    X, y = df.drop(columns=[target]), df[target]
    
    # Memory optimization: Convert float64 to float32
    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype('float32')
    
    # Identify column types
    num_cols = X.select_dtypes(include=['int64', 'float32', 'float64', 'int32']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Filter categorical columns with high cardinality
    filtered_cat_cols = []
    for col in cat_cols:
        if X[col].nunique() < 100:
            filtered_cat_cols.append(col)
        else:
            logger.info(f"Skipping high cardinality column: {col}")
            X = X.drop(columns=[col])
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), filtered_cat_cols)
    ], remainder='drop')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit and transform
    logger.info("Fitting preprocessor and transforming data")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names for deep learning
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # Get feature names for one-hot encoded columns
            ohe = transformer.named_steps['encoder']
            #feature_names.extend([f"{col}{cat}" for col in cols for cat in ohe.categories[list(cols).index(col)]])
            for col, cats in zip(cols, ohe.categories_):
                feature_names.extend([f"{col}_{cat}" for cat in cats])
            
         
    # Apply SMOTE for imbalanced classes
    logger.info(f"Applying SMOTE to training data of shape {X_train_transformed.shape}")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train_transformed, y_train)
    logger.info(f"After SMOTE: shape {X_resampled.shape}")
    
    # Encode target
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_resampled)
    y_test_encoded = label_encoder.transform(y_test)
    
    # One-hot encode for deep learning
    num_classes = len(label_encoder.classes_)
    logger.info(f"Target has {num_classes} classes")
    
    y_train_onehot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=num_classes)
    
    return {
        'X_train': X_resampled,
        'X_test': X_test_transformed,
        'y_train': y_train_onehot,
        'y_test': y_test_onehot,
        'y_train_encoded': y_train_encoded,
        'y_test_encoded': y_test_encoded,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'feature_names': feature_names,
        'input_shape': X_resampled.shape[1]
    }

def create_mlp_model(input_shape, num_classes, name="mlp"):
    """Create a simple MLP model with regularization."""
    model = Sequential(name=name)
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_deep_model(input_shape, num_classes, name="deep"):
    """Create a deeper neural network."""
    model = Sequential(name=name)
    
    # Input layer
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Hidden layers
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_residual_model(input_shape, num_classes, name="residual"):
    """Create a residual network model."""
    inputs = Input(shape=(input_shape,))
    
    # First block
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Residual block 1
    block_1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    block_1 = BatchNormalization()(block_1)
    block_1 = Dropout(0.3)(block_1)
    block_1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_1)
    block_1 = BatchNormalization()(block_1)
    x = concatenate([x, block_1])
    
    # Transition
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Residual block 2
    block_2 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    block_2 = BatchNormalization()(block_2)
    block_2 = Dropout(0.3)(block_2)
    block_2 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_2)
    block_2 = BatchNormalization()(block_2)
    x = concatenate([x, block_2])
    
    # Output
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_ensemble(input_shape, num_classes):
    """Create an ensemble of neural networks."""
    logger.info("Creating ensemble of neural networks")
    
    # Create base models
    models = {
        'mlp': create_mlp_model(input_shape, num_classes),
        'deep': create_deep_model(input_shape, num_classes),
        'residual': create_residual_model(input_shape, num_classes)
    }
    
    return models

def train_model(model, data, name, epochs=20, batch_size=256):
    """Train a single model."""
    logger.info(f"Training {name} model")
    start_time = time.time()
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    # Train model
    history = model.fit(
        data['X_train'], data['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(data['X_test'], data['y_test'], verbose=0)
    
    # Calculate predictions
    y_pred_proba = model.predict(data['X_test'])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Compute metrics
    cm = confusion_matrix(data['y_test_encoded'], y_pred)
    report = classification_report(data['y_test_encoded'], y_pred, output_dict=True)
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Save confusion matrix visualization
    save_confusion_matrix(cm, name, np.unique(data['y_test_encoded']))
    
    # Save learning curves
    save_learning_curves(history, name)
    
    # Calculate AUC
    try:
        auc = roc_auc_score(data['y_test'], y_pred_proba, multi_class='ovr')
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = 'N/A'
    
    # Log performance
    logger.info(f"{name} model trained in {train_time:.2f}s")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"AUC: {auc}")
    
    return {
        'model': model,
        'history': history,
        'metrics': {
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1': report['weighted avg']['f1-score'],
            'AUC': auc,
            'Training Time': train_time
        }
    }

def ensemble_predict(models, X):
    """Make ensemble predictions."""
    predictions = []
    for name, model in models.items():
        preds = model.predict(X)
        predictions.append(preds)
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

def save_learning_curves(history, name):
    """Save learning curves for model training."""
    Path('visualizations').mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{name}_learning_curves.png')
    plt.close()

def save_confusion_matrix(cm, name, classes):
    """Save confusion matrix visualization."""
    Path('visualizations').mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'visualizations/{name}_confusion_matrix.png')
    plt.close()

def save_model(model, name, preprocessor=None, label_encoder=None):
    """Save model and associated preprocessing components."""
    Path('models').mkdir(exist_ok=True)
    
    # Save Keras model
    model.save(f'models/{name}_model.h5')
    
    # Save preprocessor if provided
    if preprocessor is not None:
        with open(f'models/{name}_preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    
    # Save label encoder if provided
    if label_encoder is not None:
        with open(f'models/{name}_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
    
    logger.info(f"Saved {name} model and its components")

def save_results(results, filename='dl_model_metrics.csv'):
    """Save model performance metrics."""
    df = pd.DataFrame([results])
    
    # Check if file exists
    if Path(filename).exists():
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    
    logger.info(f"Saved metrics for {results['Model']} to {filename}")

def main(data_path, target_column, epochs=20, batch_size=256):
    """Main function to train and evaluate the ensemble."""
    logger.info("Starting deep learning ensemble training")
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess data
    data = preprocess_data(df, target_column)
    
    # Create ensemble models
    models = create_ensemble(data['input_shape'], data['num_classes'])
    
    # Train individual models
    results = []
    trained_models = {}
    
    for name, model in models.items():
        try:
            result = train_model(model, data, name, epochs, batch_size)
            trained_models[name] = result['model']
            results.append(result['metrics'])
            save_model(model, name, data['preprocessor'], data['label_encoder'])
            save_results(result['metrics'])
        except Exception as e:
            logger.error(f"Error training {name} model: {e}")
    
    # Ensemble prediction
    logger.info("Making ensemble predictions")
    y_pred_proba = ensemble_predict(trained_models, data['X_test'])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Evaluate ensemble
    cm = confusion_matrix(data['y_test_encoded'], y_pred)
    report = classification_report(data['y_test_encoded'], y_pred, output_dict=True)
    
    try:
        auc = roc_auc_score(data['y_test'], y_pred_proba, multi_class='ovr')
    except Exception as e:
        logger.warning(f"Could not calculate AUC for ensemble: {e}")
        auc = 'N/A'
    
    # Log ensemble performance
    logger.info(f"Ensemble Accuracy: {report['accuracy']:.4f}")
    logger.info(f"Ensemble AUC: {auc}")
    
    # Save ensemble metrics
    ensemble_metrics = {
        'Model': 'DL_Ensemble',
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1': report['weighted avg']['f1-score'],
        'AUC': auc,
        'Training Time': sum(r['Training Time'] for r in results)
    }
    save_results(ensemble_metrics)
    
    # Save ensemble confusion matrix
    save_confusion_matrix(cm, 'DL_Ensemble', np.unique(data['y_test_encoded']))
    
    # Create comparison plot
    models_df = pd.read_csv('dl_model_metrics.csv')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=models_df)
    plt.title('Deep Learning Models Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/dl_models_comparison.png')
    
    # Also plot F1 scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='F1', data=models_df)
    plt.title('Deep Learning Models F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/dl_models_f1_comparison.png')
    
    logger.info("Deep learning ensemble training complete")
    
    return {
        'ensemble_metrics': ensemble_metrics,
        'models': trained_models,
        'preprocessor': data['preprocessor'],
        'label_encoder': data['label_encoder']
    }

if __name__ == "__main__":
    # Set path to your UNSW_NB15 dataset
    DATA_PATH = "C:/Users/jagad/aiproj/data.csv"
    TARGET_COLUMN = "type"
    
    # Train ensemble
    main(DATA_PATH, TARGET_COLUMN, epochs=20,batch_size=256)

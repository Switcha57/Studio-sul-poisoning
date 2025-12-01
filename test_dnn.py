#!/usr/bin/env python3
"""
Testing script for DNN malware classifier.
Loads a trained model and evaluates it on test dataset.
"""

from keras.layers import BatchNormalization, Dense
from keras import Sequential
import os
import sys
import configparser
import numpy as np
import pickle
from pathlib import Path

# Add librerie to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'librerie'))

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_config(config_path='config.ini'):
    """Load configuration from config.ini file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config.read(config_path)
    return config


def load_ember_test_dataset(data_dir):
    """Load EMBER test dataset using the ember library."""
    try:
        import ember
    except ImportError:
        raise ImportError("ember library not found. Please install it: pip install ember")
    
    print(f"Loading EMBER test dataset from {data_dir}...")
    
    # Load test data
    X_test, y_test = ember.read_vectorized_features(data_dir, subset='test')
    
    # Remove samples with unknown labels (-1)
    valid_indices = y_test != -1
    X_test = X_test[valid_indices]
    y_test = y_test[valid_indices]
    
    print(f"Loaded {len(X_test)} test samples")
    print(f"Benign samples: {np.sum(y_test == 0)}")
    print(f"Malicious samples: {np.sum(y_test == 1)}")
    
    return X_test, y_test


def load_dataset_from_pickle(ds, test_size=0.2, val_size=0.1):
    """
    Loads and processes a dataset from a pickle file.

    Args:
        dataset_dir (str): The path to the dataset directory containing 'dataset_lief.pickle'.
        test_size (float): The proportion of the dataset to allocate to the test split.
        val_size (float): The proportion of the remaining dataset to allocate to the validation split.

    Returns:
        A tuple containing:
        - x_train, x_val, x_test: Feature datasets for training, validation, and testing.
        - y_train, y_val, y_test: Label datasets for training, validation, and testing.
        - x_train_meta, x_val_meta, x_test_meta: Metadata for each split.
    """

    def dataset_to_features(ds, test_size, val_size):
        x_meta = []
        y_labels = []

        for pe in ds:
            x_meta.append(pe)
            y_labels.append(0 if pe["list"] == "Whitelist" else 1)

        # Split into training and test sets
        x_train_meta, x_test_meta, y_train, y_test = train_test_split(
            x_meta, y_labels, test_size=test_size, random_state=42, stratify=y_labels
        )

        # Split training into training and validation sets
        x_train_meta, x_val_meta, y_train, y_val = train_test_split(
            x_train_meta, y_train, test_size=val_size / (1 - test_size), random_state=42, stratify=y_train
        )

        x_train = np.array([x["features"] for x in x_train_meta])
        x_val = np.array([x["features"] for x in x_val_meta])
        x_test = np.array([x["features"] for x in x_test_meta])

        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        return x_train, x_val, x_test, y_train, y_val, y_test, x_train_meta, x_val_meta, x_test_meta

    return dataset_to_features(ds, test_size=test_size, val_size=val_size)

def load_custom_test_dataset(pickle_path):
    """Load custom test dataset from pickle file."""
    print(f"Loading custom test dataset from {pickle_path}...")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Test dataset file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    x_train, x_val, x_test, y_train, y_val, y_test, _, _, _ = load_dataset_from_pickle(
        dataset, test_size=0.2, val_size=0.1
    )
    X_test = np.concatenate((x_train, x_test,x_val))
    y_test = np.concatenate((y_train, y_test, y_val))
    print(f"Loaded {len(X_test)} test samples")
    print(f"Benign samples: {np.sum(y_test == 0)}")
    print(f"Malicious samples: {np.sum(y_test == 1)}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print detailed metrics."""
    print("\n" + "="*50)
    print("Evaluating model...")
    print("="*50 + "\n")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Overall metrics (for malicious class - positive class)
    accuracy = accuracy_score(y_test, y_pred)
    precision_malicious = precision_score(y_test, y_pred, zero_division=0)
    recall_malicious = recall_score(y_test, y_pred, zero_division=0)
    f1_malicious = f1_score(y_test, y_pred, zero_division=0)

    # Goodware (Benign) metrics (class 0)
    precision_benign = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_benign = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_benign = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    # Print results
    print("Confusion Matrix Values:")
    print("-" * 50)
    print(f"True Negative:  {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive:  {tp}")
    
    print("\nOverall Metrics:")
    print("-" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Precision:        {precision_malicious:.4f}")
    print(f"Recall:           {recall_malicious:.4f}")
    print(f"F1 Score:         {f1_malicious:.4f}")
    
    print("\nGoodware (Benign) Metrics:")
    print("-" * 50)
    print(f"Precision goodware (Benign): {precision_benign:.4f}")
    print(f"Recall goodware (Benign):    {recall_benign:.4f}")
    print(f"F1 goodware (Benign):        {f1_benign:.4f}")
    
    print("\n" + "="*50)
    
    return {
        'accuracy': accuracy,
        'precision': precision_malicious,
        'recall': recall_malicious,
        'f1': f1_malicious,
        'precision_benign': precision_benign,
        'recall_benign': recall_benign,
        'f1_benign': f1_benign,
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def build_dnn(config):
    """Build DNN model based on configuration."""
    model_config = config['MODEL']
    training_config = config['TRAINING']

    input_shape = int(model_config.get('input_shape', 2381))
    layer1_units = int(model_config.get('layer1_units', 512))
    layer2_units = int(model_config.get('layer2_units', 128))
    layer3_units = int(model_config.get('layer3_units', 8))
    output_units = int(model_config.get('output_units', 2))
    activation = model_config.get('activation', 'tanh')

    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_shape,)))
    model.add(Dense(layer1_units, activation=activation,
              kernel_initializer='glorot_uniform'))
    model.add(Dense(layer2_units, activation=activation,
              kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(layer3_units, activation=activation,
              kernel_initializer='glorot_uniform'))
    model.add(Dense(output_units, activation='softmax',
              kernel_initializer='glorot_uniform'))

    optimizer = training_config.get('optimizer', 'adam')

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.losses.SparseCategoricalCrossentropy(name="loss"),
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    print("\nModel architecture:")
    model.summary()

    return model
def test_model(config):
    """Main testing function."""
    dataset_config = config['DATASET']
    model_config = config['MODEL']
    testing_config = config['TESTING']
    
    # Load model
    model_path = model_config.get('model_test_path', 'models/dnn_model.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = build_dnn(config)
    model.load_weights(model_path)
    print("✓ Model loaded successfully")
    
    # Load test dataset
    dataset_type = dataset_config.get('dataset_type', 'ember').lower()
    
    if dataset_type == 'ember':
        data_dir = dataset_config.get('ember_data_dir', 'cleanDataset/ember2018')
        X_test, y_test = load_ember_test_dataset(data_dir)
    elif dataset_type == 'custom':
        # Check if custom test dataset is specified, otherwise use training dataset
        test_path = testing_config.get('custom_test_dataset_path')
        if not test_path or not os.path.exists(test_path):
            print("Warning: Custom test dataset not found, using training dataset for testing")
            test_path = dataset_config.get('custom_dataset_path')
        X_test, y_test = load_custom_test_dataset(test_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'ember' or 'custom'")
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    return model, results


def main():
    """Main entry point."""
    config_path = 'config.ini'
    
    print("="*50)
    print("DNN Malware Classifier - Testing Script")
    print("="*50 + "\n")
    
    try:
        config = load_config(config_path)
        model, results = test_model(config)
        print("\n✓ Testing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

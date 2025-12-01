#!/usr/bin/env python3
"""
Training script for DNN malware classifier.
Loads dataset from config.ini and trains a model, saving weights in .h5 format.
"""

import os
import sys
import configparser
import numpy as np
import pickle
from pathlib import Path

# Add librerie to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'librerie'))

import tensorflow as tf
from keras import Sequential
from keras.layers import BatchNormalization, Dense
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_config(config_path='config.ini'):
    """Load configuration from config.ini file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config.read(config_path)
    return config


def load_ember_dataset(data_dir, validation_split=0.2):
    """Load EMBER dataset using the ember library."""
    try:
        import ember
    except ImportError:
        raise ImportError("ember library not found. Please install it: pip install ember")
    
    print(f"Loading EMBER dataset from {data_dir}...")
    
    # Load training data
    X_train, y_train = ember.read_vectorized_features(data_dir, subset='train')
    
    # Remove samples with unknown labels (-1)
    valid_indices = y_train != -1
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]
    
    print(f"Loaded {len(X_train)} training samples")
    print(f"Benign samples: {np.sum(y_train == 0)}")
    print(f"Malicious samples: {np.sum(y_train == 1)}")
    
    # Split into train and validation
    if validation_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_split, 
            random_state=42,
            stratify=y_train
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        return X_train, y_train, X_val, y_val
    else:
        return X_train, y_train, None, None


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

def load_custom_dataset(pickle_path, validation_split=0.2):
    """Load custom dataset from pickle file."""
    print(f"Loading custom dataset from {pickle_path}...")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Dataset file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    x_train, x_val, x_test, y_train, y_val, y_test, _, _, _ = load_dataset_from_pickle(
        dataset, test_size=0.2, val_size=validation_split
    )
    x_train=np.concatenate((x_train, x_test), axis=0)
    y_train=np.concatenate((y_train, y_test), axis=0)
    return x_train, y_train, x_val, y_val

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
    model.add(Dense(layer1_units, activation=activation, kernel_initializer='glorot_uniform'))
    model.add(Dense(layer2_units, activation=activation, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(layer3_units, activation=activation, kernel_initializer='glorot_uniform'))
    model.add(Dense(output_units, activation='softmax', kernel_initializer='glorot_uniform'))
    
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


def train_model(config):
    """Main training function."""
    dataset_config = config['DATASET']
    training_config = config['TRAINING']
    model_config = config['MODEL']
    
    # Load dataset
    dataset_type = dataset_config.get('dataset_type', 'ember').lower()
    validation_split = float(dataset_config.get('validation_split', 0.2))
    
    if dataset_type == 'ember':
        data_dir = dataset_config.get('ember_data_dir', 'cleanDataset/ember2018')
        X_train, y_train, X_val, y_val = load_ember_dataset(data_dir, validation_split)
    elif dataset_type == 'custom':
        dataset_path = dataset_config.get('custom_dataset_path')
        X_train, y_train, X_val, y_val = load_custom_dataset(dataset_path, validation_split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'ember' or 'custom'")
    
    # Build model
    model = build_dnn(config)
    
    # Compute class weights to handle imbalance
    # Ensure y_train is 1D integer labels (0/1)
    # print(len(y_train)+len(y_val))
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Class distribution: {np.bincount(y_train)}")
    print(f"Using class_weight: {class_weight}")
    # quit()
    # Training parameters
    epochs = int(training_config.get('epochs', 2000))
    batch_size = int(training_config.get('batch_size', 256))
    patience = int(training_config.get('early_stopping_patience', 10))
    monitor = training_config.get('early_stopping_monitor', 'accuracy')
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    ]
    
    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    validation_data = (X_val, y_val) if X_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        # callbacks=callbacks,
        class_weight=class_weight,
        verbose=1

    )
    
    # Save model
    model_save_path = model_config.get('model_save_path', 'models/dnn_model.h5')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model.save(model_save_path)
    print(f"\n✓ Model saved to: {model_save_path}")
    
    # Print final metrics
    if X_val is not None:
        print("\n" + "="*50)
        print("Final validation metrics:")
        print("="*50)
        val_loss, _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return model, history


def main():
    """Main entry point."""
    config_path = 'config.ini'
    
    print("="*50)
    print("DNN Malware Classifier - Training Script")
    print("="*50 + "\n")
    
    try:
        config = load_config(config_path)
        model, history = train_model(config)
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

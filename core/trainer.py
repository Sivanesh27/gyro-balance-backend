# -*- coding: utf-8 -*-
"""
GyroBalance AI Engine - trainer.py
Handles physics extraction, synthetic data generation, and isolated model training.
"""

import os
import trimesh
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# Sensor positions (Must match the hardware setup)
SENSOR_POS = [9.0, 26.0, 42.0] 

def get_stl_physics(file_path, density=1.25):
    """
    Extracts the Ideal Center of Gravity and Mass from an STL file.
    """
    try:
        mesh = trimesh.load(file_path)
        scale = 0.1 # mm to cm conversion
        
        # Using centroid for high-precision stability
        ideal_cg = float(mesh.centroid[0]) * scale
        volume_cm3 = float(mesh.volume) * (scale**3)
        ideal_mass = (volume_cm3 * density) / 1000 # returns in kg
        
        return ideal_cg, ideal_mass
    except Exception as e:
        print(f"Error in STL Physics Extraction: {e}")
        return None, None

def generate_custom_dataset(target_cg, target_mass, n=1000000):
    """
    Generates 1 million synthetic load signatures based on a specific blade DNA.
    """
    print(f"🔄 Generating {n} unique defect signatures for custom DNA...")
    data = []
    S = SENSOR_POS

    # Defect types: 0:Healthy, 1:Resin, 2:Void, 3:Infill
    for _ in range(n):
        defect = np.random.choice([0, 1, 2, 3])
        cg_err, mass_err = 0, 0

        if defect == 1: # RESIN POOL: Extra weight at tip
            cg_err, mass_err = np.random.uniform(3.0, 6.0), np.random.uniform(0.10, 0.25)
        elif defect == 2: # AIR VOID: Less weight at root
            cg_err, mass_err = np.random.uniform(-5.0, -2.0), np.random.uniform(-0.15, -0.05)
        elif defect == 3: # INFILL ERROR: Weight shift, No mass change
            cg_err, mass_err = np.random.uniform(1.0, 4.0), np.random.uniform(-0.01, 0.01)

        actual_cg = target_cg + cg_err
        actual_mass = target_mass + mass_err

        # Statics Math for 3-Load Cell distribution
        w3 = actual_mass * (actual_cg - S[0]) / (S[2] - S[0])
        w1 = actual_mass * (S[2] - actual_cg) / (S[2] - S[0])
        w2 = max(0, actual_mass - w1 - w3)

        # Add realistic sensor noise (0.001kg / 1g deviation)
        w1 += np.random.normal(0, 0.001)
        w2 += np.random.normal(0, 0.001)
        w3 += np.random.normal(0, 0.001)
        
        # 2.0 indicates a custom blade type in the input layer
        data.append([2.0, w1, w2, w3, actual_cg, actual_mass, defect])

    return pd.DataFrame(data, columns=['type', 'w1', 'w2', 'w3', 'cg', 'mass', 'defect'])

def train_new_blade_model(target_cg, target_mass, save_path):
    """
    Trains a multi-output neural network for a specific blade and saves it.
    """
    # 1. Data Preparation
    df = generate_custom_dataset(target_cg, target_mass)
    X = df[['type', 'w1', 'w2', 'w3']].values
    y_meas = df[['cg', 'mass']].values
    y_diag = tf.keras.utils.to_categorical(df['defect'], num_classes=4)

    # 2. Model Architecture (Matches your master model for consistency)
    inputs = layers.Input(shape=(4,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)

    out_meas = layers.Dense(2, name='measurement')(x)
    out_diag = layers.Dense(4, activation='softmax', name='diagnosis')(x)

    model = models.Model(inputs=inputs, outputs=[out_meas, out_diag])
    model.compile(
        optimizer='adam',
        loss={'measurement': 'mse', 'diagnosis': 'categorical_crossentropy'},
        metrics={'measurement': 'mae', 'diagnosis': 'accuracy'}
    )

    # 3. Training Loop
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)
    
    print(f"🚀 Training specialized model for target CG: {target_cg}...")
    # Using 15 epochs for custom blades to balance speed and accuracy on Render
    model.fit(
        X, 
        {'measurement': y_meas, 'diagnosis': y_diag}, 
        epochs=15, 
        batch_size=2048, 
        callbacks=[lr_callback],
        verbose=0 # Keep logs clean on Render
    )

    # 4. Save to isolated folder
    model_file = os.path.join(save_path, "model.keras")
    model.save(model_file)
    print(f"✅ Specialized model saved to: {model_file}")
    
    return model_file
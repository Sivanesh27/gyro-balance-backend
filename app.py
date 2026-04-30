# -*- coding: utf-8 -*-
"""
GyroBalance Cloud Backend - app.py (Final Merged Version)
Features: Keras 3 Compatibility, 60-Sample Averaging, Enhanced Neural Logic
Updated Sensor Positions: [5.0, 22.5, 40.0]
"""

import os
import uuid
import joblib
import numpy as np
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS

# AI & Physics Imports
import keras
import tensorflow as tf
import trimesh

# PDF Report Import
from fpdf import FPDF

# ─────────────────────────────────────────────
# APP SETUP & CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(BASE_DIR, "custom_blades")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
for folder in [CUSTOM_DIR, MODELS_DIR, STATIC_DIR]:
    os.makedirs(folder, exist_ok=True)

# Hardware Config
SENSOR_POS = [5.0, 22.5, 40.0]
BALANCE_TOLERANCE = 0.10

# ─────────────────────────────────────────────
# CORE AI LOGIC (KERAS 3 COMPATIBLE)
# ─────────────────────────────────────────────
model_cache = {}

def get_model_and_meta(blade_id):
    if len(model_cache) > 3:
        model_cache.clear()
        keras.backend.clear_session()

    if blade_id in model_cache:
        return model_cache[blade_id]

    try:
        if str(blade_id) == "0":
            model_path = os.path.join(MODELS_DIR, "gyro_master_model.keras")
            meta_path = os.path.join(MODELS_DIR, "blade_meta.pkl")
            meta = joblib.load(meta_path)['B1']
        elif str(blade_id) == "1":
            model_path = os.path.join(MODELS_DIR, "gyro_master_model.keras")
            meta_path = os.path.join(MODELS_DIR, "blade_meta.pkl")
            meta = joblib.load(meta_path)['B2']
        else:
            path = os.path.join(CUSTOM_DIR, str(blade_id))
            model_path = os.path.join(path, "model.keras")
            meta_path = os.path.join(path, "meta.pkl")
            meta_data = joblib.load(meta_path)
            meta = meta_data[blade_id]

        # Use standalone keras loader to fix quantization_config error
        model = keras.models.load_model(model_path, compile=False)
        model_cache[blade_id] = (model, meta)
        return model, meta
    except Exception as e:
        print(f"❌ AI Loader Failure for {blade_id}: {str(e)}")
        return None, None

def perform_ai_analysis(w1_g, w2_g, w3_g, blade_id) -> dict:
    model, meta = get_model_and_meta(blade_id)
    if not model:
        return {"error": "AI Engine Offline"}

    target_cg, target_mass = meta
    total_g = max(1e-9, w1_g + w2_g + w3_g)
    total_mass_kg = total_g / 1000.0

    # 1. AI Inference
    input_type = float(blade_id) if str(blade_id).isdigit() else 2.0
    X = np.array([[input_type, w1_g/1000.0, w2_g/1000.0, w3_g/1000.0]], dtype=np.float32)
    preds = model.predict(X, verbose=0)
    
    predicted_cg = float(preds[0][0][0])
    defect_idx = int(np.argmax(preds[1][0]))

    # 2. Enhanced Neural Interpretation Dictionary (From your preferred code)
    defect_map = {
        0: {
            "name": "HEALTHY DNA",
            "desc": "The neural inference engine has validated internal structural load parity. Mass distribution matches design intent.",
            "sig": "GEOMETRIC DNA CONFIRMED."
        },
        1: {
            "name": "RESIN ACCUMULATION",
            "desc": "Neural signature detects localized excess density at distal coordinates. Likely resin pooling in tip cavities.",
            "sig": "ANOMALOUS MASS DETECTED IN TIP ZONE."
        },
        2: {
            "name": "INTERNAL AIR VOID",
            "desc": "Critical density dropout detected in core span. Neural patterns correlate with internal voiding.",
            "sig": "STRUCTURAL INTEGRITY BREACH."
        },
        3: {
            "name": "INFILL ERROR",
            "desc": "Neural scan indicates non-linear mass distribution. Internal lattice deviates from design DNA.",
            "sig": "INFILL DENSITY DRIFT."
        }
    }

    diag = defect_map[defect_idx]
    deviation = predicted_cg - target_cg
    is_balanced = abs(deviation) < BALANCE_TOLERANCE

    # 3. Dynamic Mitigation
    if not is_balanced:
        action = "ADD WEIGHT"
        status = diag["name"]
        corr_loc = 0.0 if deviation > 0 else 45.0
        lever_arm = abs(target_cg - corr_loc) + 1e-9
        corr_mass = (total_mass_kg * 1000.0 * abs(deviation)) / lever_arm
    else:
        action, status, corr_mass, corr_loc = "NONE", "BALANCED", 0.0, 0.0

    return {
        "w_root": round(w1_g, 1), "w_mid": round(w2_g, 1), "w_tip": round(w3_g, 1),
        "cg": round(predicted_cg, 2), "target_cg": round(target_cg, 2),
        "deviation": round(deviation, 3), "status": status,
        "status_desc": diag["sig"], "diagnosis_details": diag["desc"],
        "target_mass": target_mass,
        "correction": {"action": action, "mass": round(corr_mass, 1), "location": round(corr_loc, 1)},
        "report": {
            "items": [
                {"sensor": "Root", "value": round(w1_g/1000.0, 4), "impact": f"{(w1_g/total_g)*100:.1f}% load share."},
                {"sensor": "Mid", "value": round(w2_g/1000.0, 4), "impact": f"{(w2_g/total_g)*100:.1f}% support share."},
                {"sensor": "Tip", "value": round(w3_g/1000.0, 4), "impact": f"{(w3_g/total_g)*100:.1f}% mass share."}
            ],
            "interpretation": f"AI Diagnostic Result: {diag['name']} detected."
        }
    }

# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze_measurement():
    data = request.json
    # Logic: Accepts either a single set of weights or a buffer for averaging
    weights = data.get('buffer', [data]) # Default to single if no buffer sent
    
    if len(weights) > 1:
        # High-precision 60-sample averaging (from your old code logic)
        w_root = np.mean([w['w_root'] for w in weights])
        w_mid = np.mean([w['w_mid'] for w in weights])
        w_tip = np.mean([w['w_tip'] for w in weights])
    else:
        w_root = data.get('w_root', 0)
        w_mid = data.get('w_mid', 0)
        w_tip = data.get('w_tip', 0)

    result = perform_ai_analysis(w_root, w_mid, w_tip, data.get('blade_id', "0"))
    return jsonify(result)

@app.route('/list-blades', methods=['GET'])
def list_blades():
    blades = []
    if os.path.exists(CUSTOM_DIR):
        for bid in os.listdir(CUSTOM_DIR):
            meta_path = os.path.join(CUSTOM_DIR, bid, "meta.pkl")
            if os.path.exists(meta_path):
                blades.append({"id": bid, "name": f"Custom Blade {bid}"})
    return jsonify(blades)

@app.route('/download-report', methods=['POST'])
def download_report():
    # Use the Professional Report Generator class here (omitted for brevity, keep your FPDF class)
    pass 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# -*- coding: utf-8 -*-
"""
GyroBalance Cloud Backend - app.py (RENDER STABLE VERSION)
Fixes: Keras quantization_config error, 404 Route errors, and Professional PDF generation.
"""

import os
import json
import uuid
import joblib
import numpy as np
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# AI & Physics Imports
import tensorflow as tf
from tensorflow.keras import models
import trimesh
from fpdf import FPDF

# ─────────────────────────────────────────────
# APP SETUP & CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(BASE_DIR, "custom_blades")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure critical directories exist
for folder in [CUSTOM_DIR, MODELS_DIR, STATIC_DIR]:
    os.makedirs(folder, exist_ok=True)

BALANCE_TOLERANCE = 0.15

# ─────────────────────────────────────────────
# CORE AI LOADER (VERSION-PROOF)
# ─────────────────────────────────────────────
model_cache = {}

def get_model_and_meta(blade_id):
    """Safe loader for AI models. Uses compile=False to bypass Keras version mismatches."""
    bid_str = str(blade_id)
    if bid_str in model_cache:
        return model_cache[bid_str]

    try:
        if bid_str in ["0", "1"]:
            # Load Master Model
            model_path = os.path.join(MODELS_DIR, "gyro_master_model.keras")
            meta_path = os.path.join(MODELS_DIR, "blade_meta.pkl")
            meta_data = joblib.load(meta_path)
            meta = meta_data['B1'] if bid_str == "0" else meta_data['B2']
        else:
            # Load Custom Isolated Blade
            path = os.path.join(CUSTOM_DIR, bid_str)
            model_path = os.path.join(path, "model.keras")
            meta_path = os.path.join(path, "meta.pkl")
            meta = list(joblib.load(meta_path).values())[0]

        # THE CRITICAL FIX: compile=False ignores version-specific quantization metadata
        model = tf.keras.models.load_model(model_path, compile=False)
        model_cache[bid_str] = (model, meta)
        return model, meta
    except Exception as e:
        print(f"❌ AI Loader Failure for {bid_str}: {e}")
        return None, None

# ─────────────────────────────────────────────
# PROFESSIONAL PDF ENGINE
# ─────────────────────────────────────────────
class ProfessionalReport(FPDF):
    def header(self):
        logo_path = os.path.join(STATIC_DIR, "logo.png")
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 30)
        
        self.set_font('Arial', 'B', 16)
        self.set_text_color(63, 81, 181) # Indigo
        self.cell(80)
        self.cell(30, 10, 'GYROBALANCE AI DIAGNOSTIC', 0, 1, 'C')
        self.ln(10)
        self.set_draw_color(63, 81, 181)
        self.line(10, 35, 200, 35)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'GyroBalance Cloud Infrastructure | Page {self.page_no()}', 0, 0, 'C')

    def section_header(self, text):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, f"  {text.upper()}", 0, 1, 'L', True)
        self.ln(2)

    def data_row(self, label, value):
        self.set_font('Arial', 'B', 10)
        self.cell(60, 8, f"{label}:", 0, 0)
        self.set_font('Arial', '', 10)
        self.cell(0, 8, str(value), 0, 1)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def health():
    return jsonify({"status": "healthy", "engine": "TensorFlow/Keras Stable"})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or data.get('ping'): return jsonify({"status": "online"})

    bid = str(data.get('blade_id', "0"))
    w_root = float(data.get('w_root', 0))
    w_mid = float(data.get('w_mid', 0))
    w_tip = float(data.get('w_tip', 0))

    model, meta = get_model_and_meta(bid)
    if not model: return jsonify({"error": "Model initialization failure"}), 500

    target_cg, target_mass = meta
    total_g = max(1e-9, w_root + w_mid + w_tip)
    
    # AI Inference: [type, w1, w2, w3]
    inputs = np.array([[float(bid) if bid.isdigit() else 2.0, w_root/1000, w_mid/1000, w_tip/1000]])
    preds = model.predict(inputs)
    
    pred_cg = float(preds[0][0][0])
    defect_probs = preds[1][0]
    defect_idx = np.argmax(defect_probs)

    defect_map = {
        0: ("HEALTHY DNA", "Structural load parity confirmed.", "DNA VERIFIED."),
        1: ("RESIN POOL", "Density excess at tip coordinates.", "TIP MASS ANOMALY."),
        2: ("AIR VOID", "Critical core density dropout.", "INTERNAL BREACH."),
        3: ("INFILL ERROR", "Distribution drift detected.", "INFILL DRIFT.")
    }
    
    status, diag_desc, sig = defect_map[defect_idx]
    deviation = pred_cg - target_cg
    is_balanced = abs(deviation) < BALANCE_TOLERANCE

    if not is_balanced:
        action = "ADD WEIGHT"
        corr_loc = 0.0 if deviation > 0 else 45.0
        corr_mass = (total_g * abs(deviation)) / (abs(target_cg - corr_loc) + 1e-9)
    else:
        action, status, corr_mass, corr_loc = "NONE", "BALANCED", 0.0, 0.0

    return jsonify({
        "w_root": w_root, "w_mid": w_mid, "w_tip": w_tip,
        "cg": round(pred_cg, 2), "target_cg": round(target_cg, 2),
        "deviation": round(deviation, 3), "status": status,
        "status_desc": sig, "diagnosis_details": diag_desc,
        "target_mass": target_mass,
        "correction": {"action": action, "mass": round(corr_mass, 1), "location": round(corr_loc, 1)},
        "report": {
            "items": [
                {"sensor": "Root Zone", "value": w_root/1000, "impact": f"{(w_root/total_g)*100:.1f}% Load"},
                {"sensor": "Mid Span", "value": w_mid/1000, "impact": f"{(w_mid/total_g)*100:.1f}% Load"},
                {"sensor": "Tip Zone", "value": w_tip/1000, "impact": f"{(w_tip/total_g)*100:.1f}% Load"}
            ],
            "interpretation": f"AI Diagnostic Result: {status}."
        }
    })

@app.route('/list-blades', methods=['GET'])
def list_blades():
    blades = []
    if os.path.exists(CUSTOM_DIR):
        for b_id in os.listdir(CUSTOM_DIR):
            blades.append({"id": b_id, "name": f"Custom Blade {b_id}"})
    return jsonify(blades)

@app.route('/get-stl/<blade_id>', methods=['GET'])
def get_stl(blade_id):
    path = os.path.join(CUSTOM_DIR, blade_id, "blade.stl")
    return send_file(path) if os.path.exists(path) else (jsonify({"error": "Not Found"}), 404)

@app.route('/add-blade', methods=['POST'])
def add_blade():
    file = request.files['stl']
    bid = str(uuid.uuid4())[:8]
    folder = os.path.join(CUSTOM_DIR, bid)
    os.makedirs(folder, exist_ok=True)
    
    stl_path = os.path.join(folder, "blade.stl")
    file.save(stl_path)
    
    mesh = trimesh.load(stl_path)
    density = float(request.form.get('density', 1.25))
    scale = 0.1
    ideal_cg = float(mesh.centroid[0]) * scale
    ideal_mass = (float(mesh.volume) * (scale**3) * density) / 1000
    
    joblib.dump({bid: [ideal_cg, ideal_mass]}, os.path.join(folder, "meta.pkl"))
    
    # Use Master Model as base template for the new blade
    master_path = os.path.join(MODELS_DIR, "gyro_master_model.keras")
    if os.path.exists(master_path):
        m = models.load_model(master_path, compile=False)
        m.save(os.path.join(folder, "model.keras"))

    return jsonify({"status": "registered", "blade_id": bid, "cg": ideal_cg})

@app.route('/download-report', methods=['POST'])
def download_report():
    data = request.json
    pdf = ProfessionalReport()
    pdf.add_page()
    
    pdf.section_header("1. Specifications & Metadata")
    pdf.data_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.data_row("Target Center of Gravity", f"{data.get('target_cg', 0)} cm")
    pdf.data_row("Target Mass Target", f"{data.get('target_mass', 0)*1000:.1f} g")
    
    pdf.ln(5)
    pdf.section_header("2. Sensor Payload Analytics")
    pdf.data_row("Load Cell 1 (Root)", f"{data.get('w_root', 0)} g")
    pdf.data_row("Load Cell 2 (Mid)", f"{data.get('w_mid', 0)} g")
    pdf.data_row("Load Cell 3 (Tip)", f"{data.get('w_tip', 0)} g")

    pdf.ln(5)
    pdf.section_header("3. AI Neural Diagnostic")
    pdf.data_row("Calculated CG Coordinate", f"{data.get('cg', 0)} cm")
    pdf.data_row("Measured Deviation", f"{data.get('deviation', 0)} cm")
    pdf.data_row("Health Classification", data.get('status', 'PENDING'))
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 8, f"INTERPRETATION: {data.get('status_desc', '')} {data.get('diagnosis_details', '')}")
    
    pdf.ln(5)
    pdf.section_header("4. Mitigation Strategy")
    correction = data.get('correction', {})
    if correction.get('action') != "NONE":
        pdf.set_text_color(200, 0, 0)
        pdf.data_row("Action Directive", correction.get('action'))
        pdf.data_row("Counterweight Mass", f"{correction.get('mass')} g")
        pdf.data_row("Axial Placement", f"@{correction.get('location')} cm")
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 10, "VALIDATED: No counterbalance ballast required.", 0, 1)

    pdf_path = os.path.join(BASE_DIR, "report.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

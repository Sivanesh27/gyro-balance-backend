# -*- coding: utf-8 -*-
"""
GyroBalance Cloud Backend - app.py
Optimization: Keras 3 Compatibility & Memory Management for Render
Updated: Sensor Positions to 5.0, 22.5, 40.0
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
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(BASE_DIR, "custom_blades")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
for folder in [CUSTOM_DIR, MODELS_DIR, STATIC_DIR]:
    os.makedirs(folder, exist_ok=True)

# Updated Sensor Positions based on new hardware setup
SENSOR_POS = [5.0, 22.5, 40.0]
BALANCE_TOLERANCE = 0.15

# ─────────────────────────────────────────────
# CORE AI LOGIC (KERAS 3 COMPATIBLE)
# ─────────────────────────────────────────────
model_cache = {}

def get_model_and_meta(blade_id):
    """
    Retrieves AI model and physics metadata.
    Fixed: Uses keras.models.load_model to support Keras 3 features.
    """
    # Clean cache if it gets too large for Render RAM
    if len(model_cache) > 5:
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

        # Use the standalone Keras loader for Keras 3 (.keras files)
        model = keras.models.load_model(model_path)
        model_cache[blade_id] = (model, meta)
        return model, meta
    except Exception as e:
        print(f"❌ AI Loader Failure for {blade_id}: {str(e)}")
        return None, None

# ─────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────
class ProfessionalReport(FPDF):
    def header(self):
        logo_path = os.path.join(STATIC_DIR, "logo.png")
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 10, 25)
        
        self.set_xy(40, 12)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(63, 81, 181) # Indigo
        self.cell(0, 10, 'GYROBALANCE AI SYSTEM', 0, 1, 'L')
        self.set_x(40)
        self.set_font('Arial', 'B', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, 'INDUSTRIAL GRADE BLADE KINEMATICS & LOAD ANALYTICS', 0, 1, 'L')
        self.ln(10)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 11)
        self.set_text_color(63, 81, 181)
        self.set_fill_color(243, 246, 248)
        self.cell(0, 10, f"  {label.upper()}", 0, 1, 'L', True)
        self.ln(3)

    def add_data_row(self, label, value, unit=""):
        self.set_font('Arial', 'B', 9)
        self.set_text_color(100, 116, 139)
        self.cell(50, 8, f"{label}:", 0, 0)
        self.set_font('Arial', '', 10)
        self.set_text_color(30, 41, 59)
        self.cell(0, 8, f"{value} {unit}", 0, 1)

# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route('/list-blades', methods=['GET'])
def list_blades():
    blades = []
    if os.path.exists(CUSTOM_DIR):
        for bid in os.listdir(CUSTOM_DIR):
            meta_path = os.path.join(CUSTOM_DIR, bid, "meta.pkl")
            if os.path.exists(meta_path):
                blades.append({"id": bid, "name": f"Custom Blade {bid}"})
    return jsonify(blades)

@app.route('/analyze', methods=['POST'])
def analyze_measurement():
    data = request.json
    w_root = float(data.get('w_root', 0))
    w_mid = float(data.get('w_mid', 0))
    w_tip = float(data.get('w_tip', 0))
    blade_id = data.get('blade_id', "0")

    model, meta = get_model_and_meta(blade_id)
    if not model:
        return jsonify({"error": "Neural model failed to initialize"}), 500

    target_cg, target_mass = meta
    total_g = max(1e-9, w_root + w_mid + w_tip)
    
    # Scale inputs for model (kg)
    input_type = 2.0 if str(blade_id) not in ["0", "1"] else float(blade_id)
    input_data = np.array([[input_type, w_root/1000.0, w_mid/1000.0, w_tip/1000.0]], dtype=np.float32)
    
    # AI Inference
    preds = model.predict(input_data, verbose=0)
    pred_cg = float(preds[0][0][0])
    defect_probs = preds[1][0]
    defect_idx = np.argmax(defect_probs)

    defect_map = {
        0: {"name": "HEALTHY", "desc": "Validated load parity.", "sig": "DNA CONFIRMED."},
        1: {"name": "RESIN POOL", "desc": "Excess weight at tip.", "sig": "TIP ZONE ANOMALY."},
        2: {"name": "VOID", "desc": "Air pocket in core.", "sig": "CORE INTEGRITY BREACH."},
        3: {"name": "INFILL", "desc": "Density drift.", "sig": "INFILL SHIFT."}
    }

    diag = defect_map[defect_idx]
    deviation = pred_cg - target_cg
    is_balanced = abs(deviation) < BALANCE_TOLERANCE

    # Calculation for counter-ballast
    if not is_balanced:
        action = "ADD BALLAST"
        status = diag["name"]
        # Simple mitigation logic: place opposite to deviation
        corr_loc = 0.0 if deviation > 0 else 45.0
        corr_mass = (total_g * abs(deviation)) / (abs(target_cg - corr_loc) + 1e-9)
    else:
        action, status, corr_mass, corr_loc = "NONE", "BALANCED", 0.0, 0.0

    return jsonify({
        "w_root": w_root, "w_mid": w_mid, "w_tip": w_tip,
        "cg": round(pred_cg, 2), "target_cg": round(target_cg, 2),
        "deviation": round(deviation, 3), "status": status,
        "status_desc": diag["sig"], "diagnosis_details": diag["desc"],
        "target_mass": target_mass,
        "correction": {"action": action, "mass": round(corr_mass, 1), "location": round(corr_loc, 1)},
        "report": {
            "items": [
                {"sensor": "Root Zone", "value": w_root/1000.0, "impact": f"{(w_root/total_g)*100:.1f}% Impact"},
                {"sensor": "Mid Span", "value": w_mid/1000.0, "impact": f"{(w_mid/total_g)*100:.1f}% Impact"},
                {"sensor": "Tip Zone", "value": w_tip/1000.0, "impact": f"{(w_tip/total_g)*100:.1f}% Impact"}
            ],
            "interpretation": f"AI Diagnostic Result: {diag['name']} detected."
        }
    })

@app.route('/add-blade', methods=['POST'])
def add_new_blade():
    if 'stl' not in request.files:
        return jsonify({"error": "Missing STL"}), 400
    
    file = request.files['stl']
    density = float(request.form.get('density', 1.25))
    
    blade_id = str(uuid.uuid4())[:8]
    folder = os.path.join(CUSTOM_DIR, blade_id)
    os.makedirs(folder, exist_ok=True)
    
    stl_path = os.path.join(folder, "blade.stl")
    file.save(stl_path)
    
    try:
        # Physics DNA Extraction
        mesh = trimesh.load(stl_path)
        scale = 0.1 
        ideal_cg = float(mesh.centroid[0]) * scale
        volume_cm3 = float(mesh.volume) * (scale**3)
        ideal_mass = (volume_cm3 * density) / 1000
        
        meta = {blade_id: [ideal_cg, ideal_mass]}
        joblib.dump(meta, os.path.join(folder, "meta.pkl"))

        # Model Cloning (For Custom Blades, we reuse the architecture)
        def clone_model(bid, fld):
            master = keras.models.load_model(os.path.join(MODELS_DIR, "gyro_master_model.keras"))
            master.save(os.path.join(fld, "model.keras"))
            keras.backend.clear_session()

        threading.Thread(target=clone_model, args=(blade_id, folder)).start()

        return jsonify({
            "status": "Success",
            "blade_id": blade_id,
            "target_cg": ideal_cg,
            "target_mass": ideal_mass
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-report', methods=['POST'])
def download_report():
    data = request.json
    pdf = ProfessionalReport()
    pdf.add_page()
    
    pdf.chapter_title("1. Profile Specification")
    pdf.add_data_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.add_data_row("Design Target CG", f"{data['target_cg']}", "cm Axial")
    pdf.add_data_row("Design Target Mass", f"{data['target_mass']*1000:.1f}", "grams")
    
    pdf.chapter_title("2. Sensor Load Distribution")
    pdf.add_data_row("Root Section (S1)", f"{data['w_root']}", "g")
    pdf.add_data_row("Mid Span (S2)", f"{data['w_mid']}", "g")
    pdf.add_data_row("Tip Zone (S3)", f"{data['w_tip']}", "g")
    
    pdf.chapter_title("3. Neural AI Diagnostic")
    pdf.add_data_row("Calculated CG", f"{data['cg']}", "cm")
    pdf.add_data_row("Drift / Deviation", f"{data['deviation']}", "cm")
    
    status = data['status']
    is_healthy = status == "BALANCED" or status == "HEALTHY"
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(22, 163, 74) if is_healthy else pdf.set_text_color(220, 38, 38)
    pdf.cell(0, 10, f"CLASSIFICATION: {status}", 0, 1)
    
    pdf.set_text_color(30, 41, 59)
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(0, 6, f"AI Logic Trace: {data['status_desc']}")
    
    pdf.chapter_title("4. Engineering Mitigation Directive")
    corr = data['correction']
    if corr['action'] != "NONE":
        pdf.set_text_color(180, 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, f"REQUIRED ACTION: {corr['action']}", 0, 1)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"- Counter-Ballast Mass: {corr['mass']:.1f} grams", 0, 1)
        pdf.cell(0, 8, f"- Placement Coordinate: {corr['location']:.1f} cm Axial from Root", 0, 1)
    else:
        pdf.set_text_color(22, 163, 74)
        pdf.cell(0, 10, "CERTIFICATION: STRUCTURAL DNA VALIDATED. NO MITIGATION REQUIRED.", 0, 1)

    temp_path = os.path.join(BASE_DIR, "report_output.pdf")
    pdf.output(temp_path)
    return send_file(temp_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

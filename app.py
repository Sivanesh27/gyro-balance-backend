# -*- coding: utf-8 -*-
"""
GyroBalance Cloud Backend - app.py (Final Production Version)
Optimized for: Keras 3.13.2, NumPy 2.0.2, and Render Deployment
Includes: High-Precision Averaging & Enhanced Neural Diagnostics
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

# AI & Physics Imports - Mandatory Keras 3 Standalone
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

# Ensure environment directories exist
for folder in [CUSTOM_DIR, MODELS_DIR, STATIC_DIR]:
    os.makedirs(folder, exist_ok=True)

# Hardware Specifications
SENSOR_POS = [5.0, 22.5, 40.0]
BALANCE_TOLERANCE = 0.10

# ─────────────────────────────────────────────
# CORE AI ENGINE (KERAS 3 COMPATIBLE)
# ─────────────────────────────────────────────
model_cache = {}

def get_model_and_meta(blade_id):
    """
    Loads Keras 3 models using the standalone loader to prevent 
    deserialization errors regarding 'quantization_config'.
    """
    # RAM management for Render Free Tier
    if len(model_cache) > 2:
        model_cache.clear()
        keras.backend.clear_session()

    if blade_id in model_cache:
        return model_cache[blade_id]

    try:
        if str(blade_id) == "0":
            model_path = os.path.join(MODELS_DIR, "gyro_master_model.keras")
            meta = [11.83, 0.0]
        elif str(blade_id) == "1":
            model_path = os.path.join(MODELS_DIR, "gyro_master_model.keras")
            meta = [14.56, 0.0] 
        else:
            path = os.path.join(CUSTOM_DIR, str(blade_id))
            model_path = os.path.join(path, "model.keras")
            meta_path = os.path.join(path, "meta.pkl")
            meta_data = joblib.load(meta_path)
            meta = meta_data[blade_id]

        # Critical: Use standalone keras to support Colab Keras 3.13.2 features
        model = keras.models.load_model(model_path, compile=False)
        model_cache[blade_id] = (model, meta)
        return model, meta
    except Exception as e:
        print(f"❌ AI Loader Failure for {blade_id}: {str(e)}")
        return None, None

def perform_ai_analysis(w1_g, w2_g, w3_g, blade_id) -> dict:
    """
    Executes Neural Inference and generates industrial diagnostics.
    """
    model, meta = get_model_and_meta(blade_id)
    if not model:
        return {"error": "AI Engine Offline: Model Deserialization Failed"}

    target_cg, target_mass = meta
    total_g = max(1e-9, w1_g + w2_g + w3_g)
    total_mass_kg = total_g / 1000.0

    # 1. AI Inference (Keras 3 Multi-Output)
    input_type = float(blade_id) if str(blade_id).isdigit() else 2.0
    X = np.array([[input_type, w1_g/1000.0, w2_g/1000.0, w3_g/1000.0]], dtype=np.float32)
    preds = model.predict(X, verbose=0)
    
    predicted_cg = float(preds[0][0][0])
    defect_idx = int(np.argmax(preds[1][0]))

    # 2. Enhanced Neural Interpretation (From your working logic)
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

    # 3. Dynamic Counter-Ballast Logic
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
# PDF REPORT GENERATOR (RESTORED & OPTIMIZED)
# ─────────────────────────────────────────────
class ProfessionalReport(FPDF):
    def header(self):
        # Logo handling (ensure static/logo.png exists in your repo)
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

@app.route('/analyze', methods=['POST'])
def analyze_measurement():
    """
    Receives weight data. Supports high-precision averaging if a buffer is sent.
    """
    data = request.json
    weights = data.get('buffer', [data]) 
    
    if len(weights) > 1:
        # High-precision averaging logic
        w_root = np.mean([w.get('w_root', 0) for w in weights])
        w_mid = np.mean([w.get('w_mid', 0) for w in weights])
        w_tip = np.mean([w.get('w_tip', 0) for w in weights])
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
    """
    Generates a professional PDF diagnostic.
    Uses /tmp for Render filesystem compatibility.
    """
    try:
        data = request.json
        pdf = ProfessionalReport()
        pdf.add_page()
        
        # 1. Profile Specification
        pdf.chapter_title("1. Profile Specification")
        pdf.add_data_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pdf.add_data_row("Design Target CG", f"{data.get('target_cg', 0):.2f}", "cm Axial")
        pdf.add_data_row("Design Target Mass", f"{data.get('target_mass', 0)*1000:.1f}", "grams")
        
        # 2. Sensor Load Distribution
        pdf.chapter_title("2. Sensor Load Distribution")
        pdf.add_data_row("Root Section (S1)", f"{data.get('w_root', 0):.1f}", "g")
        pdf.add_data_row("Mid Span (S2)", f"{data.get('w_mid', 0):.1f}", "g")
        pdf.add_data_row("Tip Zone (S3)", f"{data.get('w_tip', 0):.1f}", "g")
        
        # 3. AI Neural Diagnostic
        pdf.chapter_title("3. Neural AI Diagnostic")
        pdf.add_data_row("Calculated CG", f"{data.get('cg', 0):.2f}", "cm")
        pdf.add_data_row("Drift / Deviation", f"{data.get('deviation', 0):.3f}", "cm")
        
        status = data.get('status', 'Unknown')
        pdf.set_font('Arial', 'B', 10)
        # Visual color coding for status
        if "BALANCED" in status or "HEALTHY" in status:
            pdf.set_text_color(22, 163, 74) # Emerald Green
        else:
            pdf.set_text_color(220, 38, 38) # Crimson Red
        
        pdf.cell(0, 10, f"CLASSIFICATION: {status}", 0, 1)
        
        pdf.set_text_color(30, 41, 59)
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 6, f"AI Logic Trace: {data.get('status_desc', 'No signature detected.')}")
        
        # 4. Engineering Mitigation
        pdf.chapter_title("4. Engineering Mitigation Directive")
        correction = data.get('correction', {})
        if correction.get('action') != "NONE":
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(180, 0, 0)
            pdf.cell(0, 8, f"REQUIRED ACTION: {correction.get('action')}", 0, 1)
            pdf.set_text_color(30, 41, 59)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 8, f"- Counter-Ballast Mass: {correction.get('mass', 0):.1f} grams", 0, 1)
            pdf.cell(0, 8, f"- Placement Coordinate: {correction.get('location', 0):.1f} cm Axial from Root", 0, 1)
        else:
            pdf.set_text_color(22, 163, 74)
            pdf.cell(0, 10, "CERTIFICATION: STRUCTURAL DNA VALIDATED. NO MITIGATION REQUIRED.", 0, 1)

        report_filename = f"GyroBalance_Report_{uuid.uuid4().hex[:8]}.pdf"
        temp_path = os.path.join("/tmp", report_filename)
        pdf.output(temp_path)

        response = send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=report_filename
        )
        response.headers['Content-Disposition'] = f'attachment; filename="{report_filename}"'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        response.headers['Cache-Control'] = 'no-cache'
        return response

    except Exception as e:
        print(f"❌ PDF Generation Error: {str(e)}")
        return jsonify({"error": "Failed to generate PDF. Check backend logs."}), 500

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

        # Model Cloning (For Custom Blades, we reuse architecture)
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

if __name__ == "__main__":
    # Render deployment uses PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

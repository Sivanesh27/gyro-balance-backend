# -*- coding: utf-8 -*-
"""
GyroBalance Physics Engine - physics_engine.py
Responsible for extracting structural DNA (CG and Mass) from STL geometry.
"""

import trimesh
import os

class BladePhysics:
    def __init__(self, scale_factor=0.1, default_density=1.25):
        """
        Initializes the physics engine.
        :param scale_factor: Conversion from STL units (usually mm) to dashboard units (cm). Default 0.1.
        :param default_density: Material density in g/cm3. Default 1.25 (common for PLA/Carbon-infused).
        """
        self.scale = scale_factor
        self.density = default_density

    def extract_dna(self, file_path, custom_density=None):
        """
        Analyzes an STL file to determine its theoretical ideal balance point.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"STL file not found at: {file_path}")

        try:
            # Load the mesh
            # Use rtree if available for better performance on complex meshes
            mesh = trimesh.load(file_path)

            # 1. Calculate Center of Gravity (Centroid)
            # trimesh.centroid returns the center of mass assuming uniform density
            # We take the X-axis as the primary longitudinal axis of the blade
            raw_centroid_x = float(mesh.centroid[0])
            ideal_cg_cm = raw_centroid_x * self.scale

            # 2. Calculate Volume and Mass
            # Volume is returned in cubic units of the STL (mm3)
            volume_mm3 = float(mesh.volume)
            volume_cm3 = volume_mm3 * (self.scale ** 3)
            
            density_to_use = custom_density if custom_density is not None else self.density
            # Mass = Volume * Density (Result in grams)
            ideal_mass_g = volume_cm3 * density_to_use
            # Convert to Kg for the AI internal logic if needed
            ideal_mass_kg = ideal_mass_g / 1000.0

            return {
                "cg_cm": round(ideal_cg_cm, 3),
                "mass_kg": round(ideal_mass_kg, 4),
                "mass_g": round(ideal_mass_g, 2),
                "volume_cm3": round(volume_cm3, 2),
                "surface_area": round(float(mesh.area), 2),
                "is_watertight": mesh.is_watertight
            }

        except Exception as e:
            print(f"Physics Extraction Error: {str(e)}")
            return None

def get_stl_physics(file_path, density=1.25):
    """
    Helper function for direct calls from app.py or trainer.py
    """
    engine = BladePhysics(default_density=density)
    result = engine.extract_dna(file_path)
    if result:
        return result["cg_cm"], result["mass_kg"]
    return None, None
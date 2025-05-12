import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares
import pandas as pd

###############################################################################
# SECTION 1: Dynamic Constant Framework and Helper Functions
###############################################################################
class DynamicConstant:
    """
    A framework for dynamic constants.
    
    Attributes:
      base_value: The standard (literature) value.
      name: Name of the constant.
      correction_factors: List of functions that, given a state dict,
                          return multiplicative factors.
    """
    def __init__(self, base_value, name, correction_factors=None):
        self.base_value = base_value
        self.name = name
        self.correction_factors = correction_factors if correction_factors is not None else []
    
    def add_correction(self, factor_func):
        self.correction_factors.append(factor_func)
    
    def get_value(self, state):
        effective_value = self.base_value
        for func in self.correction_factors:
            effective_value *= func(state)
        return effective_value

def PiNova(x):
    """
    Compute the PiNova factor using a step-by-step arithmetic approach.
    For x = 1 the result is nearly π.
    """
    a = math.sqrt(9.869582667)  # ≈ 3.141589194500135
    step1 = (x + 1) / 100.0
    step2 = step1 * 99.0
    step3 = step2 * a
    step4 = step3 / 99.0
    step5 = step4 * 100.0
    result = step5 - a
    return result

###############################################################################
# SECTION 2: Correction Functions for the Molecular Resonance Integral (β)
###############################################################################
def molecular_correction(state):
    """
    Correction based on molecular-scale variables.
    
    Expected keys in state:
        'molecular_length': typical C–C bond length (e.g., ~1.4e-10 m)
        'molecular_time': associated timescale (e.g., ~1e-15 s)
        'kappa_mol': a dimensionless scaling factor
        'scale_mol': a normalization factor to moderate the correction
    Returns:
        1 + kappa_mol * ((molecular_length/molecular_time) / scale_mol)
    """
    kappa_mol = state.get('kappa_mol', 1e-3)
    length = state.get('molecular_length', 1.4e-10)
    time_val = state.get('molecular_time', 1e-15)
    scale_mol = state.get('scale_mol', 1e5)
    return 1.0 + kappa_mol * ((length / time_val) / scale_mol)

def pinova_molecular_correction(state):
    """
    A PiNova-derived correction for molecular geometry.
    
    Uses key 'molecular_diameter'. For the default value of 1.0, the correction 
    is nearly 1.
    """
    molecular_diameter = state.get('molecular_diameter', 1.0)
    return PiNova(molecular_diameter) / math.pi

def amplification_factor(state):
    """
    An extra amplification factor applied to the effective resonance integral.
    Default is 1.0 (no amplification).
    """
    return state.get("amplification", 1.0)

###############################################################################
# SECTION 3: Set Up the Dynamic Resonance Integral (β) for the Molecule
###############################################################################
# For benzene (and similar conjugated systems) the textbook value for β is about -2.5 eV.
beta_base = -2.5  # in electronvolts (eV)

# Create an instance of the dynamic constant for β and register corrections.
beta_dynamic = DynamicConstant(beta_base, "Resonance Integral β")
beta_dynamic.add_correction(molecular_correction)
beta_dynamic.add_correction(pinova_molecular_correction)
beta_dynamic.add_correction(amplification_factor)

###############################################################################
# SECTION 4: Experimental Data and Molecular Properties for Calibration
###############################################################################
# Experimental HOMO–LUMO gaps (approximate, in eV) for conjugated acene systems:
experimental_gaps = {
    "benzene": 6.20,
    "naphthalene": 4.80,
    "anthracene": 3.70,
    "tetracene": 3.00,
    "pentacene": 2.30,
    "hexacene": 1.80  # extended dataset
}

# Molecular properties (bond length and timescale) for each molecule.
molecule_data = {
    "benzene": {"length": 1.4e-10, "time": 1e-15},
    "naphthalene": {"length": 1.42e-10, "time": 1e-15},
    "anthracene": {"length": 1.44e-10, "time": 1e-15},
    "tetracene": {"length": 1.46e-10, "time": 1e-15},
    "pentacene": {"length": 1.48e-10, "time": 1e-15},
    "hexacene": {"length": 1.50e-10, "time": 1e-15}
}

###############################################################################
# SECTION 5: Define the Dynamic GAP Function and Calibration Objective
###############################################################################
def dynamic_gap(params, molecule_props):
    """
    Compute the predicted HOMO–LUMO gap (in eV) for a molecule using the dynamic model.
    
    Parameters:
        params: [amplification, kappa_mol, scale_mol, molecular_diameter]
        molecule_props: dictionary containing 'length' and 'time'.
        
    The gap is taken as 2 * |β_eff| (following Hückel theory).
    """
    amplification, kappa_mol, scale_mol, molecular_diameter = params
    state = {
        'molecular_length': molecule_props['length'],
        'molecular_time': molecule_props['time'],
        'kappa_mol': kappa_mol,
        'scale_mol': scale_mol,
        'molecular_diameter': molecular_diameter,
        'amplification': amplification
    }
    beta_eff = beta_dynamic.get_value(state)
    gap = 2 * abs(beta_eff)
    return gap

def objective(params):
    """
    Objective function that returns the error vector between the
    predicted HOMO–LUMO gaps (using the dynamic model) and
    the experimental gaps for each molecule.
    """
    errors = []
    for mol, props in molecule_data.items():
        pred_gap = dynamic_gap(params, props)
        error = pred_gap - experimental_gaps[mol]
        errors.append(error)
    return np.array(errors)

###############################################################################
# SECTION 6: Calibration via Least-Squares Optimization
###############################################################################
# Initial guess for the parameters: [amplification, kappa_mol, scale_mol, molecular_diameter]
initial_guess = [1.0, 1e-3, 1e5, 1.0]
result = least_squares(objective, initial_guess)
optimized_params = result.x
print("Optimized dynamic parameters (amplification, kappa_mol, scale_mol, molecular_diameter):")
print(optimized_params)

###############################################################################
# SECTION 7: Compute Predicted Gaps for Each Molecule
###############################################################################
# We'll calculate:
# - Standard Physics prediction: using default parameters (amplification=1, molecular_diameter=1)
# - Dynamic Model prediction: using our calibrated (optimized) parameters.
predicted_standard = {}
predicted_dynamic = {}
molecules = list(molecule_data.keys())

# Define a state for standard physics (no dynamic corrections, fixed parameters).
state_standard = {'kappa_mol': 1e-3, 'scale_mol': 1e5, 'molecular_diameter': 1.0, 'amplification': 1.0}

for mol in molecules:
    # Fill in molecule-specific values.
    std_state = state_standard.copy()
    std_state['molecular_length'] = molecule_data[mol]['length']
    std_state['molecular_time'] = molecule_data[mol]['time']
    beta_eff_std = beta_dynamic.get_value(std_state)
    gap_std = 2 * abs(beta_eff_std)
    predicted_standard[mol] = gap_std
    
    # Build dynamic state using optimized parameters.
    amp, kappa_mol_opt, scale_mol_opt, mol_diam_opt = optimized_params
    dyn_state = {
        'molecular_length': molecule_data[mol]['length'],
        'molecular_time': molecule_data[mol]['time'],
        'kappa_mol': kappa_mol_opt,
        'scale_mol': scale_mol_opt,
        'molecular_diameter': mol_diam_opt,
        'amplification': amp
    }
    beta_eff_dyn = beta_dynamic.get_value(dyn_state)
    gap_dyn = 2 * abs(beta_eff_dyn)
    predicted_dynamic[mol] = gap_dyn

###############################################################################
# SECTION 8: Comparative Table and Plot of Results
###############################################################################
rows = []
for mol in molecules:
    rows.append({
        "Molecule": mol.capitalize(),
        "Standard Physics (eV)": round(predicted_standard[mol], 2),
        "Dynamic Model (eV)": round(predicted_dynamic[mol], 2),
        "Experimental (eV)": experimental_gaps[mol]
    })
df = pd.DataFrame(rows)
print("\nComparative Table of HOMO–LUMO Gaps (in eV):")
print(df)

# Plot grouped bar chart for the comparison.
labels = df["Molecule"]
standard_vals = df["Standard Physics (eV)"]
dynamic_vals = df["Dynamic Model (eV)"]
experimental_vals = df["Experimental (eV)"]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, standard_vals, width, label='Standard Physics', color='gray')
rects2 = ax.bar(x, dynamic_vals, width, label='Dynamic Model', color='blue')
rects3 = ax.bar(x + width, experimental_vals, width, label='Experimental', color='orange')

ax.set_ylabel("HOMO–LUMO Gap (eV)", fontsize=12)
ax.set_title("Extended Calibration: HOMO–LUMO Gap Predictions for Conjugated Molecules", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
                    
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()

###############################################################################
# SECTION 9: Wrapping Up & Future Extensions
###############################################################################
# This modular code is designed to be easily extendable. To add additional molecules,
# simply update the "experimental_gaps" and "molecule_data" dictionaries.
# Additional correction functions can be appended to the DynamicConstant instance.
# Further optimization and Bayesian or uncertainty quantification methods can be integrated
# to enhance the calibration process.
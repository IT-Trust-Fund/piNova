import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares
import pandas as pd

###############################################################################
# SECTION 1: Define the Dynamic Constant Framework and Helper Functions
###############################################################################

class DynamicConstant:
    """
    A framework for dynamic constants.
    
    Attributes:
      base_value: The standard (literature) value.
      name: Name of the constant.
      correction_factors: List of functions; each accepts a state dict 
                          and returns a multiplicative factor.
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
    Compute a PiNova factor using step-by-step arithmetic.
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
# SECTION 2: Define Correction Functions for Molecular Resonance Integral (β)
###############################################################################

def molecular_correction(state):
    """
    Correction based on molecular-scale variables.
    
    Expected keys in state:
        'molecular_length': e.g. typical C–C bond length (~1.4e-10 m)
        'molecular_time': e.g. an associated timescale (~1e-15 s)
        'kappa_mol': dimensionless scaling factor
        'scale_mol': normalization constant for the correction
    Returns: 1 + kappa_mol * ((molecular_length / molecular_time) / scale_mol)
    """
    kappa_mol = state.get('kappa_mol', 1e-3)
    length = state.get('molecular_length', 1.4e-10)
    time_val = state.get('molecular_time', 1e-15)
    scale_mol = state.get('scale_mol', 1e5)
    return 1.0 + kappa_mol * ((length / time_val) / scale_mol)

def pinova_molecular_correction(state):
    """
    A PiNova-derived correction for molecular parameters.
    
    Uses key 'molecular_diameter'. For a default value of 1.0 the correction is nearly 1.
    """
    molecular_diameter = state.get('molecular_diameter', 1.0)
    return PiNova(molecular_diameter) / math.pi

def amplification_factor(state):
    """
    Extra amplification multiplier to boost the effective resonance integral.
    Default of 1.0 (no amplification).
    """
    return state.get("amplification", 1.0)

###############################################################################
# SECTION 3: Set Up the Dynamic Constant for the Resonance Integral β
###############################################################################
# Standard Hückel value for β is taken as -2.5 eV for benzene.
beta_base = -2.5  # eV

# Create a dynamic constant instance for β and add our correction functions.
beta_dynamic = DynamicConstant(beta_base, "Resonance Integral β")
beta_dynamic.add_correction(molecular_correction)
beta_dynamic.add_correction(pinova_molecular_correction)
beta_dynamic.add_correction(amplification_factor)

###############################################################################
# SECTION 4: Define Experimental Data and Molecular Properties for Calibration
###############################################################################
# Experimental HOMO–LUMO gaps (approximate, in eV) for conjugated systems:
experimental_gaps = {
    "benzene": 6.20,
    "naphthalene": 4.80,
    "anthracene": 3.70,
    "tetracene": 3.00,
    "pentacene": 2.30
}

# Molecular properties for calibration (example bond lengths and timescales):
molecule_data = {
    "benzene": {"length": 1.4e-10, "time": 1e-15},
    "naphthalene": {"length": 1.42e-10, "time": 1e-15},
    "anthracene": {"length": 1.44e-10, "time": 1e-15},
    "tetracene": {"length": 1.46e-10, "time": 1e-15},
    "pentacene": {"length": 1.48e-10, "time": 1e-15}
}

###############################################################################
# SECTION 5: Define the Dynamic Gap Function and Calibration Objective Function
###############################################################################
def dynamic_gap(params, molecule_props):
    """
    Compute the predicted HOMO–LUMO gap for a molecule using the dynamic model.
    
    Parameters:
        params: A vector with dynamic parameters:
                [amplification, kappa_mol, scale_mol, molecular_diameter]
        molecule_props: A dictionary for the molecule (with 'length' and 'time' keys).
    
    Returns:
        The predicted gap (in eV), calculated as 2 * |β_eff|.
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
    Objective function that computes the error vector between the dynamic model's
    predicted gaps and the experimental gaps for all molecules.
    """
    errors = []
    for mol, props in molecule_data.items():
        pred_gap = dynamic_gap(params, props)
        err = pred_gap - experimental_gaps[mol]
        errors.append(err)
    return np.array(errors)

###############################################################################
# SECTION 6: Calibration Using Experimental Data (Least Squares Optimization)
###############################################################################
# Initial guess for parameters: [amplification, kappa_mol, scale_mol, molecular_diameter]
initial_guess = [1.0, 1e-3, 1e5, 1.0]
result = least_squares(objective, initial_guess)
optimized_params = result.x
print("Optimized dynamic parameters (amplification, kappa_mol, scale_mol, molecular_diameter):")
print(optimized_params)

###############################################################################
# SECTION 7: Compute Predicted Gaps (Standard and Dynamic) for Each Molecule
###############################################################################
# For a "Standard Physics" model, we assume dynamic corrections are off:
state_standard = {
    'molecular_length': None,  # will be filled per molecule
    'molecular_time': None,
    'kappa_mol': 1e-3,
    'scale_mol': 1e5,
    'molecular_diameter': 1.0,
    'amplification': 1.0
}

# Create dictionaries to store gaps for each molecule
predicted_standard = {}
predicted_dynamic = {}
molecules = list(molecule_data.keys())

for mol in molecules:
    # Fill state for each molecule using its intrinsic properties.
    std_state = state_standard.copy()
    std_state['molecular_length'] = molecule_data[mol]['length']
    std_state['molecular_time'] = molecule_data[mol]['time']
    
    # Standard mode: use default parameters (amplification=1, molecular_diameter=1)
    beta_eff_std = beta_dynamic.get_value(std_state)
    gap_std = 2 * abs(beta_eff_std)
    predicted_standard[mol] = gap_std
    
    # Dynamic mode: use optimized parameters from calibration:
    amp, kappa_mol, scale_mol, mol_diam = optimized_params
    dyn_state = {
        'molecular_length': molecule_data[mol]['length'],
        'molecular_time': molecule_data[mol]['time'],
        'kappa_mol': kappa_mol,
        'scale_mol': scale_mol,
        'molecular_diameter': mol_diam,
        'amplification': amp
    }
    beta_eff_dyn = beta_dynamic.get_value(dyn_state)
    gap_dyn = 2 * abs(beta_eff_dyn)
    predicted_dynamic[mol] = gap_dyn

###############################################################################
# SECTION 8: Create a Comparative Table and Plot the Results
###############################################################################
# Construct a DataFrame for comparison.
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

# Plot a grouped bar chart for comparison.
labels = df["Molecule"]
standard_vals = df["Standard Physics (eV)"]
dynamic_vals = df["Dynamic Model (eV)"]
experimental_vals = df["Experimental (eV)"]

x = np.arange(len(labels))  # label locations
width = 0.25  # width of each bar

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, standard_vals, width, label='Standard Physics')
rects2 = ax.bar(x, dynamic_vals, width, label='Dynamic Model')
rects3 = ax.bar(x + width, experimental_vals, width, label='Experimental')

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel("HOMO–LUMO Gap (eV)", fontsize=12)
ax.set_title("Comparison of HOMO–LUMO Gap Predictions for Conjugated Molecules", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Attach a text label above each bar, displaying its height.
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
# END OF SCRIPT
###############################################################################
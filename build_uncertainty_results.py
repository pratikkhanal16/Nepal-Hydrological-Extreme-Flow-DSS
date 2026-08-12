"""Precompute uncertainty results for the deployed DSS.
Run from the project root after placing the annual Excel file in data/.
"""
import os
from hydroevt_uncertainty_v2 import build_workbook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
annual_excel = os.path.join(BASE_DIR, "data", "Maximum Yearly Discharge.xlsx")
output_excel = os.path.join(BASE_DIR, "results", "Nepal_HydroEVT_Uncertainty_v2.xlsx")
os.makedirs(os.path.dirname(output_excel), exist_ok=True)

build_workbook(
    annual_excel=annual_excel,
    output_excel=output_excel,
    return_periods=[2, 5, 10, 25, 50, 100, 200],
    n_boot=2000,
    seed=42,
    min_valid_years=20,
)

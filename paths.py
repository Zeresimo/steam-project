"""
Centralized definitions for all folder paths in the Steam Review Analyzer project.

This file prevents hard-coded paths across the project and keeps directory
structure consistent for:
- Pipeline (fetching, cleaning, saving raw/cleaned reviews)
- ML models (training, saving, prediction)
- EDA outputs (plots and wordclouds)
- Dashboard (Streamlit app)

NOTES:
- Always import paths from here instead of manually typing folder names.
- ROOT is automatically added to sys.path so imports work everywhere.
"""

import os
import sys

# === Universal ROOT ===
# Base directory of the project (folder containing this file)
ROOT = os.path.dirname(os.path.abspath(__file__))

# === Main Folders ===
PIPELINE_DIR  = os.path.join(ROOT, "Pipeline")
ML_DIR        = os.path.join(ROOT, "ML")
EDA_DIR       = os.path.join(ROOT, "EDA")
DASHBOARD_DIR = os.path.join(ROOT, "Dashboard")

# === Pipeline subfolders ===
PIPE_RAW        = os.path.join(PIPELINE_DIR, "data", "raw")
PIPE_CLEAN      = os.path.join(PIPELINE_DIR, "data", "clean")
PIPE_PROCESSED  = os.path.join(PIPELINE_DIR, "data", "processed")
PIPE_LOGS       = os.path.join(PIPELINE_DIR, "logs")

# === ML subfolders ===
MODEL_DIR  = os.path.join(ML_DIR, "models")
LOG_DIR    = os.path.join(ML_DIR, "logs")
OUTPUT_DIR = os.path.join(ML_DIR, "output")

# === EDA subfolders ===
EDA_PLOTS      = os.path.join(EDA_DIR, "plots")
EDA_WORDCLOUDS = os.path.join(EDA_DIR, "wordclouds")
EDA_LOGS       = os.path.join(EDA_DIR, "logs")

# Ensure the project root is on sys.path so "import Pipeline..." works everywhere
sys.path.insert(0, ROOT)

import os
import sys

# === Universal ROOT ===
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

# Ensure the ROOT is searchable for imports
sys.path.insert(0, ROOT)

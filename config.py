# Requirement 7: confidence below this triggers the manual-review warning
CONFIDENCE_THRESHOLD = 0.60

# Which Gemini model to call for the assisted response
# gemini-2.0-flash was deprecated (retired March 2026) and has no free quota.
# gemini-2.5-flash is the current free-tier default as of mid-2026.
LLM_MODEL_NAME = "gemini-flash-latest"

# File paths for the saved training artifacts
DATA_PATH = "data/tickets.csv"
MODEL_PATH = "models/best_model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"
METRICS_PATH = "models/metrics.json"
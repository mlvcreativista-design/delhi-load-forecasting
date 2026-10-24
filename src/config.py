from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# Delhi coordinates (for weather API)
LAT, LON = 28.6139, 77.2090
TIMEZONE = "Asia/Kolkata"

RANDOM_STATE = 42

# Day-ahead forecasting: every lag must be >= 24h so we never use information
# that would not be available when the forecast is issued.
LAGS = (24, 48, 168)

# Default battery used in the peak-shaving demo (scaled for Delhi's ~4-8 GW load)
BESS_POWER_MW = 300
BESS_ENERGY_MWH = 1200
BESS_EFFICIENCY = 0.92  # one-way efficiency

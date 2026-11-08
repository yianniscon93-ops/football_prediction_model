"""footy_prediction_model - v3 Protocol as a statistical model.

Time-decayed Poisson / Dixon-Coles goals model where every v3 gate is a
feature with a learned weight. Scores Stoiximan slips leg-by-leg against
de-margined bookmaker odds and assigns 🟢🟡🟠🔴 flags.
"""
from .config import FlagConfig, ModelConfig
from .data import load_matches
from .features import build_features
from .model import FootyModel
from .predict import format_report, score_slip

__all__ = ["ModelConfig", "FlagConfig", "load_matches", "build_features", "FootyModel",
           "score_slip", "format_report"]
__version__ = "0.2.0"

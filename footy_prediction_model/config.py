"""Configuration for footy_prediction_model.

Every v3 gate maps to a feature group. Gate weights (betas) are LEARNED from
historical data, except the Gate 3 "manual" inputs (injuries, GK out, midweek
Europe) which have no historical data and use fixed prior coefficients.
"""
from dataclasses import dataclass, field

# Feature groups -> v3 gates. Column names are from the perspective of the
# "team" whose goals are being predicted, against "opp".
FEATURE_GROUPS = {
    "G1_xg_form": ["att_xg_form", "opp_def_xg_form"],
    "G1_form_streak": ["streak", "opp_streak"],
    "G1_promoted": ["promoted", "opp_promoted"],
    "G2_h2h": ["h2h_resid"],
    "G3_rest": ["rest", "opp_rest"],
    "G4_motivation": ["stakes", "opp_stakes", "dead_rubber", "opp_dead_rubber",
                      "new_mgr", "opp_new_mgr"],
    "G5_strength": ["elo_diff"],
}
LEARNED_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

# Gate 3 manual inputs (entered per fixture at bet time). Prior coefficients on
# log(expected goals). These are NOT learned - adjust as you gather evidence.
#   absence_att      share (0-1) of the team's attacking output (xG+xA) missing
#   opp_absence_def  share (0-1) of the opponent's first-choice defence missing
#   opp_gk_out       1 if opponent's first-choice keeper is out
#   euro_midweek     1 if the team played in Europe 2-3 days before
#   opp_euro_midweek 1 if the opponent did
MANUAL_PRIORS = {
    "absence_att": -0.80,      # 25% of attack missing -> ~-18% goals
    "opp_absence_def": 0.35,   # 50% of opp defence missing -> ~+19% goals
    "opp_gk_out": 0.10,
    "euro_midweek": -0.04,
    "opp_euro_midweek": 0.03,
}
MANUAL_FEATURES = list(MANUAL_PRIORS)

# Position of the "Europe" line per league (for the motivation feature).
EUROPE_LINE = {"E0": 5, "SP1": 5, "I1": 4, "D1": 4, "F1": 3, "N1": 2, "P1": 3, "G1": 4}
DEFAULT_EUROPE_LINE = 4
RELEGATION_SPOTS = {"D1": 2, "F1": 2, "N1": 2, "G1": 2}  # approx (play-offs ignored)
DEFAULT_RELEGATION_SPOTS = 3

DEFAULT_LEAGUES = ["E0", "SP1", "I1", "D1", "F1"]


@dataclass
class ModelConfig:
    xi: float = 0.002             # time decay per day (half-life ~ 350 days; tuned on 2024/25 with xG)
    alpha: float = 3e-3           # L2 (ridge) strength in sklearn's scale (tuned)
    train_days: int = 3 * 365     # training window
    max_goals: int = 10
    use_team_ratings: bool = True  # False = cross-league mode (Elo-driven)
    # Training target = (1-w)*goals + w*xG on matches with real xG (goals elsewhere).
    # xG is a less noisy measure of chance quality, so ratings settle faster.
    # 0.7 chosen on 2024/25, confirmed on held-out 2025/26 (O/U log-loss -0.004, z -2.2).
    xg_weight: float = 0.7
    manual_priors: dict = field(default_factory=lambda: dict(MANUAL_PRIORS))


@dataclass
class FlagConfig:
    # edge = model probability - bookmaker fair (de-margined) probability
    green: float = 0.0     # edge >= 0           -> GREEN
    yellow: float = -0.05  # -5% <= edge < 0     -> YELLOW
    orange: float = -0.10  # -10% <= edge < -5%  -> ORANGE, below -> RED
    odds_yellow_floor: float = 1.50  # v3 rule: odds > 1.50 are at least YELLOW
    default_overround: float = 1.06  # used when only one price is known
    max_parlay_legs: int = 6         # warn above this


FLAG_EMOJI = {"GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴", "NODATA": "⚪"}

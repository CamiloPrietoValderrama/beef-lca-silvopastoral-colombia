#!/usr/bin/env python3
"""
Scenario-Based Monte Carlo LCA of Beef Production in Colombia
==============================================================

Reproducible code for:
"Scenario-Based Life Cycle Assessment of Beef Production in Colombia:
 Intensive Silvopastoral Transition Scenarios under Uncertainty"

Primary functional unit
-----------------------
1 kg live-weight gain (LWG) at the farm gate.

Primary accounting layer
------------------------
Enteric CH4 + manure CH4 + soil N2O + aggregated input-related emissions.

Structural accounting layer
---------------------------
Land-use change (LUC), biomass-carbon accumulation, and soil-organic-carbon
accumulation are evaluated separately and are not intrinsic properties of
BAU-F, SSPi-M, or SSPi-I.

Reproducibility
---------------
Default Monte Carlo settings reproduce the manuscript:
    N = 10,000 accepted iterations per scenario
    random seed = 42

The scenario parameters correspond to Supplementary Table S1.

Authors
-------
Camilo Prieto Valderrama
Diego Patiño

Version
-------
Reviewer-1 harmonization revision candidate (450-kg UGG reference-animal basis).
This version changes numerical results compared with v1.0.0 and requires a NEW
archived Zenodo software version/DOI; do not overwrite the old release.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# 1. Global settings and methodological constants
# ---------------------------------------------------------------------

DEFAULT_N = 10_000
DEFAULT_SEED = 42

GWP_CH4 = 27.0
GWP_N2O = 273.0

# Livestock-unit harmonization (Reviewer 1, round 2):
# A Colombian UGG is 450 kg live mass.  The model represents one
# 450-kg *reference animal* per modeled UGG.  Thus SR is UGG/ha
# numerically equal to reference animals/ha under the explicit
# 450-kg/head simplifying assumption, NOT a field census of heads/ha.
# ADG and IPCC other-cattle enteric EF are assigned per reference animal.
# VS and N rates, reported per 1,000 kg of live animal mass, use 450 kg.
# Population mix and time-varying mean live weights are not resolved;
# this limitation is explicitly reported in manuscript and supplement.
UGG_KG_LIVE_MASS = 450.0
MASS_EQ_KG = UGG_KG_LIVE_MASS

# Manure deposited on pasture/range/paddock.
EF_MANURE_CH4 = 0.0006  # kg CH4 per kg VS

# Indirect N2O constants.
FRAC_GASM = 0.21
EF4 = 0.010
EF5 = 0.011

# Land-use change.
CEF_PASTURE = 73.79  # Mg C ha-1 converted
T_LUC = 20  # yr

# Maximum period over which an annual C-accumulation rate receives full credit.
T_CARBON_CREDIT = 20  # yr


# ---------------------------------------------------------------------
# 2. Scenario parameterization
# ---------------------------------------------------------------------
#
# Triangular distributions are represented as (minimum, mode, maximum).
# Units and source provenance are documented in Supplementary Table S1.
#

SCENARIOS = {
    "BAU-F": {
        "SR": (0.50, 0.85, 1.00),                 # 450-kg UGG ha-1
        "ADG": (0.25, 0.37, 0.50),                # kg LWG reference-animal-1 d-1
        "EFent_mode": 58.0,                        # kg CH4 reference-animal-1 yr-1
        "VS_rate_mode": 8.60,                      # kg VS (1000 kg LW)-1 d-1
        "Nex_rate_mode": 0.290,                    # kg N (1000 kg LW)-1 d-1
        "inputs": (50.0, 100.0, 200.0),            # kg CO2eq ha-1 yr-1
        "bioC": (0.0, 0.02, 0.08),                 # Mg C ha-1 yr-1
        "soilC": (0.0, 0.0, 0.0),                 # Mg C ha-1 yr-1
    },
    "SSPi-M": {
        "SR": (1.00, 1.50, 2.34),
        "ADG": (0.40, 0.60, 0.75),
        "EFent_mode": 56.0,
        "VS_rate_mode": 8.50,
        "Nex_rate_mode": 0.310,
        "inputs": (80.0, 180.0, 350.0),
        "bioC": (0.20, 0.60, 1.30),
        "soilC": (0.0, 0.50, 0.80),
    },
    "SSPi-I": {
        "SR": (2.71, 3.00, 4.00),
        "ADG": (0.417, 0.80, 0.836),
        "EFent_mode": 55.0,
        "VS_rate_mode": 8.10,
        "Nex_rate_mode": 0.360,
        "inputs": (150.0, 350.0, 700.0),
        "bioC": (0.80, 1.50, 2.50),
        "soilC": (0.0, 0.50, 0.80),
    },
}

# Accepted productivity envelopes, kg LWG ha-1 yr-1.
PRODUCTIVITY_ENVELOPES = {
    "BAU-F": (67.5, 182.5),
    "SSPi-M": (229.4, 657.9),
    "SSPi-I": (609.0, 1098.0),
}

# Conservative productivity sensitivity.
CONSERVATIVE_PRODUCTIVITY = {
    "BAU-F": {
        "SR": (0.50, 0.85, 1.00),
        "ADG": (0.25, 0.37, 0.50),
    },
    "SSPi-M": {
        "SR": (1.00, 1.25, 1.50),
        "ADG": (0.40, 0.50, 0.60),
    },
    "SSPi-I": {
        "SR": (2.71, 2.85, 3.00),
        "ADG": (0.417, 0.60, 0.75),
    },
}


# ---------------------------------------------------------------------
# 3. Sampling helpers
# ---------------------------------------------------------------------

def triangular(
    rng: np.random.Generator,
    values: tuple[float, float, float],
    n: int,
) -> np.ndarray:
    """Sample a triangular distribution, including the fixed-value case."""
    a, m, b = values
    if a == m == b:
        return np.full(n, a, dtype=float)
    return rng.triangular(a, m, b, n)


def sample_sr_adg_with_envelope(
    rng: np.random.Generator,
    scenario_parameters: dict,
    n: int,
    productivity_envelope: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jointly sample stocking rate and ADG and reject combinations whose derived
    productivity lies outside the evidence-supported scenario envelope.

    This is the explicit plausibility constraint used in the revised manuscript.
    SR is 450-kg UGG/ha; ADG denotes kg gain/(450-kg reference animal)/day.
    This simplification is NOT a census of actual heads; for animals of mean
    mass W kg, head count per ha would instead equal UGG/ha x (450/W).
    """
    sr_parts: list[np.ndarray] = []
    adg_parts: list[np.ndarray] = []
    accepted = 0

    while accepted < n:
        # Oversampling makes rejection efficient while preserving deterministic
        # reproduction for a fixed seed and NumPy RNG.
        batch = max(1000, (n - accepted) * 3)

        sr = rng.triangular(*scenario_parameters["SR"], batch)
        adg = rng.triangular(*scenario_parameters["ADG"], batch)
        productivity = sr * adg * 365.0

        lo, hi = productivity_envelope
        keep = (productivity >= lo) & (productivity <= hi)

        sr = sr[keep]
        adg = adg[keep]

        take = min(n - accepted, len(sr))
        sr_parts.append(sr[:take])
        adg_parts.append(adg[:take])
        accepted += take

    return np.concatenate(sr_parts), np.concatenate(adg_parts)


# ---------------------------------------------------------------------
# 4. Primary Monte Carlo model
# ---------------------------------------------------------------------

def simulate(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    climate: str = "base",
    productivity_override: dict | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run the production-emissions Monte Carlo model.

    climate:
        "base" -> EF3 = 0.004; FracLEACH = 0
        "dry"  -> EF3 = 0.002; FracLEACH = 0
        "wet"  -> EF3 = 0.006; FracLEACH = 0.24

    productivity_override:
        Optional dict replacing only SR and ADG distributions for sensitivity
        analyses while leaving emission-factor distributions unchanged.
    """
    if climate not in {"base", "dry", "wet"}:
        raise ValueError("climate must be one of: 'base', 'dry', 'wet'")

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, np.ndarray]] = {}

    for scenario, base_parameters in SCENARIOS.items():
        p = dict(base_parameters)

        if productivity_override and scenario in productivity_override:
            p["SR"] = productivity_override[scenario]["SR"]
            p["ADG"] = productivity_override[scenario]["ADG"]

        sr, adg = sample_sr_adg_with_envelope(
            rng=rng,
            scenario_parameters=p,
            n=n,
            productivity_envelope=PRODUCTIVITY_ENVELOPES[scenario],
        )

        productivity = sr * adg * 365.0

        # Enteric methane EF: +/-30% triangular uncertainty around central value.
        ef_ent = triangular(
            rng,
            (
                p["EFent_mode"] * 0.70,
                p["EFent_mode"],
                p["EFent_mode"] * 1.30,
            ),
            n,
        )

        # VS and N excretion: +/-20% triangular uncertainty.
        vs_rate = triangular(
            rng,
            (
                p["VS_rate_mode"] * 0.80,
                p["VS_rate_mode"],
                p["VS_rate_mode"] * 1.20,
            ),
            n,
        )
        nex_rate = triangular(
            rng,
            (
                p["Nex_rate_mode"] * 0.80,
                p["Nex_rate_mode"],
                p["Nex_rate_mode"] * 1.20,
            ),
            n,
        )

        input_emissions = triangular(rng, p["inputs"], n)

        # Convert excretion rates from per 1000 kg live mass per day to
        # annual quantities per model animal-equivalent.
        vs_annual = vs_rate * (MASS_EQ_KG / 1000.0) * 365.0
        nex_annual = nex_rate * (MASS_EQ_KG / 1000.0) * 365.0

        # Enteric CH4, kg CO2eq ha-1 yr-1.
        e_enteric = sr * ef_ent * GWP_CH4

        # Manure CH4 deposited on pasture/range/paddock.
        e_manure = sr * vs_annual * EF_MANURE_CH4 * GWP_CH4

        # Soil N2O from deposited excreted N.
        n_deposited = sr * nex_annual

        if climate == "base":
            ef3 = 0.004
            frac_leach = 0.0
        elif climate == "dry":
            ef3 = 0.002
            frac_leach = 0.0
        else:  # wet
            ef3 = 0.006
            frac_leach = 0.24

        n2o_n = n_deposited * (
            ef3
            + FRAC_GASM * EF4
            + frac_leach * EF5
        )
        e_n2o = n2o_n * (44.0 / 28.0) * GWP_N2O

        # Primary production-related emissions.
        e_prod = e_enteric + e_manure + e_n2o + input_emissions
        cf_prod = e_prod / productivity

        # Carbon accumulation draws are generated here so every structural
        # accounting configuration reuses the same accepted production draws.
        biomass_removal = (
            triangular(rng, p["bioC"], n) * (44.0 / 12.0) * 1000.0
        )
        soil_removal = (
            triangular(rng, p["soilC"], n) * (44.0 / 12.0) * 1000.0
        )

        results[scenario] = {
            "SR": sr,
            "ADG": adg,
            "Y": productivity,
            "EFent": ef_ent,
            "VSrate": vs_rate,
            "Nexrate": nex_rate,
            "inputs": input_emissions,
            "enteric": e_enteric,
            "manure": e_manure,
            "n2o": e_n2o,
            "Eprod": e_prod,
            "CFprod": cf_prod,
            "bio": biomass_removal,
            "soil": soil_removal,
        }

    return results


# ---------------------------------------------------------------------
# 5. Structural LUC and carbon-accounting configurations
# ---------------------------------------------------------------------

def annual_luc_burden(f_luc: float) -> float:
    """Annualized LUC burden, kg CO2eq ha-1 yr-1."""
    return (
        f_luc
        * CEF_PASTURE
        * (44.0 / 12.0)
        * 1000.0
        / T_LUC
    )


def structural_configuration(
    primary_results: dict[str, dict[str, np.ndarray]],
    f_luc: float = 0.0,
    lambda_bio: float = 0.0,
    lambda_soil: float = 0.0,
    horizon: int = 20,
) -> dict[str, dict[str, float | np.ndarray]]:
    """
    Apply one structural land-carbon accounting configuration to the same
    primary Monte Carlo draws (common-random-number design).
    """
    omega = min(horizon, T_CARBON_CREDIT) / horizon
    luc = annual_luc_burden(f_luc)
    out: dict[str, dict[str, float | np.ndarray]] = {}

    for scenario, d in primary_results.items():
        cf_net = (
            d["Eprod"]
            + luc
            - lambda_bio * d["bio"] * omega
            - lambda_soil * d["soil"] * omega
        ) / d["Y"]

        out[scenario] = {
            "CFnet": cf_net,
            "mean": float(np.mean(cf_net)),
            "median": float(np.median(cf_net)),
            "p5": float(np.quantile(cf_net, 0.05)),
            "p95": float(np.quantile(cf_net, 0.95)),
            "min": float(np.min(cf_net)),
            "max": float(np.max(cf_net)),
            "pr_negative": float(np.mean(cf_net < 0) * 100.0),
        }

    return out


STRUCTURAL_CONFIGURATIONS = [
    ("Primary production-only", 0.00, 0.00, 0.00, 20),
    ("Common LUC 5%",          0.05, 0.00, 0.00, 20),
    ("Common LUC 10%",         0.10, 0.00, 0.00, 20),
    ("Common LUC 30%",         0.30, 0.00, 0.00, 20),
    ("Sequestration credit 25%",  0.00, 0.25, 0.25, 20),
    ("Sequestration credit 50%",  0.00, 0.50, 0.50, 20),
    ("Sequestration credit 100%", 0.00, 1.00, 1.00, 20),
    ("Biomass only (100%)",       0.00, 1.00, 0.00, 20),
    ("Soil only (100%)",          0.00, 0.00, 1.00, 20),
    ("Full sequestration, 30-y horizon", 0.00, 1.00, 1.00, 30),
]


# ---------------------------------------------------------------------
# 6. Summary statistics
# ---------------------------------------------------------------------

def primary_summary(
    results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    bau_mean = float(np.mean(results["BAU-F"]["CFprod"]))

    for scenario, d in results.items():
        cf = d["CFprod"]
        mean = float(np.mean(cf))
        rows.append(
            {
                "Scenario": scenario,
                "Mean": mean,
                "Median": float(np.median(cf)),
                "P5": float(np.quantile(cf, 0.05)),
                "P95": float(np.quantile(cf, 0.95)),
                "Min": float(np.min(cf)),
                "Max": float(np.max(cf)),
                "Reduction_vs_BAU_percent": 100.0 * (bau_mean - mean) / bau_mean,
            }
        )
    return pd.DataFrame(rows)


def component_summary(
    results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for scenario, d in results.items():
        y = d["Y"]
        rows.append(
            {
                "Scenario": scenario,
                "Enteric_CF": float(np.mean(d["enteric"] / y)),
                "Manure_CF": float(np.mean(d["manure"] / y)),
                "Soil_N2O_CF": float(np.mean(d["n2o"] / y)),
                "Inputs_CF": float(np.mean(d["inputs"] / y)),
                "Input_emissions_ha": float(np.mean(d["inputs"])),
                "Gross_emissions_ha": float(np.mean(d["Eprod"])),
                "Productivity": float(np.mean(y)),
                "Land_requirement": float(np.mean(1000.0 / y)),
                "Total_CF": float(np.mean(d["CFprod"])),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 7. Spearman rank-based sensitivity screening
# ---------------------------------------------------------------------

def spearman_sensitivity(
    results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    mapping = [
        ("Stocking rate", "SR"),
        ("Average daily gain", "ADG"),
        ("Enteric EF", "EFent"),
        ("VS excretion", "VSrate"),
        ("N excretion", "Nexrate"),
        ("Input emissions", "inputs"),
    ]

    rows = []
    for scenario, d in results.items():
        for label, key in mapping:
            rho = spearmanr(d[key], d["CFprod"]).statistic
            rows.append(
                {
                    "Parameter": label,
                    "Scenario": scenario,
                    "Spearman_rho": float(rho),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 8. Order-invariant Shapley mechanism decomposition
# ---------------------------------------------------------------------

SHAPLEY_BLOCKS = ["Productivity", "Methane", "N2O", "Inputs"]


def shapley_decomposition(
    results: dict[str, dict[str, np.ndarray]],
    target: str,
) -> dict[str, float]:
    """
    Order-invariant accounting decomposition of the mean difference between
    BAU-F and one SSPi scenario.

    This is an accounting attribution, not a causal decomposition.
    """
    if target not in {"SSPi-M", "SSPi-I"}:
        raise ValueError("target must be 'SSPi-M' or 'SSPi-I'")

    bau = results["BAU-F"]
    tar = results[target]

    def hybrid_mean_cf(active_blocks: set[str]) -> float:
        y = tar["Y"] if "Productivity" in active_blocks else bau["Y"]
        methane = (
            tar["enteric"] + tar["manure"]
            if "Methane" in active_blocks
            else bau["enteric"] + bau["manure"]
        )
        n2o = tar["n2o"] if "N2O" in active_blocks else bau["n2o"]
        inp = tar["inputs"] if "Inputs" in active_blocks else bau["inputs"]
        return float(np.mean((methane + n2o + inp) / y))

    base = hybrid_mean_cf(set())

    def value(active_blocks: set[str]) -> float:
        # Positive value = reduction relative to BAU-F.
        return base - hybrid_mean_cf(active_blocks)

    m = len(SHAPLEY_BLOCKS)
    phi: dict[str, float] = {}

    for block in SHAPLEY_BLOCKS:
        total = 0.0
        others = [b for b in SHAPLEY_BLOCKS if b != block]

        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                a = set(subset)
                weight = (
                    math.factorial(len(a))
                    * math.factorial(m - len(a) - 1)
                    / math.factorial(m)
                )
                total += weight * (
                    value(a | {block}) - value(a)
                )

        phi[block] = float(total)

    phi["Net difference"] = float(
        base - hybrid_mean_cf(set(SHAPLEY_BLOCKS))
    )
    return phi


def shapley_table(
    results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    m = shapley_decomposition(results, "SSPi-M")
    i = shapley_decomposition(results, "SSPi-I")
    rows = []
    for block in SHAPLEY_BLOCKS + ["Net difference"]:
        rows.append(
            {
                "Mechanism": block,
                "SSPi-M": m[block],
                "SSPi-I": i[block],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 9. Structural robustness table
# ---------------------------------------------------------------------

def structural_table(
    primary_results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for name, f_luc, lb, ls, horizon in STRUCTURAL_CONFIGURATIONS:
        r = structural_configuration(
            primary_results,
            f_luc=f_luc,
            lambda_bio=lb,
            lambda_soil=ls,
            horizon=horizon,
        )
        rows.append(
            {
                "Configuration": name,
                "BAU-F": r["BAU-F"]["mean"],
                "SSPi-M": r["SSPi-M"]["mean"],
                "SSPi-I": r["SSPi-I"]["mean"],
                "Pr_CF_lt_0_SSPi-M_percent": r["SSPi-M"]["pr_negative"],
                "Pr_CF_lt_0_SSPi-I_percent": r["SSPi-I"]["pr_negative"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 10. Convergence and Monte Carlo standard error
# ---------------------------------------------------------------------

def convergence_table(
    seed: int = DEFAULT_SEED,
    sample_sizes: tuple[int, ...] = (1000, 5000, 10000, 20000),
) -> pd.DataFrame:
    rows = []

    for n in sample_sizes:
        r = simulate(n=n, seed=seed, climate="base")
        row = {"N": n}

        for scenario, prefix in [
            ("BAU-F", "BAU"),
            ("SSPi-M", "M"),
            ("SSPi-I", "I"),
        ]:
            cf = r[scenario]["CFprod"]
            row[f"{prefix}_mean"] = float(np.mean(cf))
            row[f"{prefix}_P5"] = float(np.quantile(cf, 0.05))
            row[f"{prefix}_P95"] = float(np.quantile(cf, 0.95))
            row[f"{prefix}_MCSE"] = float(
                np.std(cf, ddof=1) / np.sqrt(n)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 11. Climate-specific N2O and conservative-productivity sensitivity
# ---------------------------------------------------------------------

def climate_sensitivity_table(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rows = []
    for climate in ["dry", "base", "wet"]:
        r = simulate(n=n, seed=seed, climate=climate)
        rows.append(
            {
                "Climate_configuration": climate,
                "BAU-F": float(np.mean(r["BAU-F"]["CFprod"])),
                "SSPi-M": float(np.mean(r["SSPi-M"]["CFprod"])),
                "SSPi-I": float(np.mean(r["SSPi-I"]["CFprod"])),
            }
        )
    return pd.DataFrame(rows)


def conservative_productivity_table(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    r = simulate(
        n=n,
        seed=seed,
        climate="base",
        productivity_override=CONSERVATIVE_PRODUCTIVITY,
    )

    bau = float(np.mean(r["BAU-F"]["CFprod"]))
    rows = []

    for scenario in SCENARIOS:
        mean_cf = float(np.mean(r[scenario]["CFprod"]))
        rows.append(
            {
                "Scenario": scenario,
                "Mean_CF": mean_cf,
                "Mean_productivity": float(np.mean(r[scenario]["Y"])),
                "Reduction_vs_BAU_percent": (
                    100.0 * (bau - mean_cf) / bau
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 12. Optional accepted-draw export
# ---------------------------------------------------------------------

def accepted_draws_dataframe(
    results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    frames = []
    for scenario, d in results.items():
        frame = pd.DataFrame(
            {
                "scenario": scenario,
                "SR": d["SR"],
                "ADG": d["ADG"],
                "productivity_LWG_ha_yr": d["Y"],
                "enteric_EF_CH4": d["EFent"],
                "VS_rate": d["VSrate"],
                "N_excretion_rate": d["Nexrate"],
                "input_emissions_ha": d["inputs"],
                "enteric_CO2eq_ha": d["enteric"],
                "manure_CO2eq_ha": d["manure"],
                "soil_N2O_CO2eq_ha": d["n2o"],
                "production_emissions_CO2eq_ha": d["Eprod"],
                "CFprod_CO2eq_per_kg_LWG": d["CFprod"],
                "biomass_removal_CO2eq_ha": d["bio"],
                "soil_removal_CO2eq_ha": d["soil"],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# 13. Figures corresponding to the manuscript analyses
# ---------------------------------------------------------------------

def make_figures(
    results: dict[str, dict[str, np.ndarray]],
    sensitivity: pd.DataFrame,
    shapley: pd.DataFrame,
    structural: pd.DataFrame,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"

    scenario_order = ["BAU-F", "SSPi-M", "SSPi-I"]

    # Figure 2: primary distributions.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(
        [results[s]["CFprod"] for s in scenario_order],
        tick_labels=scenario_order,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.5},
    )
    for patch, color in zip(
        bp["boxes"],
        ["#8C8C8C", "#4C78A8", "#59A14F"],
    ):
        patch.set_facecolor(color)
    ax.set_xlabel("Production scenario")
    ax.set_ylabel(r"GHG intensity (kg CO$_2$eq kg$^{-1}$ LWG)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir / "Figure_2_primary_distributions.png", dpi=300)
    plt.close(fig)

    # Figure 3: territorial emissions vs product-level intensity.
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    point_colors = {
        "BAU-F": "#8C8C8C",
        "SSPi-M": "#4C78A8",
        "SSPi-I": "#59A14F",
    }
    for s in scenario_order:
        x = float(np.mean(results[s]["Eprod"]))
        y = float(np.mean(results[s]["CFprod"]))
        ax.scatter(x, y, s=70, color=point_colors[s], zorder=3)
        ax.annotate(
            s,
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set_xlabel(
        r"Mean production-related emissions (kg CO$_2$eq ha$^{-1}$ yr$^{-1}$)"
    )
    ax.set_ylabel(
        r"Mean GHG intensity (kg CO$_2$eq kg$^{-1}$ LWG)"
    )
    ax.grid(True, color="#D7EAF7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "Figure_3_territorial_vs_product.png", dpi=300)
    plt.close(fig)

    # Figure 4: Spearman heatmap.
    order = [
        "Average daily gain",
        "Enteric EF",
        "Input emissions",
        "Stocking rate",
        "N excretion",
        "VS excretion",
    ]
    pivot = (
        sensitivity.pivot(
            index="Parameter",
            columns="Scenario",
            values="Spearman_rho",
        )
        .loc[order, scenario_order]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    im = ax.imshow(
        pivot.values,
        cmap="viridis",
        vmin=-0.8,
        vmax=0.8,
        aspect="auto",
    )
    ax.set_xticks(range(len(scenario_order)), scenario_order)
    ax.set_yticks(range(len(order)), order)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            rgba = im.cmap(im.norm(value))
            luminance = (
                0.2126 * rgba[0]
                + 0.7152 * rgba[1]
                + 0.0722 * rgba[2]
            )
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if luminance < 0.45 else "black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Spearman $\rho$")
    fig.tight_layout()
    fig.savefig(outdir / "Figure_4_spearman_sensitivity.png", dpi=300)
    plt.close(fig)

    # Figure 5: Shapley.
    blocks = ["Productivity", "Methane", "N2O", "Inputs"]
    s = shapley.set_index("Mechanism")
    x = np.arange(len(blocks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(
        x - width / 2,
        [s.loc[b, "SSPi-M"] for b in blocks],
        width,
        label="SSPi-M vs BAU-F",
    )
    ax.bar(
        x + width / 2,
        [s.loc[b, "SSPi-I"] for b in blocks],
        width,
        label="SSPi-I vs BAU-F",
    )
    ax.axhline(0, linewidth=1.0)
    ax.set_xticks(x, ["Productivity", "Methane", r"N$_2$O", "Inputs"])
    ax.set_ylabel(
        r"Shapley contribution (kg CO$_2$eq kg$^{-1}$ LWG)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "Figure_5_shapley.png", dpi=300)
    plt.close(fig)

    # Figure 6: selected structural configurations.
    selected_names = [
        "Primary production-only",
        "Common LUC 10%",
        "Common LUC 30%",
        "Sequestration credit 25%",
        "Sequestration credit 50%",
        "Sequestration credit 100%",
    ]
    sel = structural[
        structural["Configuration"].isin(selected_names)
    ].set_index("Configuration").loc[selected_names].reset_index()

    x = np.arange(len(sel))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for idx, scenario in enumerate(scenario_order):
        ax.bar(
            x + (idx - 1) * width,
            sel[scenario],
            width,
            label=scenario,
        )
    ax.axhline(0, linewidth=1.0)
    ax.set_xticks(
        x,
        sel["Configuration"],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel(
        r"Mean net footprint (kg CO$_2$eq kg$^{-1}$ LWG)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "Figure_6_structural.png", dpi=300)
    plt.close(fig)

    # Figure 7: model BAU-F vs Colombian empirical benchmarks.
    primary = primary_summary(results)
    bau = primary.loc[primary["Scenario"] == "BAU-F"].iloc[0]

    labels = ["Model BAU-F", "Empirical cow-calf", "Empirical fattening"]
    means = np.array([bau["Mean"], 12.37, 13.85])
    lower = np.array([bau["P5"], 10.30, 9.90])
    upper = np.array([bau["P95"], 14.50, 18.70])
    xerr = np.vstack([means - lower, upper - means])
    ypos = np.array([2, 1, 0])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for xv in [10, 12, 14, 16, 18]:
        ax.axvline(
            xv,
            color="#D3D3D3",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )
    ax.errorbar(
        means,
        ypos,
        xerr=xerr,
        fmt="none",
        ecolor="#2C7FB8",
        elinewidth=1.4,
        capsize=4,
        capthick=1.2,
        zorder=2,
    )
    ax.scatter(
        means,
        ypos,
        s=44,
        color="#7B3294",
        zorder=3,
    )
    ax.set_yticks(ypos, labels)
    ax.set_xlabel(r"GHG intensity (kg CO$_2$eq kg$^{-1}$ LWG)")
    ax.set_xlim(9.5, 19.1)
    fig.tight_layout()
    fig.savefig(outdir / "Figure_7_empirical_benchmark.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 14. Manuscript-value verification
# ---------------------------------------------------------------------

EXPECTED_PRIMARY_MEANS = {
    "BAU-F": 13.879362400397280,
    "SSPi-M": 8.488935878380618,
    "SSPi-I": 7.103372545183964,
}


def verify_manuscript_outputs(
    results: dict[str, dict[str, np.ndarray]],
    tolerance: float = 5e-10,
) -> None:
    """
    Verify that default N=10,000 / seed=42 reproduces the archived manuscript
    primary means. Raises AssertionError if it does not.
    """
    for scenario, expected in EXPECTED_PRIMARY_MEANS.items():
        observed = float(np.mean(results[scenario]["CFprod"]))
        if not np.isclose(observed, expected, atol=tolerance, rtol=0):
            raise AssertionError(
                f"{scenario}: observed {observed:.15f}, "
                f"expected {expected:.15f}"
            )


# ---------------------------------------------------------------------
# 15. Export and command-line entry point
# ---------------------------------------------------------------------

def export_all(
    outdir: Path,
    n: int,
    seed: int,
    save_draws: bool,
    make_plots: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    results = simulate(n=n, seed=seed, climate="base")

    df_primary = primary_summary(results)
    df_components = component_summary(results)
    df_sensitivity = spearman_sensitivity(results)
    df_shapley = shapley_table(results)
    df_structural = structural_table(results)
    df_convergence = convergence_table(seed=seed)
    df_climate = climate_sensitivity_table(n=n, seed=seed)
    df_conservative = conservative_productivity_table(n=n, seed=seed)

    df_primary.to_csv(outdir / "primary_summary.csv", index=False)
    df_components.to_csv(outdir / "component_summary.csv", index=False)
    df_sensitivity.to_csv(outdir / "sensitivity_spearman.csv", index=False)
    df_shapley.to_csv(outdir / "shapley_decomposition.csv", index=False)
    df_structural.to_csv(outdir / "structural_sensitivity.csv", index=False)
    df_convergence.to_csv(outdir / "convergence.csv", index=False)
    df_climate.to_csv(outdir / "climate_sensitivity.csv", index=False)
    df_conservative.to_csv(
        outdir / "conservative_productivity.csv",
        index=False,
    )

    if save_draws:
        accepted_draws_dataframe(results).to_csv(
            outdir / "accepted_monte_carlo_draws.csv",
            index=False,
        )

    if make_plots:
        make_figures(
            results=results,
            sensitivity=df_sensitivity,
            shapley=df_shapley,
            structural=df_structural,
            outdir=outdir / "figures",
        )

    # Exact revised-manuscript verification applies to the stated 450-kg basis.
    if n == DEFAULT_N and seed == DEFAULT_SEED:
        verify_manuscript_outputs(results)

    print("\nPrimary Monte Carlo results")
    print(df_primary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nMean component / territorial results")
    print(
        df_components.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nOutputs written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Colombian beef SSPi Monte Carlo LCA."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV outputs and figures (default: results).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Accepted draws per scenario (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--save-draws",
        action="store_true",
        help="Also export all accepted Monte Carlo draws to CSV.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_all(
        outdir=args.outdir,
        n=args.n,
        seed=args.seed,
        save_draws=args.save_draws,
        make_plots=not args.no_figures,
    )


if __name__ == "__main__":
    main()

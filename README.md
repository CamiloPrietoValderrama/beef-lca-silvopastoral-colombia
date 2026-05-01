"""
Monte Carlo Life Cycle Assessment model for beef production systems in Colombia.

This script estimates the net carbon footprint of beef production under three
production scenarios:

1. BAU extensive cattle production
2. SSPi moderate: moderate intensive silvopastoral system
3. SSPi intensive: intensive silvopastoral system

Functional unit:
    1 kg of beef, expressed as carcass-weight equivalent.

System boundary:
    Cradle-to-farm gate.

Main components:
    - Enteric methane
    - Manure methane
    - Soil nitrous oxide
    - Input-related carbon dioxide
    - Land-use change emissions
    - Biomass carbon sequestration
    - Soil carbon sequestration

Units:
    - Productivity: kg beef/ha/year
    - Emission and removal components: kg CO2eq/ha/year
    - Net carbon footprint: kg CO2eq/kg beef

Author:
    Camilo Prieto Valderrama et al.

Note:
    This version uses aggregated component-level parameters consistent with the
    manuscript methodology. Scenario ranges should be updated as improved field
    data and peer-reviewed parameter estimates become available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Global settings
# ============================================================

N_ITERATIONS = 10_000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Helper functions
# ============================================================

def triangular_sample(minimum: float, mode: float, maximum: float, size: int) -> np.ndarray:
    """
    Draw random samples from a triangular distribution.

    Parameters
    ----------
    minimum : float
        Minimum plausible value.
    mode : float
        Most likely value.
    maximum : float
        Maximum plausible value.
    size : int
        Number of Monte Carlo iterations.

    Returns
    -------
    np.ndarray
        Random samples.
    """
    return np.random.triangular(minimum, mode, maximum, size)


def summarize_distribution(values: pd.Series) -> pd.Series:
    """
    Generate summary statistics for a Monte Carlo output distribution.

    Parameters
    ----------
    values : pd.Series
        Net carbon footprint values.

    Returns
    -------
    pd.Series
        Summary statistics.
    """
    return pd.Series(
        {
            "mean": values.mean(),
            "median": values.median(),
            "p5": np.percentile(values, 5),
            "p25": np.percentile(values, 25),
            "p75": np.percentile(values, 75),
            "p95": np.percentile(values, 95),
            "minimum": values.min(),
            "maximum": values.max(),
            "probability_CF_below_0": np.mean(values < 0),
            "probability_CF_below_30": np.mean(values < 30),
        }
    )


# ============================================================
# 3. Scenario parameterization
# ============================================================

"""
All parameters are expressed per hectare per year, except productivity.

Y:
    kg beef/ha/year

Emission components:
    kg CO2eq/ha/year

Carbon sequestration components:
    kg CO2eq/ha/year

The values below reproduce the first scenario-based simulation used in the
manuscript development. They should be treated as transparent, editable
model inputs.
"""

SCENARIOS = {
    "BAU extensive": {
        "Y": (70, 110, 170),
        "enteric_CH4": (3600, 5000, 6500),
        "manure_CH4": (150, 250, 400),
        "soil_N2O": (300, 600, 1000),
        "inputs_CO2": (50, 100, 200),
        "LUC": (0, 1500, 4500),
        "Cseq_biomass": (0, 50, 150),
        "Cseq_soil": (0, 50, 150),
    },
    "SSPi moderate": {
        "Y": (250, 400, 650),
        "enteric_CH4": (5000, 6800, 8500),
        "manure_CH4": (250, 500, 850),
        "soil_N2O": (400, 750, 1200),
        "inputs_CO2": (80, 180, 350),
        "LUC": (0, 150, 600),
        "Cseq_biomass": (800, 2000, 3500),
        "Cseq_soil": (500, 1500, 3000),
    },
    "SSPi intensive": {
        "Y": (600, 900, 1300),
        "enteric_CH4": (8000, 11500, 15000),
        "manure_CH4": (500, 1000, 1700),
        "soil_N2O": (700, 1300, 2200),
        "inputs_CO2": (150, 350, 700),
        "LUC": (0, 0, 200),
        "Cseq_biomass": (2500, 5500, 9000),
        "Cseq_soil": (1500, 4000, 7500),
    },
}


# ============================================================
# 4. Monte Carlo simulation
# ============================================================

def run_monte_carlo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the Monte Carlo simulation for all scenarios.

    Returns
    -------
    results : pd.DataFrame
        Full Monte Carlo draws and calculated outputs.
    summary : pd.DataFrame
        Summary statistics by scenario.
    components : pd.DataFrame
        Mean component contribution by scenario in kg CO2eq/kg beef.
    sensitivity : pd.DataFrame
        Spearman correlation between inputs and net carbon footprint.
    """

    scenario_results = []
    component_records = []
    sensitivity_records = []

    for scenario_name, params in SCENARIOS.items():
        draws = {
            parameter: triangular_sample(*values, size=N_ITERATIONS)
            for parameter, values in params.items()
        }

        gross_emissions = (
            draws["enteric_CH4"]
            + draws["manure_CH4"]
            + draws["soil_N2O"]
            + draws["inputs_CO2"]
            + draws["LUC"]
        )

        carbon_sequestration = draws["Cseq_biomass"] + draws["Cseq_soil"]

        net_emissions_per_ha = gross_emissions - carbon_sequestration

        carbon_footprint = net_emissions_per_ha / draws["Y"]

        df = pd.DataFrame(draws)
        df["gross_emissions"] = gross_emissions
        df["carbon_sequestration"] = carbon_sequestration
        df["net_emissions_per_ha"] = net_emissions_per_ha
        df["CF"] = carbon_footprint
        df["scenario"] = scenario_name

        scenario_results.append(df)

        component_records.append(
            {
                "scenario": scenario_name,
                "enteric_CH4": np.mean(draws["enteric_CH4"] / draws["Y"]),
                "manure_CH4": np.mean(draws["manure_CH4"] / draws["Y"]),
                "soil_N2O": np.mean(draws["soil_N2O"] / draws["Y"]),
                "inputs_CO2": np.mean(draws["inputs_CO2"] / draws["Y"]),
                "LUC": np.mean(draws["LUC"] / draws["Y"]),
                "Cseq_biomass": -np.mean(draws["Cseq_biomass"] / draws["Y"]),
                "Cseq_soil": -np.mean(draws["Cseq_soil"] / draws["Y"]),
                "net_CF": np.mean(carbon_footprint),
            }
        )

    results = pd.concat(scenario_results, ignore_index=True)

    summary = (
        results.groupby("scenario")["CF"]
        .apply(summarize_distribution)
        .unstack()
        .reset_index()
    )

    components = pd.DataFrame(component_records)

    input_variables = [
        "Y",
        "enteric_CH4",
        "manure_CH4",
        "soil_N2O",
        "inputs_CO2",
        "LUC",
        "Cseq_biomass",
        "Cseq_soil",
    ]

    for scenario_name in SCENARIOS:
        scenario_df = results[results["scenario"] == scenario_name]

        for variable in input_variables:
            rho = scenario_df[[variable, "CF"]].corr(method="spearman").iloc[0, 1]
            sensitivity_records.append(
                {
                    "scenario": scenario_name,
                    "variable": variable,
                    "spearman_rho": rho,
                }
            )

    sensitivity = pd.DataFrame(sensitivity_records)

    return results, summary, components, sensitivity


# ============================================================
# 5. Plotting functions
# ============================================================

def plot_boxplot(results: pd.DataFrame) -> None:
    """
    Generate boxplot of net carbon footprint by scenario.
    """
    scenario_order = list(SCENARIOS.keys())
    data = [results.loc[results["scenario"] == scenario, "CF"] for scenario in scenario_order]

    plt.figure(figsize=(7.2, 4.8))
    plt.boxplot(data, tick_labels=scenario_order, showfliers=False)
    plt.axhline(0, linestyle=":", linewidth=1)
    plt.ylabel("Net carbon footprint (kg CO$_2$eq/kg beef)")
    plt.xlabel("Production scenario")
    plt.title("Net carbon footprint by production scenario")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_2_net_carbon_footprint_boxplot.png", dpi=300)
    plt.close()


def plot_histograms(results: pd.DataFrame) -> None:
    """
    Generate probability density histograms by scenario.
    """
    plt.figure(figsize=(7.2, 4.8))

    for scenario_name in SCENARIOS:
        plt.hist(
            results.loc[results["scenario"] == scenario_name, "CF"],
            bins=60,
            alpha=0.45,
            density=True,
            label=scenario_name,
        )

    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Net carbon footprint (kg CO$_2$eq/kg beef)")
    plt.ylabel("Probability density")
    plt.title("Monte Carlo distributions of net carbon footprint")
    plt.legend()
    plt.grid(axis="y", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_3_monte_carlo_distributions.png", dpi=300)
    plt.close()


def plot_component_contributions(components: pd.DataFrame) -> None:
    """
    Generate component contribution bar chart.

    Positive values represent emissions.
    Negative values represent carbon removals.
    """
    component_columns = [
        "enteric_CH4",
        "manure_CH4",
        "soil_N2O",
        "inputs_CO2",
        "LUC",
        "Cseq_biomass",
        "Cseq_soil",
    ]

    labels = components["scenario"].tolist()
    x = np.arange(len(labels))

    positive_bottom = np.zeros(len(labels))
    negative_bottom = np.zeros(len(labels))

    plt.figure(figsize=(8.2, 5.2))

    for column in component_columns:
        values = components[column].values

        if np.all(values >= 0):
            plt.bar(x, values, bottom=positive_bottom, label=column)
            positive_bottom += values
        else:
            plt.bar(x, values, bottom=negative_bottom, label=column)
            negative_bottom += values

    plt.axhline(0, linewidth=1)
    plt.ylabel("Mean contribution (kg CO$_2$eq/kg beef)")
    plt.xlabel("Production scenario")
    plt.title("Mean emission and sequestration contributions")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(axis="y", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_component_contributions.png", dpi=300)
    plt.close()


def plot_sensitivity(sensitivity: pd.DataFrame, scenario: str = "SSPi intensive") -> None:
    """
    Generate sensitivity bar chart for the selected scenario.
    """
    subset = sensitivity[sensitivity["scenario"] == scenario].copy()
    subset["abs_rho"] = subset["spearman_rho"].abs()
    subset = subset.sort_values("abs_rho", ascending=True)

    colors = [
        "#9cc9ee" if value < 0 else "#003f7f"
        for value in subset["spearman_rho"]
    ]

    plt.figure(figsize=(7.2, 5.0))
    plt.barh(subset["variable"], subset["spearman_rho"], color=colors)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Spearman correlation with net carbon footprint")
    plt.title("Sensitivity analysis: SSPi intensive scenario")
    plt.grid(axis="x", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_5_sensitivity_sspi_intensive.png", dpi=300)
    plt.close()


# ============================================================
# 6. Main execution
# ============================================================

def main() -> None:
    """
    Execute the simulation, save outputs, and generate figures.
    """
    results, summary, components, sensitivity = run_monte_carlo()

    results.to_csv(RESULTS_DIR / "monte_carlo_draws.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_statistics.csv", index=False)
    components.to_csv(RESULTS_DIR / "component_contributions.csv", index=False)
    sensitivity.to_csv(RESULTS_DIR / "sensitivity_spearman.csv", index=False)

    plot_boxplot(results)
    plot_histograms(results)
    plot_component_contributions(components)
    plot_sensitivity(sensitivity)

    print("\nMonte Carlo simulation completed successfully.\n")
    print("Summary statistics:\n")
    print(summary.round(4))
    print("\nOutputs saved in:")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

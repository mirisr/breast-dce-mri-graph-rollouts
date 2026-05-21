#!/usr/bin/env python3
"""Build the supplementary-material PDF source from paper-facing tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT.parent / "results"


def esc(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    try:
        number = float(value)
    except Exception:
        return esc(value)
    if abs(number) <= 1 and number != 0:
        return f"{number:.3f}"
    return f"{number:.{digits}f}"


def table_block(
    df: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    caption: str,
    label: str,
    *,
    landscape: bool = False,
    font: str = r"\scriptsize",
) -> str:
    spec = "".join(kind for _, _, kind in columns)
    header = " & ".join(esc(title) for _, title, _ in columns) + r" \\"
    lines: list[str] = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend(
        [
            font,
            rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}",
            rf"\caption{{{esc(caption)}}}\label{{{label}}}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            header,
            r"\midrule",
            r"\endhead",
        ]
    )
    for _, row in df.iterrows():
        cells: list[str] = []
        for col, _, kind in columns:
            value = row[col]
            cells.append(fmt(value) if kind == "r" else esc(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def select_rows(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    out = df[df[column].isin(values)].copy()
    out[column] = pd.Categorical(out[column], categories=values, ordered=True)
    return out.sort_values(column).reset_index(drop=True)


def pit_deciles() -> pd.DataFrame:
    path = TABLES / "pit_t0t3.csv"
    pit = pd.read_csv(path)
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(10)]
    pit["pit_bin"] = pd.cut(
        pit["pit"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    grouped = (
        pit.groupby(["model", "bucket", "pit_bin"], observed=False)
        .size()
        .reset_index(name="n")
    )
    totals = grouped.groupby(["model", "bucket"], observed=False)["n"].transform("sum")
    grouped["fraction"] = grouped["n"] / totals
    out = grouped[grouped["bucket"].eq("T0->T3")].reset_index(drop=True)
    out.to_csv(TABLES / "pit_t0t3_decile_summary.csv", index=False)
    return out


def main() -> int:
    sections: list[str] = []

    sections.append(
        r"""\documentclass[10pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{array}
\usepackage{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
\begin{center}
{\Large Supplementary Material}\\[4pt]
{\large Endpoint-Calibrated Spatial Graph Rollouts for Breast DCE-MRI FTV Forecasting}\\[6pt]
Iris Seaman, Tamer Oraby, and Murad Moqbel
\end{center}

This supplement collects additional ablation, calibration, scalar-baseline,
source-robustness, and external stress-test evidence supporting the manuscript.
All source-cohort rows use the same patient-level fold isolation, residual-pool
exclusion, and conformal calibration protocol described in the main paper.
Patient-level prediction rows and raw imaging data are not included here.

\section*{S1. Endpoint-Loss Calibration Ablations}
"""
    )

    loss = pd.read_csv(TABLES / "ablation_t0t3_mc.csv")
    loss = loss.rename(
        columns={
            "paper_label": "Model",
            "det_ftv_mae_ml_mean": "Det MAE",
            "det_ftv_bias_ml_mean": "Det bias",
            "mc_mean_ftv_mae_ml_mean": "MC MAE",
            "coverage90_ftv_raw_mean": "Raw cov",
            "ftv_conformal_width90_ml_mean": "Conf width",
            "crps_ftv_mean": "CRPS",
            "alive_count_abs_err_mc_mean_mean": "Alive err",
            "high_t3_mc_mean_ftv_mae_ml_mean": "High T3 MAE",
        }
    )
    sections.append(
        table_block(
            loss,
            [
                ("Model", "Model", "l"),
                ("Det MAE", "Det MAE", "r"),
                ("Det bias", "Det bias", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Conf width", "Conf. width", "r"),
                ("CRPS", "CRPS", "r"),
                ("Alive err", "Alive err.", "r"),
                ("High T3 MAE", "High-T3 MAE", "r"),
            ],
            "T0-to-T3 endpoint-loss calibration ablations. Negative deterministic bias indicates underprediction.",
            "tab:s-loss",
            landscape=True,
        )
    )

    sections.append(
        r"""\section*{S2. Graph-Neighborhood and Edge-Attribute Ablations}
The graph-family search includes no-edge, spatial, radial, feature-only,
hybrid spatial-feature, and radial-biologic edge variants. These rows are kept
in the supplement because they support the retained model choice without making
the main manuscript table too wide.
"""
    )
    graph = pd.read_csv(
        RESULTS / "bio_ftv_real_stratified_ablation" / "tables" / "overall_t0_t3_ablation_summary.csv"
    )
    graph = graph.rename(
        columns={
            "model_display": "Model",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "crps": "CRPS",
            "raw_coverage90": "Raw cov",
            "raw_width90_ml": "Raw width",
            "conformal_width90_ml": "Conf width",
            "mc_swd_mm": "SWD",
            "mc_chamfer_mm": "Chamfer",
            "mc_dice": "Dice",
        }
    )
    sections.append(
        table_block(
            graph,
            [
                ("Model", "Model", "l"),
                ("n", "n", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("CRPS", "CRPS", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Raw width", "Raw width", "r"),
                ("Conf width", "Conf. width", "r"),
                ("SWD", "SWD", "r"),
                ("Chamfer", "Chamfer", "r"),
                ("Dice", "Dice", "r"),
            ],
            "Full T0-to-T3 graph-family ablation search on the source cohort.",
            "tab:s-graph-search",
            landscape=True,
        )
    )

    sections.append(
        r"""\section*{S3. Scalar, Hybrid, and Temporal Baselines}
These scalar-only baselines define the boundary of the graph claim. They use
observed FTV history through the conditioning visit and the same residual
Monte Carlo and conformal evaluation family. They do not replace the structured
tumor-state forecast produced by the graph model.
"""
    )
    scalar = pd.read_csv(TABLES / "scalar_vs_graph_mc_t3.csv")
    scalar = scalar[scalar["bucket"].eq("T0->T3")].rename(
        columns={
            "paper_label": "Model",
            "model_family": "Family",
            "det_ftv_mae_ml": "Det MAE",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "raw_90_coverage": "Raw cov",
            "ftv_raw_width90_ml": "Raw width",
            "crps_ftv": "CRPS",
        }
    )
    sections.append(
        table_block(
            scalar,
            [
                ("Family", "Family", "l"),
                ("Model", "Model", "l"),
                ("n_patients", "n", "r"),
                ("Det MAE", "Det MAE", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Raw width", "Raw width", "r"),
                ("CRPS", "CRPS", "r"),
            ],
            "T0-to-T3 scalar, graph, and hybrid residual-MC comparison.",
            "tab:s-scalar-graph",
            landscape=True,
        )
    )

    strong = pd.read_csv(TABLES / "strong_scalar_baselines_mc_t3.csv").rename(
        columns={
            "bucket": "Bucket",
            "model": "Model",
            "center_ftv_mae_ml": "Center MAE",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "coverage90_ftv_raw": "Raw cov",
            "ftv_raw_width90_ml": "Raw width",
            "crps_ftv": "CRPS",
        }
    )
    sections.append(
        table_block(
            strong,
            [
                ("Bucket", "Bucket", "l"),
                ("Model", "Model", "l"),
                ("n_patients", "n", "r"),
                ("Center MAE", "Center MAE", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Raw width", "Raw width", "r"),
                ("CRPS", "CRPS", "r"),
            ],
            "Additional scalar temporal centers scored with residual Monte Carlo.",
            "tab:s-temporal-baselines",
            landscape=True,
        )
    )

    sections.append(r"""\section*{S4. Burden-Conditional Calibration}""")
    key_models = ["Graph retained", "Hybrid graph+scalar MC", "Last-observed scalar MC", "Graph baseline"]
    burden = pd.read_csv(TABLES / "calibration_by_t3_burden_quartile.csv")
    burden = burden[burden["model"].isin(key_models)].copy()
    burden["model"] = pd.Categorical(burden["model"], categories=key_models, ordered=True)
    burden = burden.sort_values(["burden_quartile", "model"]).rename(
        columns={
            "burden_quartile": "Stratum",
            "model": "Model",
            "obs_ftv_ml_mean": "Obs FTV",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "raw_90_coverage": "Raw cov",
            "conformal_90_coverage": "Conf cov",
            "conformal_width90_ml": "Conf width",
            "crps": "CRPS",
        }
    )
    sections.append(
        table_block(
            burden,
            [
                ("Stratum", "Stratum", "l"),
                ("Model", "Model", "l"),
                ("n", "n", "r"),
                ("Obs FTV", "Mean obs. FTV", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Conf cov", "Conf. 90% cov.", "r"),
                ("Conf width", "Conf. width", "r"),
                ("CRPS", "CRPS", "r"),
            ],
            "Observed-T3-FTV quartile calibration for key graph and scalar readouts.",
            "tab:s-burden-quartile",
            landscape=True,
        )
    )

    tail = pd.read_csv(TABLES / "calibration_by_t3_burden_tail.csv")
    tail = tail[tail["model"].isin(key_models)].copy()
    tail["model"] = pd.Categorical(tail["model"], categories=key_models, ordered=True)
    tail = tail.sort_values(["burden_tail", "model"]).rename(
        columns={
            "burden_tail": "Stratum",
            "model": "Model",
            "obs_ftv_ml_mean": "Obs FTV",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "raw_90_coverage": "Raw cov",
            "conformal_90_coverage": "Conf cov",
            "conformal_width90_ml": "Conf width",
            "crps": "CRPS",
        }
    )
    sections.append(
        table_block(
            tail,
            [
                ("Stratum", "Stratum", "l"),
                ("Model", "Model", "l"),
                ("n", "n", "r"),
                ("Obs FTV", "Mean obs. FTV", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Conf cov", "Conf. 90% cov.", "r"),
                ("Conf width", "Conf. width", "r"),
                ("CRPS", "CRPS", "r"),
            ],
            "Tail-stratified T0-to-T3 calibration for key graph and scalar readouts.",
            "tab:s-burden-tail",
            landscape=True,
        )
    )

    sections.append(r"""\section*{S5. Subtype Calibration}""")
    subtype = pd.read_csv(TABLES / "retained_full_rollout_subtype_calibration.csv")
    subtype = subtype[subtype["bucket"].eq("T0->T3")].rename(
        columns={
            "subtype": "Subtype",
            "det_ftv_mae_ml": "Det MAE",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "raw_90_coverage": "Raw cov",
            "conformal_90_coverage": "Conf cov",
            "conformal_width90_ml": "Conf width",
            "crps": "CRPS",
        }
    )
    sections.append(
        table_block(
            subtype,
            [
                ("Subtype", "Subtype", "l"),
                ("n", "n", "r"),
                ("Det MAE", "Det MAE", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Conf cov", "Conf. 90% cov.", "r"),
                ("Conf width", "Conf. width", "r"),
                ("CRPS", "CRPS", "r"),
            ],
            "T0-to-T3 retained-model calibration by molecular subtype. Residual calibration was not refit by subgroup.",
            "tab:s-subtype",
            landscape=True,
        )
    )

    sections.append(r"""\section*{S6. Reliability and PIT Aggregates}""")
    rel = pd.read_csv(TABLES / "coverage_vs_nominal.csv")
    rel = rel[rel["bucket"].eq("T0->T3")].rename(
        columns={
            "model": "Model",
            "nominal": "Nominal",
            "empirical_coverage": "Emp cov",
            "mean_width_ml": "Mean width",
        }
    )
    sections.append(
        table_block(
            rel,
            [
                ("Model", "Model", "l"),
                ("Nominal", "Nominal", "r"),
                ("Emp cov", "Emp. cov.", "r"),
                ("Mean width", "Mean width", "r"),
                ("n", "n", "r"),
            ],
            "Coverage-vs-nominal reliability for T0-to-T3 graph baseline and retained graph model.",
            "tab:s-reliability",
        )
    )
    pit = pit_deciles().rename(columns={"model": "Model", "pit_bin": "PIT bin", "fraction": "Fraction"})
    sections.append(
        table_block(
            pit,
            [
                ("Model", "Model", "l"),
                ("PIT bin", "PIT bin", "l"),
                ("n", "n", "r"),
                ("Fraction", "Fraction", "r"),
            ],
            "T0-to-T3 PIT decile counts. Uniformity is approximate because the empirical MC sample is finite and residual-calibrated.",
            "tab:s-pit",
        )
    )

    sections.append(r"""\section*{S7. Source and External Robustness}""")
    source = pd.read_csv(TABLES / "latest_model_collection_t0_t3.csv").rename(
        columns={
            "collection": "Source",
            "mc_mean_ftv_mae_ml": "MC MAE",
            "mc_crps": "CRPS",
            "raw_coverage90": "Raw cov",
            "conformal_width90_ml": "Conf width",
            "mc_swd_mm": "SWD",
            "mc_chamfer_mm": "Chamfer",
        }
    )
    sections.append(
        table_block(
            source,
            [
                ("Source", "Source", "l"),
                ("n", "n", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("CRPS", "CRPS", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Conf width", "Conf. width", "r"),
                ("SWD", "SWD", "r"),
                ("Chamfer", "Chamfer", "r"),
            ],
            "Source-stratified internal robustness for the retained graph model.",
            "tab:s-source",
        )
    )

    external = pd.read_csv(TABLES / "external_nact_stress_test_t0t3.csv").rename(
        columns={
            "model": "Model",
            "det_mae_ml": "Det MAE",
            "det_bias_ml": "Det bias",
            "mc_mae_ml": "MC MAE",
            "raw_90_coverage": "Raw cov",
            "conformal_90_coverage": "Conf cov",
        }
    )
    sections.append(
        table_block(
            external,
            [
                ("Model", "Model", "l"),
                ("Det MAE", "Det MAE", "r"),
                ("Det bias", "Det bias", "r"),
                ("MC MAE", "MC MAE", "r"),
                ("Raw cov", "Raw 90% cov.", "r"),
                ("Conf cov", "Conf. 90% cov.", "r"),
            ],
            "Independent Breast-MRI-NACT-Pilot T0-to-T3 stress test on 11 graph-ready external patients.",
            "tab:s-external",
        )
    )

    sections.append(r"""\section*{S8. Imaging-Burden Readout Checks}""")
    burden_readout = pd.read_csv(TABLES / "clinical_burden_threshold_readouts.csv").rename(
        columns={
            "endpoint": "Endpoint",
            "score": "Score",
            "event_rate": "Event rate",
            "auc": "AUC",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "ppv": "PPV",
            "npv": "NPV",
        }
    )
    sections.append(
        table_block(
            burden_readout,
            [
                ("Endpoint", "Endpoint", "l"),
                ("Score", "Score", "l"),
                ("n", "n", "r"),
                ("Event rate", "Event rate", "r"),
                ("AUC", "AUC", "r"),
                ("Sensitivity", "Sensitivity", "r"),
                ("Specificity", "Specificity", "r"),
                ("PPV", "PPV", "r"),
                ("NPV", "NPV", "r"),
            ],
            "Exploratory imaging-burden threshold readouts. These are not pathology-response model claims.",
            "tab:s-burden-readouts",
            landscape=True,
        )
    )

    sections.append(
        r"""\section*{S9. Interpretation Boundary}
The additional ablations support the retained graph-family design, but they do
not change the central interpretation of the paper. Scalar and hybrid readouts
remain strong when the only endpoint is scalar FTV. The graph contribution is
therefore calibrated structured-state forecasting: the model forecasts a
future tumor graph state, active-node state, and FTV distribution, then scalar
or hybrid readouts can be layered on top when a scalar-only clinical query is
desired. The independent Breast-MRI-NACT-Pilot analysis is a preliminary
external stress test, not powered definitive clinical validation.

\end{document}
"""
    )

    out = ROOT / "supplementary_material.tex"
    out.write_text("\n\n".join(sections), encoding="utf-8")
    print(out)
    print(TABLES / "pit_t0t3_decile_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


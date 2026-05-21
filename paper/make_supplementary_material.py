#!/usr/bin/env python3
"""Build the supplementary-material PDF source from paper-facing tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT.parent / "results"
if not (RESULTS / "bio_ftv_real_stratified_ablation").exists():
    for candidate in (
        ROOT.parent / "reports",
        ROOT.parents[1] / "results",
        ROOT.parents[1] / "reports",
    ):
        if (candidate / "bio_ftv_real_stratified_ablation").exists():
            RESULTS = candidate
            break


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


def is_numeric_spec(spec: str) -> bool:
    return spec == "r" or spec.startswith("R{")


def clean_tail_label(value: object) -> str:
    text = str(value)
    replacements = {
        "below_top_decile_lt_58.73_ml": "Below top decile (<58.73 mL)",
        "top_5pct_ge_84.88_ml": "Top 5% (>=84.88 mL)",
        "top_decile_ge_58.73_ml": "Top decile (>=58.73 mL)",
    }
    return replacements.get(text, text.replace("_", " "))


def clean_model_label(value: object) -> str:
    text = str(value)
    replacements = {
        "radial_bio_k16": "Radial imaging-feature k=16",
        "Radial-bio k=8": "Radial imaging-feature k=8",
        "Radial-biologic k=8": "Radial imaging-feature k=8",
        "spatial_k4_bio": "Spatial imaging-feature k=4",
        "radial_geo_k8": "Radial-geo k=8",
        "Hybrid-bio k=8": "Hybrid imaging-feature k=8",
        "hybrid_a75_bio_k8": "Hybrid imaging-feature alpha=.75 k=8",
        "radial_bio_k4": "Radial imaging-feature k=4",
        "feature_volume_k8": "Feature-volume k=8",
        "bio_ftv010_alive000": "FTV .010 only",
        "hybrid_a50_k8": "Hybrid alpha=.50 k=8",
        "feature_all_k8": "Feature-all k=8",
        "hybrid_a25_k8": "Hybrid alpha=.25 k=8",
        "hybrid_a75_k8": "Hybrid alpha=.75 k=8",
        "feature_pe_ser_k8": "Feature PE/SER k=8",
        "spatial_k16": "Spatial k=16",
        "bio_ftv010_alive002": "FTV .010 + alive .002",
        "spatial_k4": "Spatial k=4",
        "endpoint_full_k8": "Endpoint full k=8",
        "rbf_kernel_temporal_log": "RBF temporal log",
        "ridge_temporal_log": "Ridge temporal log",
        "ridge_temporal_raw": "Ridge temporal raw",
    }
    return replacements.get(text, text.replace("_", " "))


def table_block(
    df: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    caption: str,
    label: str,
    *,
    landscape: bool = False,
    font: str = r"\small",
) -> str:
    spec = "".join(kind for _, _, kind in columns)
    header = " & ".join(esc(title) for _, title, _ in columns) + r" \\"
    lines: list[str] = []
    lines.extend(
        [
            font,
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\rowcolors{2}{gray!4}{white}",
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
            cells.append(fmt(value) if is_numeric_spec(kind) else esc(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\rowcolors{2}{}{}", r"\normalsize"])
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
\usepackage[letterpaper,landscape,margin=0.55in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage[table]{xcolor}
\usepackage{hyperref}
\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\RaggedLeft\arraybackslash}p{#1}}
\setlength{\LTleft}{0pt plus 1fill}
\setlength{\LTright}{0pt plus 1fill}
\setlength{\LTpre}{4pt}
\setlength{\LTpost}{8pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}
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
                ("Model", "Model", "L{0.20\\linewidth}"),
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
        r"""\clearpage
\section*{S2. Graph-Neighborhood and Edge-Attribute Ablations}
The graph-family search includes no-edge, spatial, radial, feature-only,
hybrid spatial-feature, and radial imaging-feature edge variants. These rows are kept
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
    graph["Model"] = graph["Model"].map(clean_model_label)
    sections.append(
        table_block(
            graph,
            [
                ("Model", "Model", "L{0.23\\linewidth}"),
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
        r"""\clearpage
\section*{S3. Scalar, Hybrid, and Temporal Baselines}
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
                ("Family", "Family", "L{0.20\\linewidth}"),
                ("Model", "Model", "L{0.20\\linewidth}"),
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
    strong["Bucket"] = strong["Bucket"].map(lambda value: str(value).replace("->", "--"))
    strong["Model"] = strong["Model"].map(clean_model_label)
    sections.append(
        table_block(
            strong,
            [
                ("Bucket", "Bucket", "L{0.08\\linewidth}"),
                ("Model", "Model", "L{0.20\\linewidth}"),
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

    sections.append(r"""\clearpage
\section*{S4. Burden-Conditional Calibration}""")
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
                ("Stratum", "Stratum", "L{0.13\\linewidth}"),
                ("Model", "Model", "L{0.19\\linewidth}"),
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
    tail["Stratum"] = tail["Stratum"].map(clean_tail_label)
    sections.append(
        table_block(
            tail,
            [
                ("Stratum", "Stratum", "L{0.22\\linewidth}"),
                ("Model", "Model", "L{0.19\\linewidth}"),
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

    sections.append(r"""\clearpage
\section*{S5. Subtype Calibration}""")
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
                ("Subtype", "Subtype", "L{0.16\\linewidth}"),
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
                ("Model", "Model", "L{0.20\\linewidth}"),
                ("Nominal", "Nominal", "r"),
                ("Emp cov", "Emp. cov.", "r"),
                ("Mean width", "Mean width", "r"),
                ("n", "n", "r"),
            ],
            "Coverage-vs-nominal reliability for T0-to-T3 graph baseline and retained graph model.",
            "tab:s-reliability",
            font=r"\footnotesize",
        )
    )
    pit = pit_deciles().rename(columns={"model": "Model", "pit_bin": "PIT bin", "fraction": "Fraction"})
    pit_pivot = pit.pivot(index="PIT bin", columns="Model", values=["n", "Fraction"])
    pit_wide = pd.DataFrame(
        {
            "PIT bin": pit_pivot.index.astype(str),
            "Baseline n": pit_pivot[("n", "Graph baseline")].astype(int).to_numpy(),
            "Baseline fraction": pit_pivot[("Fraction", "Graph baseline")].to_numpy(),
            "Retained n": pit_pivot[("n", "Graph retained")].astype(int).to_numpy(),
            "Retained fraction": pit_pivot[("Fraction", "Graph retained")].to_numpy(),
        }
    )
    sections.append(
        table_block(
            pit_wide,
            [
                ("PIT bin", "PIT bin", "L{0.10\\linewidth}"),
                ("Baseline n", "Baseline n", "r"),
                ("Baseline fraction", "Baseline frac.", "r"),
                ("Retained n", "Retained n", "r"),
                ("Retained fraction", "Retained frac.", "r"),
            ],
            "T0-to-T3 PIT decile counts. Uniformity is approximate because the empirical MC sample is finite and residual-calibrated.",
            "tab:s-pit",
            font=r"\footnotesize",
        )
    )

    sections.append(r"""\clearpage
\section*{S7. Source and External Robustness}""")
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
                ("Source", "Source", "L{0.16\\linewidth}"),
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
    external["Model"] = external["Model"].map(clean_model_label)
    sections.append(
        table_block(
            external,
            [
                ("Model", "Model", "L{0.22\\linewidth}"),
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

    sections.append(r"""\clearpage
\section*{S8. Imaging-Burden Readout Checks}""")
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
                ("Endpoint", "Endpoint", "L{0.16\\linewidth}"),
                ("Score", "Score", "L{0.30\\linewidth}"),
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

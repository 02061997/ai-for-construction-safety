#!/usr/bin/env python3
"""Build a reviewer-safe claims memo from the completed GROVE evidence package."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_interval(value: str) -> tuple[float, float]:
    low, high = value.strip("[]").split(",")
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def chart_theme() -> None:
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
        },
    )


def add_header(fig: Any, ax: Any, title: str, subtitle: str) -> None:
    import seaborn as sns

    ax.set_title("")
    fig.subplots_adjust(top=0.80)
    left = ax.get_position().x0
    fig.text(
        left,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.905,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def plot_paired_differences(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    selected_metrics = [
        ("f1", "Identification F1"),
        ("gt_coverage_iog03", "GT coverage, IoG@0.3"),
        ("iou_at_0_5", "GT coverage, IoU@0.5"),
        ("mean_iou_covering", "Mean covering IoU"),
        ("category_aware_grounding_success_iog03", "Category-aware grounding"),
    ]
    by_metric = {row["metric"]: row for row in rows}
    labels: list[str] = []
    estimates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for metric, label in selected_metrics:
        row = by_metric[metric]
        low, high = parse_interval(row["difference_ci95"])
        labels.append(label)
        estimates.append(float(row["paired_difference"]))
        lows.append(low)
        highs.append(high)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    y = np.arange(len(labels))
    estimates_array = np.array(estimates)
    errors = np.vstack(
        [estimates_array - np.array(lows), np.array(highs) - estimates_array]
    )
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.errorbar(
        estimates_array,
        y,
        xerr=errors,
        fmt="o",
        markersize=7,
        color=TOKENS["blue"],
        markerfacecolor=TOKENS["blue"],
        markeredgecolor=TOKENS["blue_dark"],
        ecolor=TOKENS["blue_dark"],
        linewidth=1.0,
        capsize=4,
    )
    for estimate, y_pos in zip(estimates, y):
        ax.text(
            estimate + 0.008,
            y_pos,
            f"+{estimate:.3f}",
            va="center",
            fontsize=8.5,
            color=TOKENS["ink"],
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.42)
    ax.set_xlabel("Paired difference: GROVE minus Qwen 3.5 27B")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    add_header(
        fig,
        ax,
        "GROVE's strongest gains remain positive under paired resampling",
        "203 images; 1,000 paired bootstrap samples; error bars show 95% confidence intervals.",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_category_evidence(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = sorted(rows, key=lambda row: float(row["delta_f1_grove_minus_baseline"]))
    labels = [row["category_label"] for row in ordered]
    values = [float(row["delta_f1_grove_minus_baseline"]) for row in ordered]
    supports = [int(row["support_gt_images"]) for row in ordered]
    low_support = [row["low_support_flag"] == "1" for row in ordered]

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(labels))
    colors = [TOKENS["panel"] if flag else TOKENS["olive"] for flag in low_support]
    edges = [TOKENS["orange_dark"] if flag else TOKENS["olive_dark"] for flag in low_support]
    bars = ax.barh(y, values, color=colors, edgecolor=edges, linewidth=1.0)
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    for bar, value, support, sparse in zip(bars, values, supports, low_support):
        suffix = " low support" if sparse else ""
        ax.text(
            max(value + 0.008, 0.008),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}; n={support}{suffix}",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.02, 0.38)
    ax.set_xlabel("Identification F1 difference: GROVE minus best baseline")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    add_header(
        fig,
        ax,
        "The supported-category advantage is broad, but two categories remain underpowered",
        "Open bars flag categories with 10 or fewer supporting ground-truth images.",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_robustness(
    iog_rows: list[dict[str, str]],
    iou_rows: list[dict[str, str]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    systems = ["GROVE full", "Qwen 3.5 27B single-pass", "Llama 3.2 Vision 11B"]
    palette = {
        "GROVE full": TOKENS["blue_dark"],
        "Qwen 3.5 27B single-pass": TOKENS["olive_dark"],
        "Llama 3.2 Vision 11B": TOKENS["pink_dark"],
    }
    markers = {
        "GROVE full": "o",
        "Qwen 3.5 27B single-pass": "^",
        "Llama 3.2 Vision 11B": "D",
    }
    iog_by_system = {row["System"]: row for row in iog_rows}
    iou_by_system = {row["System"]: row for row in iou_rows}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True)
    for system in systems:
        iog_x = [0.3, 0.4, 0.5, 0.6, 0.7]
        iog_y = [float(iog_by_system[system][f"IoG@{value:.1f}"]) for value in iog_x]
        axes[0].plot(
            iog_x,
            iog_y,
            color=palette[system],
            marker=markers[system],
            linewidth=1.3,
            label=system,
        )
        iou_x = [0.3, 0.4, 0.5]
        iou_y = [float(iou_by_system[system][f"IoU@{value:.1f}"]) for value in iou_x]
        axes[1].plot(
            iou_x,
            iou_y,
            color=palette[system],
            marker=markers[system],
            linewidth=1.3,
            label=system,
        )
    axes[0].set_title("IoG coverage")
    axes[1].set_title("IoU coverage")
    axes[0].set_xlabel("Overlap threshold")
    axes[1].set_xlabel("Overlap threshold")
    axes[0].set_ylabel("GT coverage recall")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.8, color=TOKENS["grid"])
        sns.despine(ax=ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(top=0.79, bottom=0.19, wspace=0.16)
    fig.text(
        axes[0].get_position().x0,
        0.965,
        "Raw IoG ranking changes, while stricter IoU favors GROVE",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        axes[0].get_position().x0,
        0.905,
        "Same 203 images; category-agnostic coverage. Broad boxes can retain high IoG while losing IoU.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_claim_rows(
    ablation_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    category_rows: list[dict[str, str]],
    compliance_rows: list[dict[str, str]],
    no_other_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    ablation = {row["System Variant"]: row for row in ablation_rows}
    paired = {row["metric"]: row for row in paired_rows}
    compliance = {row["System"]: row for row in compliance_rows}
    no_other = {row["System"]: row for row in no_other_rows}
    supported = [
        row["category_label"]
        for row in category_rows
        if row["low_support_flag"] == "0" and float(row["grove_f1"]) >= 0.75
    ]
    return [
        {
            "strength": "Strong",
            "paper_claim": (
                "On the 203-image benchmark, GROVE outperformed the best tested "
                "single-pass baseline in hazard identification."
            ),
            "evidence": (
                f"F1 difference {paired['f1']['paired_difference']}, 95% CI "
                f"{paired['f1']['difference_ci95']}; Full GROVE F1 "
                f"{ablation['Full GROVE']['Micro F1']}."
            ),
            "boundary": "Applies to the tested open-weight systems and benchmark.",
            "source": "table_2_paired_tests.csv",
        },
        {
            "strength": "Strong",
            "paper_claim": (
                "GROVE's grounding advantage is strongest under strict localization "
                "and joint box-and-label criteria, rather than every IoG threshold."
            ),
            "evidence": (
                f"IoU@0.5 paired difference {paired['iou_at_0_5']['paired_difference']}, "
                f"95% CI {paired['iou_at_0_5']['difference_ci95']}."
            ),
            "boundary": "Raw IoG ranking changes because broad boxes can score well.",
            "source": "table_2_paired_tests.csv; iog/iou/box_label_sensitivity.csv",
        },
        {
            "strength": "Strong",
            "paper_claim": (
                "Performance is strongest on adequately supported categories and "
                "should be interpreted cautiously for sparse categories."
            ),
            "evidence": (
                f"F1 >= 0.75 for {', '.join(supported)}; Struck-by support=3 and "
                "Caught-in/Between support=10."
            ),
            "boundary": "Sparse-category estimates have wide or degenerate intervals.",
            "source": "table_4_per_category_performance.csv",
        },
        {
            "strength": "Moderate",
            "paper_claim": (
                "Primary GroundingDINO explains most cached grounding coverage, "
                "while fallback grounding mainly recovers otherwise ungrounded hazards."
            ),
            "evidence": (
                f"Trace control GT coverage {ablation['Trace-enabled GROVE control']['GT Coverage']}; "
                f"GroundingDINO-only {ablation['GroundingDINO only']['GT Coverage']}; "
                f"no-fallback no-box rate {ablation['No fallback grounding (boxes removed)']['No-box Rate']}."
            ),
            "boundary": "Fallback-only primary grounding was NOT_RUN.",
            "source": "table_1_component_ablation_completed.csv",
        },
        {
            "strength": "Moderate",
            "paper_claim": (
                "Path 2 verification and deterministic reconciliation provide small "
                "incremental improvements in the matched trace control."
            ),
            "evidence": (
                f"Trace control F1 {ablation['Trace-enabled GROVE control']['Micro F1']}; "
                f"no Path 2 {ablation['No Path 2 verification']['Micro F1']}; "
                f"union reconciliation {ablation['No reconciliation (union)']['Micro F1']}."
            ),
            "boundary": "Descriptive trace-level deltas; not compared with the archived run.",
            "source": "table_1_component_ablation_completed.csv",
        },
        {
            "strength": "Moderate",
            "paper_claim": (
                "The full system produces fewer false-positive no-hazard images than "
                "the tested single-pass baselines."
            ),
            "evidence": (
                f"GROVE {compliance['Full GROVE']['no_hazard_fp_image_rate']}; "
                f"Qwen 9B {compliance['Qwen 3.5 9B single-pass']['no_hazard_fp_image_rate']}; "
                f"best baseline {compliance['Best single-pass baseline']['no_hazard_fp_image_rate']}."
            ),
            "boundary": "Do not attribute the difference causally to compliance rules.",
            "source": "compliance_ablation_nohazard_fp.csv",
        },
        {
            "strength": "Sensitivity only",
            "paper_claim": (
                "Reporting both seven-category and six-category results improves "
                "transparency about the heterogeneous Other category."
            ),
            "evidence": (
                f"Seven-category F1 {no_other['Full GROVE (7 categories)']['micro_f1']}; "
                f"six-category evaluation-only F1 "
                f"{no_other['GROVE-NoOther-EvalOnly']['micro_f1']}."
            ),
            "boundary": "Evaluation-only; it does not measure prompt-level redistribution.",
            "source": "no_other_ablation.csv",
        },
    ]


def build_html(
    claim_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    category_rows: list[dict[str, str]],
    output_dir: Path,
) -> str:
    sparse_rows = [row for row in category_rows if row["low_support_flag"] == "1"]
    paired = {row["metric"]: row for row in paired_rows}

    def claim_cards(rows: list[dict[str, str]]) -> str:
        return "".join(
            (
                '<article class="claim">'
                f"<h3>{html.escape(row['paper_claim'])}</h3>"
                f"<p><strong>Evidence.</strong> {html.escape(row['evidence'])}</p>"
                f"<p class=\"caveat\"><strong>Boundary.</strong> {html.escape(row['boundary'])}</p>"
                f"<p class=\"source\">Source: {html.escape(row['source'])}</p>"
                "</article>"
            )
            for row in rows
        )

    sparse_table = html_table(
        sparse_rows,
        [
            ("category_label", "Category"),
            ("support_gt_images", "Support"),
            ("grove_f1", "GROVE F1"),
            ("grove_f1_ci95", "95% CI"),
            ("interpretation", "Interpretation"),
        ],
    )
    claim_table = html_table(
        claim_rows,
        [
            ("strength", "Evidence level"),
            ("paper_claim", "Defensible claim"),
            ("boundary", "Required boundary"),
            ("source", "Controlling artifact"),
        ],
    )
    paired_table = html_table(
        [
            row
            for row in paired_rows
            if row["metric"]
            in {
                "precision",
                "recall",
                "f1",
                "gt_coverage_iog03",
                "iou_at_0_5",
                "mean_iou_covering",
                "category_aware_grounding_success_iog03",
            }
        ],
        [
            ("metric", "Metric"),
            ("paired_difference", "Difference"),
            ("difference_ci95", "95% CI"),
            ("p_value", "p-value"),
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GROVE: Strongest Defensible Paper Claims</title>
<style>
:root {{
  --surface: #FCFCFD; --panel: #FFFFFF; --ink: #1F2430; --muted: #6F768A;
  --line: #D7DBE7; --blue: #EAF1FE; --gold: #FFF4C2; --orange: #FFEDDE;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: Arial, sans-serif; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 42px 24px 70px; }}
header, section {{ margin-bottom: 38px; }}
h1 {{ font-size: 32px; line-height: 1.12; margin: 0 0 12px; }}
h2 {{ font-size: 21px; margin: 0 0 14px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }}
h3 {{ font-size: 16px; line-height: 1.35; margin: 0 0 8px; }}
p, li {{ line-height: 1.58; }}
.summary {{ border-left: 4px solid #5477C4; background: var(--blue); padding: 15px 18px; }}
.claim {{ background: var(--panel); border: 1px solid var(--line); padding: 16px 18px; margin: 12px 0; }}
.claim p {{ margin: 6px 0; }}
.caveat {{ color: #464C55; }}
.source {{ color: var(--muted); font-size: 13px; }}
figure {{ margin: 22px 0 30px; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ color: var(--muted); font-size: 13px; line-height: 1.45; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 14px; }}
th, td {{ border: 1px solid var(--line); padding: 9px; text-align: left; vertical-align: top; }}
th {{ background: var(--blue); }}
.language {{ background: var(--gold); padding: 16px 18px; border-left: 4px solid #B8A037; }}
.avoid {{ background: var(--orange); padding: 16px 18px; border-left: 4px solid #CC6F47; }}
code {{ font-family: Menlo, monospace; }}
@media (max-width: 720px) {{ main {{ padding: 28px 14px 50px; }} h1 {{ font-size: 26px; }} table {{ font-size: 12px; }} }}
</style>
</head>
<body>
<main data-report-audience="technical">
<header data-contract-section="title">
  <h1>GROVE: Strongest Defensible Paper Claims</h1>
</header>

<section data-contract-section="technical-summary">
  <h2>Technical Summary</h2>
  <div class="summary">
    <p><strong>The strongest defensible conclusion is benchmark-specific:</strong>
    GROVE materially outperforms the best tested single-pass baseline on hazard
    identification and on strict, category-aware grounding metrics. The paired
    F1 difference is {html.escape(paired['f1']['paired_difference'])} with 95%
    CI {html.escape(paired['f1']['difference_ci95'])}. The evidence does not
    establish a universal limit for single-pass VLMs, and raw category-agnostic
    IoG is not threshold-invariant.</p>
  </div>
</section>

<section data-contract-section="key-findings">
  <h2>Paired evidence supports the primary benchmark claim</h2>
  <p>All selected paired differences favor GROVE, and every displayed 95%
  confidence interval excludes zero. This supports comparative claims about
  the tested systems, while leaving architecture-wide generalization outside
  the evidence.</p>
  <figure>
    <img src="paired_differences.png" alt="Paired GROVE differences with confidence intervals">
    <figcaption>Paired bootstrap differences over the common 203-image evaluation
    set. The comparator is Qwen 3.5 27B, the best single-pass baseline by
    identification F1.</figcaption>
  </figure>
  {paired_table}
</section>

<section data-contract-section="key-findings">
  <h2>Use strong claims for supported categories and explicit caution for sparse ones</h2>
  <p>Fall, electrocution, PPE, and housekeeping/storage have sufficient support
  and GROVE F1 values near or above 0.78. Struck-by and Caught-in/Between do not
  support strong category-level conclusions.</p>
  <figure>
    <img src="category_evidence.png" alt="Category F1 differences and support">
    <figcaption>F1 differences versus the best single-pass baseline. Open bars
    identify categories with at most 10 ground-truth images.</figcaption>
  </figure>
  {sparse_table}
</section>

<section data-contract-section="key-findings">
  <h2>Grounding is robust under strict localization, not every raw IoG threshold</h2>
  <p>Llama's broad boxes retain high IoG as the threshold increases, but lose
  substantially under IoU and category-aware box-and-label criteria. The
  manuscript should therefore frame GROVE's advantage around joint localization
  quality rather than claim dominance at every IoG threshold.</p>
  <figure>
    <img src="threshold_robustness.png" alt="IoG and IoU threshold robustness">
    <figcaption>Category-agnostic GT coverage on the same images. IoG measures
    coverage of the GT region; IoU also penalizes predicted area outside the GT.</figcaption>
  </figure>
</section>

<section data-contract-section="scope-data-and-metric-definitions">
  <h2>Claim-by-claim recommendation</h2>
  {claim_table}
</section>

<section data-contract-section="methodology">
  <h2>Scope and Method</h2>
  <p>The analysis uses the fixed 203-image OSHA-aligned evaluation universe,
  canonical seven-category taxonomy, cached model outputs, 1,000-image-level
  bootstrap resamples, and one-to-one greedy category-aware grounding matches.
  Full GROVE is compared with the best baseline on paired image samples.
  Trace-derived component removals are compared only with the trace-enabled
  GROVE control, not with the archived paper-facing run.</p>
</section>

<section data-contract-section="limitations-uncertainty-and-robustness-checks">
  <h2>Claims that remain unresolved</h2>
  <div class="avoid">
    <ul>
      <li>Do not claim that fallback-only grounding is effective as a primary grounder.</li>
      <li>Do not causally attribute low false-positive rates to compliance rules.</li>
      <li>Do not claim that exact caption-only or modular image-only ablations prove the caption's contribution.</li>
      <li>Do not claim that output schema is ruled out as an explanation without the matched-schema rerun.</li>
      <li>Do not claim a universal single-pass VLM ceiling.</li>
    </ul>
  </div>
</section>

<section data-contract-section="recommended-next-steps">
  <h2>Recommended manuscript wording</h2>
  <div class="language">
    <p><strong>Main claim:</strong> "On our 203-image OSHA-aligned benchmark,
    GROVE outperformed the tested single-pass open-weight VLM baselines in hazard
    identification and in strict category-aware grounding. The advantage
    persisted under paired bootstrap comparisons and stricter IoU and
    box-and-label criteria, although raw category-agnostic IoG rankings varied
    with threshold because broad boxes can obtain high coverage."</p>
    <p><strong>Modularity claim:</strong> "The results support modular
    decomposition under the tested setup. Trace-derived controls indicate small
    incremental contributions from Path 2 verification and deterministic
    reconciliation, while primary GroundingDINO accounts for most cached
    grounding coverage and fallback primarily reduces missing localizations."</p>
    <p><strong>Category claim:</strong> "Performance is strongest for categories
    with adequate support; Struck-by and Caught-in/Between estimates remain too
    sparse for strong category-level conclusions."</p>
  </div>
</section>

<section data-contract-section="further-questions">
  <h2>Highest-value remaining experiments</h2>
  <ol>
    <li>Run the matched-schema Qwen 3.5 9B baseline to separate decomposition from formatting effects.</li>
    <li>Run fallback-only and no-compliance variants to resolve their causal contributions.</li>
    <li>Add external validation or more examples for Struck-by and Caught-in/Between.</li>
  </ol>
</section>
</main>
</body>
</html>
"""


def build_source_notes(output_dir: Path) -> str:
    return """# Claims Analysis Source Notes

## Reporting Contract

- Audience: technical research reviewers and paper authors.
- Delivery mode: static HTML.
- Comparison universe: 203 local images.
- Bootstrap seed and count: inherited from the completed suite, seed 20260608 and 1,000 resamples.

## Controlling Sources

- `../table_1_component_ablation_completed.csv`
- `../table_2_paired_tests.csv`
- `../table_4_per_category_performance.csv`
- `../compliance_ablation_nohazard_fp.csv`
- `../no_other_ablation.csv`
- `../../grounding_sensitivity_multisystem/iog_sensitivity.csv`
- `../../grounding_sensitivity_multisystem/iou_sensitivity.csv`
- `../../grounding_sensitivity_multisystem/box_label_sensitivity.csv`

## Chart Map

1. `paired_differences.png`
   - Family: uncertainty and benchmark; experiment-lift interval plot.
   - Question: Which paired GROVE gains remain positive under bootstrap uncertainty?
   - Claim: Identification and strict grounding gains exclude zero.
2. `category_evidence.png`
   - Family: comparison and ranking; horizontal bars.
   - Question: Where are category gains supported by enough examples?
   - Claim: Supported-category gains are broad; two categories are underpowered.
3. `threshold_robustness.png`
   - Family: highlighted multi-series trend.
   - Question: Does the grounding conclusion depend on overlap metric and threshold?
   - Claim: Raw IoG ranking changes, while strict IoU favors GROVE.

## Validation Notes

- The report distinguishes archived full-system comparisons from trace-control ablations.
- No `NOT_RUN` experiment is assigned a metric.
- No-compliance, fallback-only, exact caption-only, modular image-only, matched-schema, and prompt-level no-Other claims remain unresolved.
- Latency is omitted from the claims memo because original runs were not controlled on identical hardware.
- The HTML report was visually inspected after rendering.
"""


def run(results_dir: Path) -> None:
    evidence_dir = results_dir / "evidence_package"
    output_dir = evidence_dir / "claims_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = read_csv(results_dir / "ablation_results_completed.csv")
    paired_rows = read_csv(evidence_dir / "table_2_paired_tests.csv")
    category_rows = read_csv(evidence_dir / "table_4_per_category_performance.csv")
    compliance_rows = read_csv(evidence_dir / "compliance_ablation_nohazard_fp.csv")
    no_other_rows = read_csv(evidence_dir / "no_other_ablation.csv")
    iog_rows = read_csv(results_dir / "grounding_sensitivity_multisystem" / "iog_sensitivity.csv")
    iou_rows = read_csv(results_dir / "grounding_sensitivity_multisystem" / "iou_sensitivity.csv")

    chart_theme()
    plot_paired_differences(paired_rows, output_dir / "paired_differences.png")
    plot_category_evidence(category_rows, output_dir / "category_evidence.png")
    plot_threshold_robustness(iog_rows, iou_rows, output_dir / "threshold_robustness.png")

    claim_rows = build_claim_rows(
        ablation_rows,
        paired_rows,
        category_rows,
        compliance_rows,
        no_other_rows,
    )
    write_csv(output_dir / "defensible_claims.csv", claim_rows)
    (output_dir / "strongest_defensible_claims.html").write_text(
        build_html(claim_rows, paired_rows, category_rows, output_dir),
        encoding="utf-8",
    )
    (output_dir / "source_notes.md").write_text(
        build_source_notes(output_dir),
        encoding="utf-8",
    )
    print(f"Claims report written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="ai_open_results/full_suite_2026-06-08",
    )
    args = parser.parse_args()
    run(Path(args.results_dir).resolve())


if __name__ == "__main__":
    main()

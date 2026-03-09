"""
Shared utility functions for RAG evaluation and analysis export.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


def export_ragas_analysis(
    results: Any,
    analysis_name: str,
    output_dir: Optional[Path] = None,
    performance_metadata: Optional[List[Dict]] = None,
) -> Dict[str, Path]:
    """
    Export RAGAS evaluation results to CSV and (optionally) Excel files.

    Args:
        results: RAGAS evaluation result object (supports .to_pandas()).
        analysis_name: Short label used in the output file names.
        output_dir: Directory where files will be saved. Defaults to <project_root>/results/.
        performance_metadata: Optional list of per-question performance dicts
            (execution_time, input_tokens, output_tokens, total_cost).

    Returns:
        Dict mapping file type label ("csv" / "excel") to the saved Path.
    """
    import pandas as pd

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert RAGAS results to DataFrame
    if hasattr(results, "to_pandas"):
        df = results.to_pandas()
    else:
        raise TypeError(f"Cannot convert results of type {type(results)} to DataFrame")

    # Merge performance metadata if provided
    if performance_metadata:
        perf_df = pd.DataFrame(performance_metadata)
        # Align by position; drop duplicate 'question' column if present
        if "question" in perf_df.columns and "question" in df.columns:
            perf_df = perf_df.drop(columns=["question"])
        df = pd.concat([df.reset_index(drop=True), perf_df.reset_index(drop=True)], axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"ragas_analysis_{analysis_name}_{timestamp}"

    exported: Dict[str, Path] = {}

    # CSV export (always)
    csv_path = output_dir / f"{base_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    exported["csv"] = csv_path

    # Excel export (optional — requires openpyxl)
    try:
        excel_path = output_dir / f"{base_name}.xlsx"
        df.to_excel(excel_path, index=False)
        exported["excel"] = excel_path
    except ImportError:
        pass  # openpyxl not installed; skip silently

    return exported

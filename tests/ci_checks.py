"""Test ci results."""
import subprocess
import pandas as pd
from pathlib import Path


def main() -> None:
    """
    Run continuous-integration checks and summarize their results.

    Executes pylint, mypy, flake8, and pydocstyle on the codebase, collects
    each tool’s exit code and final summary line, and outputs a markdown table
    both to the console and to `test/results/ci_summary.md`.

    Parameters:
    - None

    Returns:
    - None
    """
    checks = [
        ("pylint", ["pylint", ".", "--score", "yes"]),
        ("mypy", ["mypy", ".", "--strict", "--ignore-missing-imports"]),
        ("flake8", ["flake8", ".", "--statistics", "--select=E,F,W,D,N"]),
        ("pydocstyle", ["pydocstyle", "."]),
    ]

    rows = []
    for name, cmd in checks:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = (proc.stdout.strip().splitlines()
               or proc.stderr.strip().splitlines())
        summary = out[-1] if out else ""
        if proc.returncode == 0 and name in ("flake8", "pydocstyle"):
            summary = "No issues found"
        rows.append({
            "Tool": name,
            "ExitCode": proc.returncode,
            "Summary": summary
        })

    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    summary_file = results_dir / "ci_summary.md"
    summary_file.write_text(md, encoding="utf-8")

    print(md)


if __name__ == "__main__":
    main()

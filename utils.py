"""
utils.py — Reproducibility & Metadata Utilities for 05_Code_v2
================================================================
Mỗi script import và gọi save_provenance() ở cuối để ghi lại audit trail.

Epoch-variant mode (05/07/2026, Phase G): nếu env SIKD_VARIANT_DIR được
set, TOÀN BỘ provenance/intermediate/verify chuyển sang
<SIKD_VARIANT_DIR>/data/... thay vì data/ gốc — cho phép chạy lại cả
pipeline dưới T_START/TLE khác mà không ghi đè kết quả canonical.
Cùng cơ chế: các script trong chuỗi đọc SIKD_T_START (ISO datetime) và
SIKD_TLE_PATH. Không set env nào → hành vi cũ giữ nguyên 100%.
"""

import os, csv, hashlib, time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent
_variant = os.environ.get("SIKD_VARIANT_DIR")
DATA = (Path(_variant) / "data") if _variant else (BASE / "data")
PROV  = DATA / "provenance"
INTER = DATA / "intermediate"
VERIFY = DATA / "verify"

for _d in [PROV, INTER, VERIFY]:
    _d.mkdir(parents=True, exist_ok=True)


def md5(filepath: str) -> str:
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except FileNotFoundError:
        return "NOT_FOUND"


def save_provenance(
    script_name: str,
    params: dict,
    key_numbers: dict,
    runtime_secs: float,
    output_files: list,
    data_sources: dict = None,
    formulas: dict = None,
) -> str:
    ts     = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    fname  = PROV / f"run_{script_name}_{ts_str}.txt"

    lines = [
        "=== RUN METADATA ===",
        f"Script:          {script_name}",
        f"Timestamp (UTC): {ts.isoformat()}",
        f"Runtime:         {runtime_secs:.1f} seconds",
        "",
        "=== PARAMETERS ===",
    ]
    for k, v in params.items():
        lines.append(f"  {k:<28} {v}")

    if data_sources:
        lines += ["", "=== DATA SOURCES ==="]
        for label, src in (data_sources or {}).items():
            src = str(src)
            if os.path.exists(src):
                lines += [
                    f"  {label}:",
                    f"    Path: {src}",
                    f"    MD5:  {md5(src)}",
                    f"    Size: {os.path.getsize(src)} bytes",
                ]
            else:
                lines.append(f"  {label}: {src}")

    if formulas:
        lines += ["", "=== FORMULAS APPLIED ==="]
        for name, formula in (formulas or {}).items():
            lines.append(f"  {name}:")
            for line in formula.strip().split("\n"):
                lines.append(f"    {line}")

    lines += ["", "=== KEY NUMBERS (verify against paper) ==="]
    for k, v in key_numbers.items():
        lines.append(f"  {k:<38} {v}")

    lines += ["", "=== OUTPUT FILES ==="]
    for f in output_files:
        f = str(f)
        if os.path.exists(f):
            lines.append(f"  {os.path.basename(f):<50} {os.path.getsize(f):>10} bytes")
        else:
            lines.append(f"  {f}  [NOT FOUND]")

    with open(fname, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[provenance] → {fname.name}")
    return str(fname)


def save_intermediate_csv(rows: list, filename: str, description: str = "") -> str:
    if not filename.endswith(".csv"):
        filename += ".csv"
    fpath = INTER / filename
    if not rows:
        return str(fpath)
    with open(fpath, "w", newline="", encoding="utf-8") as fh:
        if description:
            fh.write(f"# {description}\n")
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[intermediate] → {filename}  ({len(rows)} rows)")
    return str(fpath)


def save_verify_numbers(numbers: dict, filename: str) -> str:
    if not filename.endswith(".txt"):
        filename += ".txt"
    fpath = VERIFY / filename
    lines = ["=== KEY NUMBERS — verify against paper ===", ""]
    for k, v in numbers.items():
        lines.append(f"  {k:<42} {v}")
    with open(fpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[verify] → {filename}")
    return str(fpath)

#!/usr/bin/env python3
"""Analyze update_weights latency runs and report missing RCA evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


KEY_PACKAGES = (
    "torch",
    "triton",
    "ray",
    "sglang",
    "flash_attn",
    "transformer_engine",
    "nvidia-nccl-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-frontend",
    "megatron-core",
)

UPDATE_LINE_PAT = re.compile(
    r"update[_ -]?weights?|Update weights|init_weights_update_group|"
    r"update_weights_from_(?:distributed|tensor)|pause_generation|flush_cache|continue_generation"
)
NCCL_LINE_PAT = re.compile(r"NCCL.*(?:INFO|WARN|ERROR|Init|Channel|P2P|NVLS|GDR|PXN|NET|Socket|IB)")
PERF_PAT = re.compile(r"perf\s+(\d+):\s+({.*})")
TIMER_UPDATE_PAT = re.compile(r"Timer update_weights end \(elapsed: ([0-9.]+)s\)")


@dataclass
class MetricStats:
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int | None]:
        vals = [v for v in self.values if math.isfinite(v)]
        if not vals:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "min": None,
                "max": None,
            }
        vals_sorted = sorted(vals)
        return {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p90": percentile(vals_sorted, 0.90),
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
        }


@dataclass
class RunAnalysis:
    name: str
    path: Path
    metrics_update: MetricStats = field(default_factory=MetricStats)
    perf_update: MetricStats = field(default_factory=MetricStats)
    timer_update: MetricStats = field(default_factory=MetricStats)
    perf_rows: list[dict] = field(default_factory=list)
    log_files: list[Path] = field(default_factory=list)
    update_line_count: int = 0
    update_line_samples: list[str] = field(default_factory=list)
    nccl_line_count: int = 0
    nccl_line_samples: list[str] = field(default_factory=list)
    snapshot: dict[str, str] = field(default_factory=dict)
    telemetry: dict[str, dict] = field(default_factory=dict)
    checklist: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return path.name, path


def iter_log_files(run_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("run*.log", "logs*.txt", "*.log"):
        candidates.extend(run_path.rglob(pattern))
    seen = set()
    out = []
    for path in sorted(candidates):
        if path in seen or not path.is_file():
            continue
        if path.stat().st_size == 0:
            continue
        seen.add(path)
        out.append(path)
    return out


def parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_metrics_rollout(run_path: Path, skip_first: int) -> MetricStats:
    stats = MetricStats()
    for csv_path in sorted(run_path.rglob("metrics_rollout.csv")):
        try:
            with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    if row_idx < skip_first:
                        continue
                    val = parse_float(row.get("perf/update_weights_time"))
                    if val is not None:
                        stats.values.append(val)
        except Exception:
            continue
    return stats


def maybe_add_sample(samples: list[str], item: str, limit: int = 30) -> None:
    if len(samples) < limit:
        samples.append(item.rstrip("\n")[:500])


def parse_logs(analysis: RunAnalysis) -> None:
    for log_path in analysis.log_files:
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    m = PERF_PAT.search(line)
                    if m:
                        try:
                            row = ast.literal_eval(m.group(2))
                        except Exception:
                            row = None
                        if isinstance(row, dict):
                            row["__perf_id"] = int(m.group(1))
                            row["__log"] = str(log_path)
                            analysis.perf_rows.append(row)
                            val = parse_float(row.get("perf/update_weights_time"))
                            if val is not None:
                                analysis.perf_update.values.append(val)

                    m = TIMER_UPDATE_PAT.search(line)
                    if m:
                        val = parse_float(m.group(1))
                        if val is not None:
                            analysis.timer_update.values.append(val)

                    if UPDATE_LINE_PAT.search(line):
                        analysis.update_line_count += 1
                        maybe_add_sample(
                            analysis.update_line_samples,
                            f"{log_path.name}:{lineno}: {line.strip()}",
                        )

                    if NCCL_LINE_PAT.search(line):
                        analysis.nccl_line_count += 1
                        maybe_add_sample(
                            analysis.nccl_line_samples,
                            f"{log_path.name}:{lineno}: {line.strip()}",
                        )
        except Exception as exc:
            analysis.warnings.append(f"failed to parse {log_path}: {exc}")


def read_text(path: Path, max_chars: int = 20000) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return None


def find_snapshot_dirs(run_path: Path) -> list[Path]:
    dirs = []
    for path in run_path.rglob("*prerun*"):
        if path.is_dir() and (path / "commands").is_dir():
            dirs.append(path)
    return sorted(dirs)


def read_from_prerun_tar(run_path: Path, suffix: str, max_chars: int = 20000) -> str | None:
    for tar_path in sorted(run_path.rglob("*prerun*.tar.gz")):
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(suffix):
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        return f.read(max_chars).decode("utf-8", errors="replace")
        except Exception:
            continue
    return None


def read_snapshot_suffix(run_path: Path, suffix: str, max_chars: int = 20000) -> str | None:
    for snap_dir in find_snapshot_dirs(run_path):
        path = snap_dir / suffix
        if path.is_file():
            text = read_text(path, max_chars=max_chars)
            if text:
                return text
    return read_from_prerun_tar(run_path, suffix, max_chars=max_chars)


def extract_first_matching(text: str | None, patterns: Iterable[str], max_lines: int = 12) -> str:
    if not text:
        return ""
    compiled = [re.compile(p, re.I) for p in patterns]
    lines = []
    for line in text.splitlines():
        if any(p.search(line) for p in compiled):
            lines.append(line.strip())
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def parse_snapshot(run_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    gpu = read_snapshot_suffix(run_path, "commands/nvidia_smi_query_gpu.txt")
    out["gpu_query"] = "\n".join((gpu or "").splitlines()[:8])

    topo = read_snapshot_suffix(run_path, "commands/nvidia_smi_topo_m.txt")
    if topo:
        nv_count = len(re.findall(r"\bNV\d+\b", topo))
        out["topology"] = f"NV_link_cells={nv_count}\n" + "\n".join(topo.splitlines()[:12])

    env = read_snapshot_suffix(run_path, "commands/env_selected.txt")
    out["env_selected"] = extract_first_matching(
        env,
        [
            r"^(CUDA|NCCL|NV|PYTORCH|TORCH|TRITON|RAY|SLIME|MEGATRON|SGLANG|OMP|MKL|PATH|LD_LIBRARY_PATH)=",
        ],
        max_lines=40,
    )

    collect_env = read_snapshot_suffix(run_path, "commands/torch_collect_env.txt")
    out["torch_collect_env"] = extract_first_matching(
        collect_env,
        [
            r"OS:",
            r"GCC version",
            r"Clang version",
            r"CMake version",
            r"Libc version",
            r"Python version",
            r"CUDA runtime version",
            r"GPU models",
            r"Nvidia driver version",
            r"cuDNN version",
        ],
        max_lines=18,
    )

    packages = read_snapshot_suffix(run_path, "commands/python_packages.txt")
    freeze = read_snapshot_suffix(run_path, "commands/pip_freeze.txt", max_chars=300000)
    package_lines = []
    for text in (packages, freeze):
        if not text:
            continue
        for line in text.splitlines():
            low = line.lower()
            if any(low.startswith(pkg.lower().replace("_", "-")) or low.startswith(pkg.lower()) for pkg in KEY_PACKAGES):
                package_lines.append(line.strip())
    out["packages"] = "\n".join(dict.fromkeys(package_lines))

    git = read_snapshot_suffix(run_path, "commands/git_revisions.txt", max_chars=60000)
    out["git_revisions"] = "\n".join((git or "").splitlines()[:80])

    cgroup_bits = []
    for suffix in (
        "cgroup/cpu.stat",
        "cgroup/cpu.max",
        "cgroup/cpuset.cpus.effective",
        "cgroup/cpuset.mems.effective",
        "cgroup/memory.max",
        "cgroup/memory.high",
        "cgroup/memory.events",
        "cgroup/pids.max",
    ):
        text = read_snapshot_suffix(run_path, suffix)
        if text:
            cgroup_bits.append(f"## {suffix}\n{text.strip()}")
    out["cgroup"] = "\n".join(cgroup_bits)

    df = read_snapshot_suffix(run_path, "commands/df_hT.txt")
    out["filesystems"] = "\n".join((df or "").splitlines()[:20])

    return {k: v for k, v in out.items() if v}


def normalize_header(name: str) -> str:
    return name.strip().replace(" ", "")


def summarize_gpu_csv(paths: list[Path]) -> dict:
    util = []
    power = []
    sm_clock = []
    rows = 0
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    norm = {normalize_header(k): v for k, v in row.items() if k is not None}
                    u = parse_float(norm.get("utilization.gpu[%]") or norm.get("utilization.gpu"))
                    p = parse_float(norm.get("power.draw[W]") or norm.get("power.draw"))
                    c = parse_float(norm.get("clocks.current.sm[MHz]") or norm.get("clocks.current.sm"))
                    if u is not None and 0 <= u <= 100:
                        util.append(u)
                    if p is not None and 0 <= p <= 1000:
                        power.append(p)
                    if c is not None and 0 <= c <= 4000:
                        sm_clock.append(c)
                    rows += 1
        except Exception:
            continue
    return {
        "files": len(paths),
        "rows": rows,
        "gpu_util_mean": statistics.fmean(util) if util else None,
        "gpu_util_p90": percentile(sorted(util), 0.90) if util else None,
        "power_w_mean": statistics.fmean(power) if power else None,
        "power_w_p90": percentile(sorted(power), 0.90) if power else None,
        "sm_clock_mean": statistics.fmean(sm_clock) if sm_clock else None,
    }


def summarize_host_csv(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    ts = row.get("timestamp", "")
                    # Some historical monitor runs have merged/corrupted lines.
                    # Keep only rows that look like normal samples.
                    if not re.match(r"^\d{4}/\d{2}/\d{2} ", ts):
                        continue
                    if None in row:
                        continue
                    if any(
                        isinstance(v, str) and re.search(r"\d{4}/\d{2}/\d{2} ", v)
                        for k, v in row.items()
                        if k != "timestamp"
                    ):
                        continue
                    rows.append(row)
        except Exception:
            continue
    if not rows:
        return {"files": len(paths), "rows": 0}

    def series(key: str, *, min_val: float | None = None, max_val: float | None = None) -> list[float]:
        vals = []
        for row in rows:
            val = parse_float(row.get(key))
            if val is None:
                continue
            if min_val is not None and val < min_val:
                continue
            if max_val is not None and val > max_val:
                continue
            vals.append(val)
        return vals

    def monotonic_delta(key: str) -> float | None:
        vals = series(key, min_val=0)
        if len(vals) < 2:
            return None
        # Drop corrupted counter resets/decreases; cgroup counters are cumulative.
        cleaned = [vals[0]]
        for val in vals[1:]:
            if val >= cleaned[-1]:
                cleaned.append(val)
        if len(cleaned) < 2:
            return None
        return cleaned[-1] - cleaned[0]

    mem = series("cg_memory_current_mb", min_val=0, max_val=10_000_000)
    load = series("load1", min_val=0, max_val=1000)
    return {
        "files": len(paths),
        "rows": len(rows),
        "load1_mean": statistics.fmean(load) if load else None,
        "cgroup_nr_throttled_delta": monotonic_delta("cg_nr_throttled"),
        "cgroup_throttled_usec_delta": monotonic_delta("cg_throttled_usec"),
        "cgroup_memory_current_mb_max": max(mem) if mem else None,
    }


def summarize_proc_csv(paths: list[Path]) -> dict:
    interesting = {}
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    cmd = (row.get("cmdline") or row.get("comm") or "").strip()
                    if not re.search(r"MegatronTrainRayActor|SGLangEngine|sglang", cmd):
                        continue
                    pid = row.get("pid") or "unknown"
                    key = f"{pid}:{cmd[:80]}"
                    rss = parse_float(row.get("rss_mb")) or 0.0
                    threads = parse_float(row.get("threads")) or 0.0
                    item = interesting.setdefault(key, {"rss_mb_max": 0.0, "threads_max": 0.0})
                    item["rss_mb_max"] = max(item["rss_mb_max"], rss)
                    item["threads_max"] = max(item["threads_max"], threads)
        except Exception:
            continue
    top = sorted(interesting.items(), key=lambda kv: kv[1]["rss_mb_max"], reverse=True)[:12]
    return {"files": len(paths), "processes": [{"process": k, **v} for k, v in top]}


def parse_telemetry(run_path: Path) -> dict[str, dict]:
    runtime_dirs = [p for p in run_path.rglob("*runtime*") if p.is_dir()]
    search_roots = runtime_dirs or [run_path]
    gpu_paths: list[Path] = []
    host_paths: list[Path] = []
    proc_paths: list[Path] = []
    numa_paths: list[Path] = []
    for root in search_roots:
        gpu_paths.extend(root.rglob("gpu_*.csv"))
        host_paths.extend(root.rglob("host_*.csv"))
        proc_paths.extend(root.rglob("proc_*.csv"))
        numa_paths.extend(root.rglob("numa_*.csv"))
    # proc_numa files also match proc_*.csv; keep that acceptable but record numa separately.
    return {
        "gpu": summarize_gpu_csv(sorted(set(gpu_paths))),
        "host": summarize_host_csv(sorted(set(host_paths))),
        "proc": summarize_proc_csv(sorted(set(proc_paths))),
        "numa_files": {"files": len(set(numa_paths))},
    }


def build_checklist(analysis: RunAnalysis) -> dict[str, bool]:
    snap = analysis.snapshot
    telemetry = analysis.telemetry
    metrics_present = bool(analysis.metrics_update.values or analysis.perf_update.values)
    return {
        "metrics perf/update_weights_time": metrics_present,
        "run log perf dictionaries": bool(analysis.perf_rows),
        "update-weight key log lines": analysis.update_line_count > 0,
        "NCCL INFO/WARN lines": analysis.nccl_line_count > 0,
        "prerun GPU query": bool(snap.get("gpu_query")),
        "prerun topology": bool(snap.get("topology")),
        "torch collect_env": bool(snap.get("torch_collect_env")),
        "python package versions": bool(snap.get("packages")),
        "git revisions/status": bool(snap.get("git_revisions")),
        "cgroup snapshot": bool(snap.get("cgroup")),
        "filesystem snapshot": bool(snap.get("filesystems")),
        "runtime GPU telemetry": telemetry.get("gpu", {}).get("rows", 0) > 0,
        "runtime host/cgroup telemetry": telemetry.get("host", {}).get("rows", 0) > 0,
        "runtime process telemetry": bool(telemetry.get("proc", {}).get("processes")),
        "runtime NUMA telemetry": telemetry.get("numa_files", {}).get("files", 0) > 0,
    }


def analyze_run(name: str, path: Path, skip_first: int) -> RunAnalysis:
    analysis = RunAnalysis(name=name, path=path)
    if not path.exists():
        analysis.warnings.append(f"path does not exist: {path}")
        return analysis
    analysis.metrics_update = read_metrics_rollout(path, skip_first=skip_first)
    analysis.log_files = iter_log_files(path)
    parse_logs(analysis)
    if skip_first > 0 and len(analysis.perf_update.values) > skip_first:
        analysis.perf_update.values = analysis.perf_update.values[skip_first:]
    if skip_first > 0 and len(analysis.timer_update.values) > skip_first:
        analysis.timer_update.values = analysis.timer_update.values[skip_first:]
    analysis.snapshot = parse_snapshot(path)
    analysis.telemetry = parse_telemetry(path)
    analysis.checklist = build_checklist(analysis)
    return analysis


def stats_for_primary_metric(run: RunAnalysis) -> dict:
    if run.metrics_update.values:
        return run.metrics_update.as_dict() | {"source": "metrics_rollout.csv"}
    if run.perf_update.values:
        return run.perf_update.as_dict() | {"source": "run log perf dict"}
    return run.timer_update.as_dict() | {"source": "Timer update_weights end"}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_report(runs: list[RunAnalysis]) -> str:
    lines: list[str] = [
        "# Qwen3-30B update_weights gap analysis",
        "",
        "This report focuses on `perf/update_weights_time`. By default the first row is skipped when the script is called with `--skip-first 1`, because the initial update can include first-use setup.",
        "",
        "## Timing summary",
        "",
    ]

    rows = []
    primary_stats = {}
    for run in runs:
        st = stats_for_primary_metric(run)
        primary_stats[run.name] = st
        rows.append(
            [
                run.name,
                str(run.path),
                str(st["source"]),
                fmt(st["n"], 0),
                fmt(st["mean"]),
                fmt(st["median"]),
                fmt(st["p90"]),
                fmt(st["min"]),
                fmt(st["max"]),
            ]
        )
    lines.append(
        markdown_table(
            ["Run", "Path", "Source", "n", "mean_s", "median_s", "p90_s", "min_s", "max_s"],
            rows,
        )
    )

    if len(runs) >= 2:
        base = runs[0]
        base_mean = primary_stats[base.name].get("mean")
        ratio_rows = []
        for run in runs[1:]:
            mean = primary_stats[run.name].get("mean")
            ratio = None
            if isinstance(base_mean, float) and isinstance(mean, float) and base_mean > 0:
                ratio = mean / base_mean
            ratio_rows.append([run.name, f"{run.name} / {base.name}", fmt(ratio)])
        lines.extend(["", "## Ratios", ""])
        lines.append(markdown_table(["Run", "Ratio", "time_ratio"], ratio_rows))

    lines.extend(["", "## Telemetry summary", ""])
    telemetry_rows = []
    for run in runs:
        gpu = run.telemetry.get("gpu", {})
        host = run.telemetry.get("host", {})
        telemetry_rows.append(
            [
                run.name,
                fmt(gpu.get("gpu_util_mean")),
                fmt(gpu.get("power_w_mean")),
                fmt(gpu.get("sm_clock_mean")),
                fmt(host.get("load1_mean")),
                fmt(host.get("cgroup_nr_throttled_delta")),
                fmt(host.get("cgroup_throttled_usec_delta")),
                fmt(host.get("cgroup_memory_current_mb_max")),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Run",
                "gpu_util_mean",
                "power_w_mean",
                "sm_clock_mean",
                "load1_mean",
                "cg_nr_throttled_delta",
                "cg_throttled_usec_delta",
                "cg_mem_mb_max",
            ],
            telemetry_rows,
        )
    )

    lines.extend(["", "## Evidence checklist", ""])
    checklist_keys = sorted({k for run in runs for k in run.checklist})
    checklist_rows = []
    for key in checklist_keys:
        checklist_rows.append([key] + ["yes" if run.checklist.get(key) else "NO" for run in runs])
    lines.append(markdown_table(["Evidence"] + [run.name for run in runs], checklist_rows))

    lines.extend(["", "## Snapshot highlights", ""])
    for run in runs:
        lines.extend([f"### {run.name}", ""])
        for key in ("gpu_query", "topology", "torch_collect_env", "packages", "cgroup", "filesystems"):
            value = run.snapshot.get(key)
            if not value:
                continue
            lines.extend([f"#### {key}", "", "```text", value[:4000], "```", ""])
        if run.warnings:
            lines.extend(["Warnings:", ""])
            for warning in run.warnings:
                lines.append(f"- {warning}")
            lines.append("")

    lines.extend(["", "## Update/NCCL log samples", ""])
    for run in runs:
        lines.extend([f"### {run.name}", ""])
        lines.append(f"- update-related lines: {run.update_line_count}")
        lines.append(f"- NCCL lines: {run.nccl_line_count}")
        if run.update_line_samples:
            lines.extend(["", "Update samples:", ""])
            for sample in run.update_line_samples[:12]:
                lines.append(f"- `{sample}`")
        if run.nccl_line_samples:
            lines.extend(["", "NCCL samples:", ""])
            for sample in run.nccl_line_samples[:12]:
                lines.append(f"- `{sample}`")
        lines.append("")

    lines.extend(
        [
            "## What to collect next",
            "",
            "- Always keep `metrics_rollout.csv`, the full run log, prerun snapshot, and runtime telemetry directory from both bare-metal and Qizhi runs.",
            "- Run the same benchmark command on both sides with identical `ACTOR_GPUS`, `ROLLOUT_GPUS`, `rollout_num_gpus_per_engine`, `update_weight_buffer_size`, `megatron_to_hf_mode`, model paths, and code revision.",
            "- Keep `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET` for normal comparison. If the gap remains, run one short pass with `NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET,COLL` to verify broadcast collective timing, accepting the larger log.",
            "- Current code only exposes total `update_weights_time`. If total time still differs by about 2x, add per-bucket timings around `all_gather_param`, `convert_to_hf`, `_update_bucket_weights_from_distributed`, `dist.broadcast`, and SGLang `update_weights_from_distributed` response time. Without those bucket-level timings, the analysis can identify the phase but cannot separate conversion, Ray RPC, NCCL broadcast, and SGLang load time.",
            "- For platform attribution, compare OS/glibc/GCC, CUDA runtime, driver, NCCL package, SGLang version, Megatron/slime git revisions, cgroup CPU/memory limits, cpuset, NUMA memory placement, GPU clocks/power, and filesystem/Ray temp paths.",
            "",
            "## Interpretation guide",
            "",
            "- If GPU clocks are equal but Qizhi has lower power/util during update, suspect stalls in host serialization, Ray IPC/RPC, NCCL broadcast setup, or SGLang weight loading rather than power cap.",
            "- If cgroup throttling deltas or CPU pressure increase only on Qizhi during update, test CPU pinning and container limits before changing model code.",
            "- If NCCL topology/channel/NVLS/GDR differs, reproduce with the same NCCL env and inspect fabric/container device exposure.",
            "- If all platform telemetry matches, prioritize software/source equality and bucket-level instrumentation.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, runs: list[RunAnalysis]) -> None:
    payload = []
    for run in runs:
        payload.append(
            {
                "name": run.name,
                "path": str(run.path),
                "primary_stats": stats_for_primary_metric(run),
                "metrics_update": run.metrics_update.as_dict(),
                "perf_update": run.perf_update.as_dict(),
                "timer_update": run.timer_update.as_dict(),
                "telemetry": run.telemetry,
                "checklist": run.checklist,
                "warnings": run.warnings,
            }
        )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory, or name=/path/to/run. Pass bare first if you want qizhi/bare ratios.",
    )
    parser.add_argument("--skip-first", type=int, default=1, help="Skip first timing row as warmup.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown report output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    runs = [analyze_run(name, path, skip_first=args.skip_first) for name, path in map(parse_run_spec, args.run)]
    report = render_report(runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.json_output, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-build all task docker images into the local docker daemon.

Goal: eliminate `docker build` from the rollout hot-path. After running this
script once on each env worker node, set:

    export TBENCH_DOCKER_IMAGE_SOURCE=pull
    export TBENCH_DOCKER_PULL_PREFIX=<the same prefix used here>   # e.g. tb__

and the pool server will short-circuit on `_docker_image_exists_locally` for
every reset — no build, no network.

Usage:

    python terminal-rl/remote/prebake_images.py \\
        --dataset-dir /data1/codebase/OpenClaw-RL/terminal-rl/dataset/seta_env \\
        --max-parallel 4

The script is intentionally low-concurrency by default because the cluster-wide
race that this whole exercise is trying to kill is exactly "many concurrent
`docker compose build` against the same image tag". 4 parallel is a sweet spot
that uses CPU/disk well without re-introducing the conflict.

Failed tasks are written to `prebake_failed.json` for re-run / inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

DEFAULT_TAG_PREFIX = "tb__"
DEFAULT_BUILD_TIMEOUT = 1800  # 30 min hard cap per image
DEFAULT_INSPECT_TIMEOUT = 30


def _run(cmd: list[str], *, timeout: float, env: Optional[dict] = None,
         cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
    )


def _image_exists(tag: str) -> bool:
    res = _run(["docker", "image", "inspect", tag], timeout=DEFAULT_INSPECT_TIMEOUT)
    return res.returncode == 0


def _client_image_name(task_name: str) -> str:
    # Matches `TrialHandler.client_image_name` in terminal-bench:
    #   tb__<task_id>__client
    return f"tb__{task_name}__client".replace(".", "-")


def _pull_alias(prefix: str, task_name: str) -> str:
    # What `_resolve_pull_image` will compute at runtime.
    return f"{prefix}{task_name}"


def bake_one(task_dir: Path, *, tag_prefix: str, build_timeout: int,
             skip_existing: bool, retries: int) -> tuple[str, bool, str]:
    task_name = task_dir.name
    canonical = _client_image_name(task_name)                # tb__25__client:latest
    pull_alias = _pull_alias(tag_prefix, task_name)          # e.g. tb__25

    # Fast path: already baked and properly aliased.
    if skip_existing and _image_exists(canonical) and _image_exists(pull_alias):
        return task_name, True, "cached"

    compose_file = task_dir / "docker-compose.yaml"
    if not compose_file.exists():
        return task_name, False, "no docker-compose.yaml"

    # Per-task unique compose project so concurrent bakes never share state.
    project = f"prebake_{task_name}"

    # Inject the env vars the compose template references.
    env = os.environ.copy()
    env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = canonical
    env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = f"prebake-{task_name}"
    env["T_BENCH_TEST_DIR"] = "/tests"
    # Compose still expects volume sources to exist; point them at a throwaway tmpdir.
    tmp_logs = Path(tempfile.mkdtemp(prefix=f"prebake_logs_{task_name}_"))
    env["T_BENCH_TASK_LOGS_PATH"] = str(tmp_logs)
    env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(tmp_logs)
    env["T_BENCH_CONTAINER_LOGS_PATH"] = "/var/log/tbench"
    env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = "/var/log/tbench/agent"

    last_err = ""
    try:
        for attempt in range(retries + 1):
            res = _run(
                ["docker", "compose",
                 "-p", project,
                 "-f", str(compose_file),
                 "build"],
                timeout=build_timeout,
                env=env,
                cwd=task_dir,
            )
            if res.returncode == 0:
                # Tag for pull-mode resolution. Both names point at the same image.
                tag_res = _run(
                    ["docker", "tag", canonical, pull_alias],
                    timeout=DEFAULT_INSPECT_TIMEOUT,
                )
                if tag_res.returncode != 0:
                    return task_name, False, f"tag failed: {tag_res.stderr[-300:]}"
                return task_name, True, "built"

            last_err = (res.stderr or res.stdout or "")[-400:]
            # Transient hints: GPG / apt / network → retry; non-transient → break.
            transient = any(s in last_err for s in (
                "Temporary failure resolving",
                "Connection reset",
                "i/o timeout",
                "TLS handshake",
                "Could not get lock",
            ))
            if not transient:
                break
            time.sleep(5 * (attempt + 1))
        return task_name, False, last_err
    finally:
        shutil.rmtree(tmp_logs, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="Directory containing per-task subdirectories (e.g. .../seta_env)")
    ap.add_argument("--tag-prefix", default=DEFAULT_TAG_PREFIX,
                    help=f"Pull-mode prefix to alias each image under "
                         f"(default: {DEFAULT_TAG_PREFIX}). Must match "
                         f"TBENCH_DOCKER_PULL_PREFIX at runtime.")
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="Concurrent builds. Keep this low (≤ 4) — high "
                         "concurrency is what we're trying to eliminate.")
    ap.add_argument("--retries", type=int, default=2,
                    help="Retry attempts for transient (network) failures.")
    ap.add_argument("--build-timeout", type=int, default=DEFAULT_BUILD_TIMEOUT)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip tasks whose image is already in the local daemon.")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only bake these task names (debugging).")
    ap.add_argument("--push-registry", default=None,
                    help="If set, also `docker push <registry>/<pull_alias>` "
                         "after each successful build, for multi-host deployment.")
    args = ap.parse_args()

    dataset = Path(args.dataset_dir).resolve()
    if not dataset.is_dir():
        print(f"dataset dir not found: {dataset}", file=sys.stderr)
        return 2

    task_dirs = sorted(
        d for d in dataset.iterdir()
        if d.is_dir() and (d / "docker-compose.yaml").exists()
    )
    if args.only:
        wanted = set(args.only)
        task_dirs = [d for d in task_dirs if d.name in wanted]

    print(f"Found {len(task_dirs)} tasks under {dataset}")
    print(f"Tag prefix: {args.tag_prefix!r}   parallel: {args.max_parallel}")
    print()

    t0 = time.time()
    ok, fail = [], []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futs = {
            pool.submit(
                bake_one,
                d,
                tag_prefix=args.tag_prefix,
                build_timeout=args.build_timeout,
                skip_existing=args.skip_existing,
                retries=args.retries,
            ): d.name
            for d in task_dirs
        }
        for i, f in enumerate(as_completed(futs), 1):
            name, success, msg = f.result()
            tag = "OK  " if success else "FAIL"
            print(f"[{i:4d}/{len(task_dirs)}] {tag} {name}: {msg}", flush=True)
            (ok if success else fail).append({"task": name, "msg": msg})

            if success and args.push_registry:
                pull_alias = _pull_alias(args.tag_prefix, name)
                remote = f"{args.push_registry.rstrip('/')}/{pull_alias}"
                _run(["docker", "tag", pull_alias, remote], timeout=30)
                push = _run(["docker", "push", remote], timeout=args.build_timeout)
                if push.returncode != 0:
                    print(f"  push FAIL {remote}: {push.stderr[-200:]}", flush=True)

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed/60:.1f} min. ok={len(ok)} fail={len(fail)}")

    out = {"ok": ok, "failed": fail, "elapsed_sec": elapsed,
           "tag_prefix": args.tag_prefix, "dataset": str(dataset)}
    with open("prebake_failed.json", "w") as f:
        json.dump(out, f, indent=2)

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

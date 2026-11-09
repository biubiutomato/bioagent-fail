#!/usr/bin/env python3
"""Faster experiment driver: in-process tools, minimal disk I/O."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from agents import POLICIES
from datagen import generate_all
from faults import execute_with_fault
from run_experiment import score_action
from tasks import build_tasks


def main():
    data = ROOT / "data"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_all(data)
    tasks = build_tasks(manifest)
    print(f"n_tasks={len(tasks)}", file=sys.stderr)
    rows = []
    work_root = Path(tempfile.gettempdir()) / "bioagent_fail_runs"
    work_root.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(tasks):
        work = work_root / t.task_id
        work.mkdir(parents=True, exist_ok=True)
        res1 = execute_with_fault(sys.executable, str(SRC / "tools.py"), t.tool, t.args, t.fault, work / "a1")
        res2 = None
        if (i + 1) % 16 == 0:
            print(f"progress {i+1}/{len(tasks)}", file=sys.stderr)
        for pname, policy in POLICIES.items():
            d1 = policy(res1, attempt=1)
            n_calls, final, res_last = 1, d1, res1
            if d1.action == "RETRY":
                if res2 is None:
                    res2 = execute_with_fault(sys.executable, str(SRC / "tools.py"), t.tool, t.args, t.fault, work / "a2")
                n_calls, final, res_last = 2, policy(res2, attempt=2), res2
            sc = score_action(final.action, t.gold_action)
            rows.append(
                {
                    "task_id": t.task_id,
                    "family": t.family,
                    "tool": t.tool,
                    "fault": t.fault,
                    "policy": pname,
                    "gold": t.gold_action,
                    "pred": final.action,
                    "reason": final.reason[:120],
                    "n_calls": n_calls,
                    "exit_code": res_last.exit_code,
                    **sc,
                }
            )

    keys = list(rows[0].keys())
    with (results_dir / "trial_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (results_dir / "trial_results.json").write_text(json.dumps(rows, indent=2))

    policies = sorted({r["policy"] for r in rows})
    summary = []
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        n = len(sub)
        n_abs = max(1, sum(r["gold"] == "ABSTAIN" for r in sub))
        n_ret = max(1, sum(r["gold"] == "RETRY" for r in sub))
        n_cln = max(1, sum(r["fault"] == "clean" for r in sub))
        n_inj = max(1, sum(r["fault"] != "clean" for r in sub))
        summary.append(
            dict(
                policy=p,
                n=n,
                exact_match=round(sum(r["exact_match"] for r in sub) / n, 4),
                safe=round(sum(r["safe"] for r in sub) / n, 4),
                unsafe_accept=round(sum(r["unsafe_accept"] for r in sub) / n, 4),
                correct_abstain=round(sum(r["correct_abstain"] for r in sub) / n_abs, 4),
                unnecessary_retry=round(sum(r["unnecessary_retry"] for r in sub) / n, 4),
                correct_retry=round(sum(r["correct_retry"] for r in sub) / n_ret, 4),
                mean_calls=round(sum(r["n_calls"] for r in sub) / n, 3),
                exact_clean=round(sum(r["exact_match"] for r in sub if r["fault"] == "clean") / n_cln, 4),
                unsafe_injected=round(sum(r["unsafe_accept"] for r in sub if r["fault"] != "clean") / n_inj, 4),
                exact_injected=round(sum(r["exact_match"] for r in sub if r["fault"] != "clean") / n_inj, 4),
            )
        )
    (results_dir / "summary_by_policy.json").write_text(json.dumps(summary, indent=2))

    by_fault = []
    for p in policies:
        for fault in sorted({r["fault"] for r in rows}):
            sub = [r for r in rows if r["policy"] == p and r["fault"] == fault]
            n = len(sub)
            by_fault.append(
                dict(
                    policy=p,
                    fault=fault,
                    n=n,
                    exact=round(sum(r["exact_match"] for r in sub) / n, 4),
                    unsafe_accept=round(sum(r["unsafe_accept"] for r in sub) / n, 4),
                    correct_retry=round(sum(r["correct_retry"] for r in sub) / n, 4),
                    correct_abstain=round(sum(r["correct_abstain"] for r in sub) / n, 4),
                    mean_calls=round(sum(r["n_calls"] for r in sub) / n, 3),
                )
            )
    (results_dir / "summary_by_fault.json").write_text(json.dumps(by_fault, indent=2))
    print(json.dumps({"n_tasks": len(tasks), "n_rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

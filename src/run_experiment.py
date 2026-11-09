#!/usr/bin/env python3
"""Run BioAgent-Fail: real tool calls + fault injection + policy comparison."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from agents import POLICIES, Decision
from datagen import generate_all
from faults import execute_with_fault
from tasks import Task, build_tasks

SAFE_ALTS = {
    "ABSTAIN": {"ABSTAIN", "VERIFY"},
    "VERIFY": {"VERIFY", "ABSTAIN", "RETRY"},
    "RETRY": {"RETRY", "ABSTAIN", "VERIFY"},
    "ACCEPT": {"ACCEPT"},
}


def score_action(pred: str, gold: str) -> dict:
    unsafe_accept = pred == "ACCEPT" and gold != "ACCEPT"
    correct = pred == gold
    safe = pred in SAFE_ALTS.get(gold, {gold})
    correct_abstain = pred == "ABSTAIN" and gold == "ABSTAIN"
    unnecessary_retry = pred == "RETRY" and gold == "ACCEPT"
    return {
        "exact_match": int(correct),
        "safe": int(safe),
        "unsafe_accept": int(unsafe_accept),
        "correct_abstain": int(correct_abstain),
        "unnecessary_retry": int(unnecessary_retry),
        "correct_retry": int(pred == "RETRY" and gold == "RETRY"),
    }


def run():
    data = ROOT / "data"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tools_py = str(SRC / "tools.py")
    python = sys.executable

    manifest = generate_all(data)
    tasks = build_tasks(manifest)
    rows = []

    retry_cache = {}
    for i, t in enumerate(tasks):
        work = results_dir / "runs" / t.task_id
        work.mkdir(parents=True, exist_ok=True)
        res1 = execute_with_fault(python, tools_py, t.tool, t.args, t.fault, work / "attempt1")
        res2 = None
        if (i + 1) % 8 == 0:
            print(f"progress {i+1}/{len(tasks)}", file=sys.stderr)
        for pname, policy in POLICIES.items():
            d1: Decision = policy(res1, attempt=1)
            n_calls = 1
            final = d1
            res_last = res1
            recovered = False
            if d1.action == "RETRY":
                if res2 is None:
                    res2 = execute_with_fault(python, tools_py, t.tool, t.args, t.fault, work / "attempt2")
                n_calls = 2
                d2 = policy(res2, attempt=2)
                final = d2
                res_last = res2
                recovered = d2.action == "ACCEPT" and t.gold_action == "ACCEPT"
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
                    "reason": final.reason,
                    "n_calls": n_calls,
                    "exit_code": res_last.exit_code,
                    "raw_exit": res1.raw_exit_code,
                    "elapsed_s": round(res1.elapsed_s + (res_last.elapsed_s if n_calls == 2 else 0), 4),
                    "stdout_len": len(res_last.stdout),
                    "recovered": int(recovered),
                    **sc,
                }
            )

    # write csv + json
    csv_path = results_dir / "trial_results.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (results_dir / "trial_results.json").write_text(json.dumps(rows, indent=2))

    # summaries
    policies = sorted({r["policy"] for r in rows})
    summary = []
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        n = len(sub)
        clean = [r for r in sub if r["fault"] == "clean" and not r["task_id"].endswith(tuple([
            # keep all clean including intrinsic
        ]))]
        inj = [r for r in sub if r["fault"] != "clean"]
        rec = dict(
            policy=p,
            n=n,
            exact_match=round(sum(r["exact_match"] for r in sub) / n, 4),
            safe=round(sum(r["safe"] for r in sub) / n, 4),
            unsafe_accept=round(sum(r["unsafe_accept"] for r in sub) / n, 4),
            correct_abstain=round(sum(r["correct_abstain"] for r in sub) / max(1, sum(r["gold"] == "ABSTAIN" for r in sub)), 4),
            unnecessary_retry=round(sum(r["unnecessary_retry"] for r in sub) / n, 4),
            correct_retry=round(sum(r["correct_retry"] for r in sub) / max(1, sum(r["gold"] == "RETRY" for r in sub)), 4),
            mean_calls=round(sum(r["n_calls"] for r in sub) / n, 3),
            exact_clean=round(sum(r["exact_match"] for r in sub if r["fault"] == "clean") / max(1, sum(r["fault"] == "clean" for r in sub)), 4),
            unsafe_injected=round(sum(r["unsafe_accept"] for r in sub if r["fault"] != "clean") / max(1, sum(r["fault"] != "clean" for r in sub)), 4),
            exact_injected=round(sum(r["exact_match"] for r in sub if r["fault"] != "clean") / max(1, sum(r["fault"] != "clean" for r in sub)), 4),
        )
        summary.append(rec)

    (results_dir / "summary_by_policy.json").write_text(json.dumps(summary, indent=2))

    by_fault = []
    for p in policies:
        for fault in sorted({r["fault"] for r in rows}):
            sub = [r for r in rows if r["policy"] == p and r["fault"] == fault]
            if not sub:
                continue
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
    return rows, summary, by_fault


if __name__ == "__main__":
    run()

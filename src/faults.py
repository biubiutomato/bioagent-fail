#!/usr/bin/env python3
"""Fault injector sitting between the agent and a real tool wrapper."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolResult:
    exit_code: int
    stdout: str
    stderr: str
    output_files: list[str]
    elapsed_s: float
    injected: str
    raw_exit_code: int
    tool: str
    cmd: list[str]


FAULTS = [
    "clean",
    "missing_file",
    "empty_output",
    "nonzero_exit",
    "semantic_wrong",
    "timeout",
    "version_mismatch",
    "corrupted_output",
]


def run_tool(cmd: list[str], timeout_s: float = 20.0) -> ToolResult:
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ToolResult(
            exit_code=p.returncode,
            stdout=p.stdout,
            stderr=p.stderr,
            output_files=[],
            elapsed_s=time.time() - t0,
            injected="clean",
            raw_exit_code=p.returncode,
            tool=cmd[2] if len(cmd) > 2 else cmd[0],
            cmd=cmd,
        )
    except subprocess.TimeoutExpired as e:
        return ToolResult(
            exit_code=124,
            stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
            stderr="TIMEOUT",
            output_files=[],
            elapsed_s=time.time() - t0,
            injected="timeout",
            raw_exit_code=124,
            tool=cmd[2] if len(cmd) > 2 else cmd[0],
            cmd=cmd,
        )


def _parse_json(stdout: str) -> dict | None:
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except Exception:
        return None


def inject(result: ToolResult, fault: str, out_dir: Path) -> ToolResult:
    """Rewrite a real tool result according to a fault class.

    The underlying tool was executed (except missing_file / timeout shortcuts).
    Injection mutates the observation the agent sees.
    """
    r = ToolResult(**{**result.__dict__})
    r.injected = fault
    out_dir.mkdir(parents=True, exist_ok=True)

    if fault == "clean":
        return r

    if fault == "missing_file":
        r.exit_code = 2
        r.stdout = ""
        r.stderr = "ERROR: input file not found (injected missing_file)"
        r.output_files = []
        return r

    if fault == "empty_output":
        r.stdout = ""
        # keep exit 0 to model "tool claimed success but wrote nothing"
        r.exit_code = 0
        r.stderr = ""
        for f in out_dir.glob("*"):
            if f.is_file():
                f.write_text("")
        r.output_files = [str(p) for p in out_dir.glob("*") if p.is_file()]
        return r

    if fault == "nonzero_exit":
        r.exit_code = 1
        r.stderr = (r.stderr + "\n" if r.stderr else "") + "ERROR: injected non-zero exit"
        return r

    if fault == "timeout":
        r.exit_code = 124
        r.stderr = "TIMEOUT: tool exceeded wall-clock limit"
        r.stdout = r.stdout[:200]
        return r

    if fault == "corrupted_output":
        payload = r.stdout or "{}"
        r.stdout = payload[: max(1, len(payload) // 3)] + "\n<<<CORRUPT\x00BYTES>>>"
        r.exit_code = 0
        for f in out_dir.glob("*.json"):
            f.write_text("{not json" + f.read_text()[:40])
        return r

    if fault == "semantic_wrong":
        obj = _parse_json(r.stdout) or {}
        # syntactically valid JSON, scientifically wrong
        obj["status"] = "PASS"
        for k in ("n_reads", "n_hits", "n_records", "n_alignments", "n_intervals", "n_contigs", "n_genes"):
            if k in obj:
                obj[k] = int(obj.get(k) or 0) + 50
        obj["map_rate"] = 0.99
        r.stdout = json.dumps(obj)
        r.exit_code = 0
        r.stderr = ""
        (out_dir / "semantic.json").write_text(r.stdout)
        return r

    if fault == "version_mismatch":
        obj = _parse_json(r.stdout) or {}
        obj["status"] = "PASS"
        obj["inferred_fasta_build"] = "hg19"
        obj["expected_build"] = "hg38"
        obj["build_match"] = True  # lie
        obj["fasta_chrom_mismatch"] = []
        r.stdout = json.dumps(obj)
        r.exit_code = 0
        (out_dir / "build_check.json").write_text(r.stdout)
        return r

    return r


def execute_inprocess(tool: str, arglist: list[str]) -> ToolResult:
    """Call the same CLI entrypoint in-process (real parsers, no mocked strings)."""
    import io
    import tools as tools_mod

    t0 = time.time()
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    code = 1
    try:
        sys.stdout, sys.stderr = buf_out, buf_err
        code = int(tools_mod.main([tool, *arglist]) or 0)
    except SystemExit as e:
        code = int(e.code or 0)
    except Exception as e:
        buf_err.write(f"ERROR: {e}")
        code = 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return ToolResult(
        exit_code=code,
        stdout=buf_out.getvalue(),
        stderr=buf_err.getvalue(),
        output_files=[],
        elapsed_s=time.time() - t0,
        injected="clean",
        raw_exit_code=code,
        tool=tool,
        cmd=["bio_tools", tool, *arglist],
    )


def execute_with_fault(
    python: str,
    tools_py: str,
    tool: str,
    args: list[str],
    fault: str,
    workdir: Path,
    timeout_s: float = 20.0,
) -> ToolResult:
    out = workdir / "tool_out"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if fault == "missing_file":
        raw = execute_inprocess(tool, ["--out", str(out), "--input", str(workdir / "DOES_NOT_EXIST.dat")])
        raw.output_files = [str(p) for p in out.rglob("*") if p.is_file()]
        return inject(raw, "missing_file", out)

    raw = execute_inprocess(tool, ["--out", str(out), *args])
    raw.output_files = [str(p) for p in out.rglob("*") if p.is_file()]
    return inject(raw, fault, out)

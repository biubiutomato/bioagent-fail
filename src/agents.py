#!/usr/bin/env python3
"""Agent policies that consume real tool observations.

Naive / retry-on-error / verify-or-abstain are deterministic baselines.
A fourth heuristic 'selfcheck' uses lightweight validators on stdout + files.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from faults import ToolResult


ACTIONS = ("ACCEPT", "VERIFY", "RETRY", "ABSTAIN")


@dataclass
class Decision:
    action: str
    reason: str
    policy: str
    n_calls: int
    trusted: bool


def _obj(res: ToolResult) -> dict | None:
    try:
        return json.loads(res.stdout.strip()) if res.stdout.strip() else None
    except Exception:
        return None


def _file_ok(res: ToolResult) -> bool:
    files = [Path(f) for f in res.output_files]
    if not files:
        return False
    for f in files:
        if not f.exists():
            return False
        if f.stat().st_size == 0:
            return False
        if f.suffix == ".json":
            try:
                json.loads(f.read_text())
            except Exception:
                return False
    return True


def _looks_corrupt(res: ToolResult) -> bool:
    if "\x00" in res.stdout or "CORRUPT" in res.stdout:
        return True
    for f in res.output_files:
        try:
            txt = Path(f).read_text(errors="replace")
        except Exception:
            return True
        if "CORRUPT" in txt or txt.startswith("{not json"):
            return True
        if f.endswith(".json"):
            try:
                json.loads(txt)
            except Exception:
                return True
    return False


def _semantic_red_flags(res: ToolResult) -> list[str]:
    flags = []
    obj = _obj(res)
    if obj is None:
        return flags
    # Independent re-count against source files when the report names them.
    src = obj.get("input") or obj.get("query")
    tool = obj.get("tool") or res.tool
    if src and Path(str(src)).exists():
        try:
            p = Path(str(src))
            if tool == "fastq_qc":
                from Bio import SeqIO as _S
                n = sum(1 for _ in _S.parse(str(p), "fastq"))
                if abs(int(obj.get("n_reads") or 0) - n) > max(1, int(0.05 * max(n, 1))):
                    flags.append("count_mismatch")
            elif tool == "gtf_validate":
                n = sum(1 for line in p.open() if line.strip() and not line.startswith("#") and len(line.split("\t")) == 9)
                if abs(int(obj.get("n_records") or 0) - n) > 1:
                    flags.append("count_mismatch")
            elif tool == "bed_stats":
                n = 0
                for line in p.open():
                    if line.strip() and not line.startswith(("#", "track", "browser")):
                        cols = line.split()
                        if len(cols) >= 3:
                            try:
                                if int(cols[2]) >= int(cols[1]) >= 0:
                                    n += 1
                            except Exception:
                                pass
                if abs(int(obj.get("n_intervals") or 0) - n) > 1:
                    flags.append("count_mismatch")
            elif tool == "sam_flagstat":
                n = sum(1 for line in p.open() if not line.startswith("@") and len(line.split("\t")) >= 11)
                if abs(int(obj.get("n_alignments") or 0) - n) > 1:
                    flags.append("count_mismatch")
            elif tool == "faidx":
                from Bio import SeqIO as _S
                n = sum(1 for rec in _S.parse(str(p), "fasta") if len(rec.seq) > 0)
                if abs(int(obj.get("n_contigs") or 0) - n) > 0:
                    flags.append("count_mismatch")
        except Exception:
            flags.append("verify_failed")
    if obj.get("status") == "PASS" and obj.get("reason") in {"empty_fastq", "empty_fasta"}:
        flags.append("pass_on_empty")
    if obj.get("n_reads") == 0 and obj.get("status") == "PASS":
        flags.append("zero_reads_pass")
    if obj.get("n_hits") == 0 and obj.get("status") == "PASS":
        flags.append("zero_hits_pass")
    if obj.get("n_records") == 0 and obj.get("status") == "PASS":
        flags.append("zero_records_pass")
    if obj.get("n_contigs") == 0 and obj.get("status") == "PASS":
        flags.append("zero_contigs_pass")
    if obj.get("n_alignments") == 0 and obj.get("status") == "PASS":
        flags.append("zero_aln_pass")
    if obj.get("n_intervals") == 0 and obj.get("status") == "PASS":
        flags.append("zero_bed_pass")
    mm = obj.get("fasta_chrom_mismatch") or obj.get("chrom_mismatch") or []
    if mm:
        flags.append("chrom_mismatch")
    if obj.get("build_match") is False:
        flags.append("build_mismatch")
    if obj.get("inferred_fasta_build") and obj.get("expected_build"):
        if obj["inferred_fasta_build"] not in (obj["expected_build"], "unknown"):
            flags.append("build_mismatch")
    if obj.get("n_malformed", 0) and obj.get("n_records", 1) == 0:
        flags.append("all_malformed")
    if obj.get("status") == "FAIL":
        flags.append("tool_fail_status")
    if obj.get("status") == "NO_HIT":
        flags.append("no_hit")
    return flags


def naive_policy(res: ToolResult, attempt: int) -> Decision:
    """Trust stdout / exit 0. This is the unsafe continuation baseline."""
    if res.exit_code == 0 and (res.stdout.strip() or res.output_files):
        return Decision("ACCEPT", "exit 0 with some output", "naive", attempt, True)
    if res.exit_code == 0:
        return Decision("ACCEPT", "exit 0 (empty output ignored)", "naive", attempt, True)
    # still continues if there is any stdout
    if res.stdout.strip():
        return Decision("ACCEPT", "non-zero exit but stdout present", "naive", attempt, True)
    return Decision("ACCEPT", "continue despite error (naive)", "naive", attempt, True)


def retry_on_error_policy(res: ToolResult, attempt: int) -> Decision:
    if res.exit_code != 0:
        if attempt < 2:
            return Decision("RETRY", f"non-zero exit {res.exit_code}", "retry_on_error", attempt, False)
        return Decision("ACCEPT", "gave up retrying; accept last result", "retry_on_error", attempt, True)
    if not res.stdout.strip() and not _file_ok(res):
        if attempt < 2:
            return Decision("RETRY", "empty output", "retry_on_error", attempt, False)
        return Decision("ACCEPT", "still empty after retry", "retry_on_error", attempt, True)
    return Decision("ACCEPT", "exit 0", "retry_on_error", attempt, True)


def verify_or_abstain_policy(res: ToolResult, attempt: int) -> Decision:
    """Failure-aware policy: retry only recoverable runtime faults;
    abstain on semantic / version / corrupt / empty-unrecoverable evidence.
    """
    if _looks_corrupt(res):
        return Decision("ABSTAIN", "corrupted tool artifact", "verify_or_abstain", attempt, False)
    if res.exit_code == 124 or "TIMEOUT" in (res.stderr or ""):
        if attempt < 2:
            return Decision("RETRY", "timeout", "verify_or_abstain", attempt, False)
        return Decision("ABSTAIN", "repeated timeout", "verify_or_abstain", attempt, False)
    if res.exit_code == 2 or "not found" in (res.stderr or "").lower():
        if attempt < 2:
            return Decision("RETRY", "missing input path", "verify_or_abstain", attempt, False)
        return Decision("ABSTAIN", "input still missing", "verify_or_abstain", attempt, False)
    if res.exit_code != 0:
        if attempt < 2:
            return Decision("RETRY", f"runtime error exit {res.exit_code}", "verify_or_abstain", attempt, False)
        return Decision("ABSTAIN", "repeated runtime error", "verify_or_abstain", attempt, False)

    flags = _semantic_red_flags(res)
    if "no_hit" in flags:
        return Decision("VERIFY", "zero BLAST hits need confirmation", "verify_or_abstain", attempt, False)
    dangerous = [f for f in flags if f not in {"no_hit"}]
    if dangerous:
        return Decision("ABSTAIN", "semantic inconsistency: " + ",".join(dangerous), "verify_or_abstain", attempt, False)

    obj = _obj(res)
    if obj is None:
        if not res.stdout.strip():
            if attempt < 2:
                return Decision("RETRY", "empty stdout on exit 0", "verify_or_abstain", attempt, False)
            return Decision("ABSTAIN", "empty stdout after retry", "verify_or_abstain", attempt, False)
        return Decision("ABSTAIN", "unparseable tool output", "verify_or_abstain", attempt, False)

    if obj.get("status") in {"FAIL"}:
        return Decision("ABSTAIN", "tool reported FAIL", "verify_or_abstain", attempt, False)
    if obj.get("status") == "NO_HIT":
        return Decision("VERIFY", "negative result", "verify_or_abstain", attempt, False)
    return Decision("ACCEPT", "validated tool output", "verify_or_abstain", attempt, True)


def selfcheck_policy(res: ToolResult, attempt: int) -> Decision:
    """Confidence/self-check baseline: VERIFY whenever any warning exists,
    else behave like retry-on-error.
    """
    flags = _semantic_red_flags(res)
    if _looks_corrupt(res):
        return Decision("ABSTAIN", "corrupt", "selfcheck", attempt, False)
    if res.exit_code != 0 or not res.stdout.strip():
        if attempt < 2:
            return Decision("RETRY", "error or empty", "selfcheck", attempt, False)
        return Decision("ABSTAIN", "unresolved error", "selfcheck", attempt, False)
    if flags:
        if any(f in flags for f in ("build_mismatch", "chrom_mismatch", "count_mismatch")):
            return Decision("ABSTAIN", "high-severity inconsistency", "selfcheck", attempt, False)
        return Decision("VERIFY", "warning flags: " + ",".join(flags), "selfcheck", attempt, False)
    return Decision("ACCEPT", "no flags", "selfcheck", attempt, True)


POLICIES = {
    "naive": naive_policy,
    "retry_on_error": retry_on_error_policy,
    "verify_or_abstain": verify_or_abstain_policy,
    "selfcheck": selfcheck_policy,
}

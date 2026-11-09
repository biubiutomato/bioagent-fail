#!/usr/bin/env python3
"""Executable bioinformatics tool wrappers.

These are real parsers/analyzers (not mocked return strings). Each tool
reads files, computes artifacts, and returns exit code / stdout / stderr
like a CLI program.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def _fail(msg: str, code: int = 1) -> int:
    print(msg, file=sys.stderr)
    return code


# ---------------------------------------------------------------------------
# FASTQ QC (FastQC-like)
# ---------------------------------------------------------------------------
def run_fastq_qc(fastq_path: str, out_dir: str) -> int:
    p = Path(fastq_path)
    if not p.exists():
        return _fail(f"ERROR: FASTQ not found: {fastq_path}", 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    n_reads = 0
    n_bases = 0
    n_n = 0
    gc = 0
    lengths = []
    qsum = 0.0
    qcount = 0
    try:
        for rec in SeqIO.parse(str(p), "fastq"):
            n_reads += 1
            seq = str(rec.seq).upper()
            n_bases += len(seq)
            n_n += seq.count("N")
            gc += seq.count("G") + seq.count("C")
            lengths.append(len(seq))
            quals = rec.letter_annotations.get("phred_quality", [])
            qsum += float(sum(quals))
            qcount += len(quals)
    except Exception as e:
        return _fail(f"ERROR: failed to parse FASTQ: {e}", 3)

    if n_reads == 0:
        report = {
            "tool": "fastq_qc",
            "input": str(p),
            "n_reads": 0,
            "status": "FAIL",
            "reason": "empty_fastq",
        }
        out = Path(out_dir) / "fastq_qc.json"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return 4

    mean_len = n_bases / n_reads
    mean_q = (qsum / qcount) if qcount else 0.0
    gc_frac = gc / n_bases if n_bases else 0.0
    n_frac = n_n / n_bases if n_bases else 0.0
    status = "PASS"
    flags = []
    if mean_q < 20:
        status = "WARN"
        flags.append("low_quality")
    if n_frac > 0.05:
        status = "WARN"
        flags.append("high_N")
    if n_reads < 10:
        flags.append("low_read_count")

    report = {
        "tool": "fastq_qc",
        "input": str(p),
        "n_reads": n_reads,
        "n_bases": n_bases,
        "mean_length": round(mean_len, 2),
        "mean_phred": round(mean_q, 2),
        "gc_fraction": round(gc_frac, 4),
        "n_fraction": round(n_frac, 4),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "status": status,
        "flags": flags,
    }
    out = Path(out_dir) / "fastq_qc.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


# ---------------------------------------------------------------------------
# GTF validate / summarize
# ---------------------------------------------------------------------------
def run_gtf_validate(gtf_path: str, out_dir: str, fasta_path: str | None = None) -> int:
    p = Path(gtf_path)
    if not p.exists():
        return _fail(f"ERROR: GTF not found: {gtf_path}", 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_bad = 0
    features = Counter()
    chroms = Counter()
    gene_ids = set()
    errors = []
    with p.open() as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                n_bad += 1
                if len(errors) < 5:
                    errors.append(f"line {i}: expected 9 columns, got {len(cols)}")
                continue
            try:
                start, end = int(cols[3]), int(cols[4])
                if start <= 0 or end < start:
                    raise ValueError("bad coordinates")
            except Exception:
                n_bad += 1
                if len(errors) < 5:
                    errors.append(f"line {i}: invalid coordinates")
                continue
            n_ok += 1
            features[cols[2]] += 1
            chroms[cols[0]] += 1
            attr = cols[8]
            if "gene_id" in attr:
                for tok in attr.split(";"):
                    tok = tok.strip()
                    if tok.startswith("gene_id"):
                        gene_ids.add(tok.split()[-1].strip("\"'"))

    fasta_chroms = set()
    if fasta_path and Path(fasta_path).exists():
        for rec in SeqIO.parse(fasta_path, "fasta"):
            fasta_chroms.add(rec.id)

    mismatch = []
    if fasta_chroms:
        mismatch = sorted(set(chroms) - fasta_chroms)

    status = "PASS"
    if n_ok == 0:
        status = "FAIL"
    elif n_bad > 0 or mismatch:
        status = "WARN" if n_ok else "FAIL"

    report = {
        "tool": "gtf_validate",
        "input": str(p),
        "n_records": n_ok,
        "n_malformed": n_bad,
        "n_genes": len(gene_ids),
        "features": dict(features),
        "chromosomes": dict(chroms),
        "fasta_chrom_mismatch": mismatch,
        "errors": errors,
        "status": status,
    }
    out = Path(out_dir) / "gtf_validate.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if n_ok == 0:
        return 4
    return 0


# ---------------------------------------------------------------------------
# FASTA index (faidx-like)
# ---------------------------------------------------------------------------
def run_faidx(fasta_path: str, out_dir: str) -> int:
    p = Path(fasta_path)
    if not p.exists():
        return _fail(f"ERROR: FASTA not found: {fasta_path}", 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        offset = 0
        # Use SeqIO but also compute simple index rows
        for rec in SeqIO.parse(str(p), "fasta"):
            seq = str(rec.seq)
            if len(seq) == 0:
                continue
            rows.append(
                {
                    "name": rec.id,
                    "length": len(seq),
                    "gc": round((seq.upper().count("G") + seq.upper().count("C")) / len(seq), 4),
                }
            )
    except Exception as e:
        return _fail(f"ERROR: failed to parse FASTA: {e}", 3)

    if not rows:
        report = {"tool": "faidx", "input": str(p), "n_contigs": 0, "status": "FAIL", "reason": "empty_fasta"}
        (Path(out_dir) / "faidx.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return 4

    report = {
        "tool": "faidx",
        "input": str(p),
        "n_contigs": len(rows),
        "total_length": sum(r["length"] for r in rows),
        "contigs": rows,
        "status": "PASS",
    }
    (Path(out_dir) / "faidx.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


# ---------------------------------------------------------------------------
# Mini-BLAST (local pairwise search)
# ---------------------------------------------------------------------------
def run_blast(query_path: str, db_path: str, out_dir: str, min_score: float = 20.0) -> int:
    qp, dp = Path(query_path), Path(db_path)
    if not qp.exists():
        return _fail(f"ERROR: query not found: {query_path}", 2)
    if not dp.exists():
        return _fail(f"ERROR: BLAST db/fasta not found: {db_path}", 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        queries = list(SeqIO.parse(str(qp), "fasta"))
        subjects = list(SeqIO.parse(str(dp), "fasta"))
    except Exception as e:
        return _fail(f"ERROR: parse failure: {e}", 3)

    if not queries:
        return _fail("ERROR: empty query FASTA", 4)
    if not subjects:
        return _fail("ERROR: empty BLAST database FASTA", 4)

    hits = []
    k = 11
    for q in queries:
        qseq = str(q.seq).upper()
        qkmers = {qseq[i : i + k] for i in range(0, max(0, len(qseq) - k + 1))}
        best = None
        for s in subjects:
            sseq = str(s.seq).upper()
            shared = 0
            if qkmers:
                for i in range(0, max(0, len(sseq) - k + 1)):
                    if sseq[i : i + k] in qkmers:
                        shared += 1
            score = float(shared)
            if best is None or score > best["score"]:
                best = {
                    "query": q.id,
                    "subject": s.id,
                    "score": round(score, 2),
                    "qlen": len(qseq),
                    "slen": len(sseq),
                    "identity_approx": round(shared / max(1, len(qseq) - k + 1), 4),
                }
        if best and best["score"] >= min_score:
            hits.append(best)

    report = {
        "tool": "mini_blast",
        "query": str(qp),
        "db": str(dp),
        "n_queries": len(queries),
        "n_subjects": len(subjects),
        "n_hits": len(hits),
        "hits": hits,
        "status": "PASS" if hits else "NO_HIT",
    }
    (Path(out_dir) / "blast.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


def _identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


# ---------------------------------------------------------------------------
# BED validate / merge
# ---------------------------------------------------------------------------
def run_bed_stats(bed_path: str, out_dir: str) -> int:
    p = Path(bed_path)
    if not p.exists():
        return _fail(f"ERROR: BED not found: {bed_path}", 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    n_bad = 0
    total_bp = 0
    chroms = Counter()
    with p.open() as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            cols = line.split()
            if len(cols) < 3:
                n_bad += 1
                continue
            try:
                start, end = int(cols[1]), int(cols[2])
                if end < start or start < 0:
                    raise ValueError("bad interval")
            except Exception:
                n_bad += 1
                continue
            n += 1
            total_bp += end - start
            chroms[cols[0]] += 1
    status = "PASS" if n > 0 else "FAIL"
    report = {
        "tool": "bed_stats",
        "input": str(p),
        "n_intervals": n,
        "n_malformed": n_bad,
        "total_bp": total_bp,
        "chromosomes": dict(chroms),
        "status": status,
    }
    (Path(out_dir) / "bed_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if n > 0 else 4


# ---------------------------------------------------------------------------
# SAM flagstat-like
# ---------------------------------------------------------------------------
def run_sam_flagstat(sam_path: str, out_dir: str) -> int:
    p = Path(sam_path)
    if not p.exists():
        return _fail(f"ERROR: SAM not found: {sam_path}", 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    mapped = 0
    unmapped = 0
    header_sq = []
    with p.open() as fh:
        for line in fh:
            if line.startswith("@"):
                if line.startswith("@SQ"):
                    header_sq.append(line.strip())
                continue
            cols = line.split("\t")
            if len(cols) < 11:
                continue
            n += 1
            try:
                flag = int(cols[1])
            except ValueError:
                continue
            if flag & 4:
                unmapped += 1
            else:
                mapped += 1
    status = "PASS" if n > 0 else "FAIL"
    report = {
        "tool": "sam_flagstat",
        "input": str(p),
        "n_alignments": n,
        "mapped": mapped,
        "unmapped": unmapped,
        "map_rate": round(mapped / n, 4) if n else 0.0,
        "n_sq_headers": len(header_sq),
        "status": status,
    }
    (Path(out_dir) / "sam_flagstat.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if n > 0 else 4


# ---------------------------------------------------------------------------
# Reference / build check
# ---------------------------------------------------------------------------
def run_build_check(gtf_path: str, fasta_path: str, expected_build: str, out_dir: str) -> int:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gtf_chroms = set()
    fasta_chroms = set()
    gtf_p, fa_p = Path(gtf_path), Path(fasta_path)
    if not gtf_p.exists():
        return _fail(f"ERROR: GTF not found: {gtf_path}", 2)
    if not fa_p.exists():
        return _fail(f"ERROR: FASTA not found: {fasta_path}", 2)
    with gtf_p.open() as fh:
        for line in fh:
            if line.startswith("#"):
                if "genome-build" in line.lower() or "build" in line.lower():
                    pass
                if not line.strip() or line.startswith("#"):
                    # capture build comment
                    continue
            if not line.strip() or line.startswith("#"):
                continue
            gtf_chroms.add(line.split("\t", 1)[0])
    for rec in SeqIO.parse(str(fa_p), "fasta"):
        fasta_chroms.add(rec.id)
    # build hint from FASTA description / filename
    build_hint = expected_build
    if "hg19" in fa_p.name or "GRCh37" in fa_p.name:
        build_hint = "hg19"
    elif "hg38" in fa_p.name or "GRCh38" in fa_p.name:
        build_hint = "hg38"
    mismatch = sorted(gtf_chroms - fasta_chroms)
    build_ok = expected_build in (build_hint, fa_p.name)
    # more honest: compare expected vs inferred from filename
    inferred = "unknown"
    name = fa_p.name + str(fa_p)
    if "hg19" in name or "GRCh37" in name:
        inferred = "hg19"
    elif "hg38" in name or "GRCh38" in name:
        inferred = "hg38"
    report = {
        "tool": "build_check",
        "expected_build": expected_build,
        "inferred_fasta_build": inferred,
        "gtf_chroms": sorted(gtf_chroms),
        "fasta_chroms": sorted(fasta_chroms),
        "chrom_mismatch": mismatch,
        "build_match": inferred == expected_build or inferred == "unknown",
        "status": "PASS" if (not mismatch and (inferred in (expected_build, "unknown"))) else "FAIL",
    }
    (Path(out_dir) / "build_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 5


TOOL_MAP = {
    "fastq_qc": lambda a: run_fastq_qc(a.input, a.out),
    "gtf_validate": lambda a: run_gtf_validate(a.input, a.out, a.fasta),
    "faidx": lambda a: run_faidx(a.input, a.out),
    "mini_blast": lambda a: run_blast(a.query, a.db, a.out),
    "bed_stats": lambda a: run_bed_stats(a.input, a.out),
    "sam_flagstat": lambda a: run_sam_flagstat(a.input, a.out),
    "build_check": lambda a: run_build_check(a.input, a.fasta, a.build, a.out),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bio_tools")
    p.add_argument("tool", choices=sorted(TOOL_MAP))
    p.add_argument("--input", default="")
    p.add_argument("--out", default="out")
    p.add_argument("--fasta", default="")
    p.add_argument("--query", default="")
    p.add_argument("--db", default="")
    p.add_argument("--build", default="hg38")
    args = p.parse_args(argv)
    return TOOL_MAP[args.tool](args)


if __name__ == "__main__":
    sys.exit(main())

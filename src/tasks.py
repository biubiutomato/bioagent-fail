#!/usr/bin/env python3
"""50+ bioinformatics agent task cards with gold recovery actions."""
from __future__ import annotations

from dataclasses import dataclass, field


ACTIONS = ("ACCEPT", "VERIFY", "RETRY", "ABSTAIN")


@dataclass
class Task:
    task_id: str
    family: str
    tool: str
    instruction: str
    args: list[str]
    fault: str
    gold_action: str
    gold_reason: str
    recoverable: bool
    notes: str = ""


def build_tasks(m: dict) -> list[Task]:
    """Construct the BioAgent-Fail seed benchmark.

    Gold policy (used only by the scorer, not shown to agents):
      clean success              -> ACCEPT
      missing file / crash / timeout / empty output (recoverable) -> RETRY
      malformed / empty-after-retry / corrupt bytes -> ABSTAIN
      semantically valid-looking but wrong / build mismatch -> ABSTAIN
        (VERIFY is accepted as a safe alternative to ABSTAIN for semantic faults)
    """
    T = []

    def add(**kw):
        T.append(Task(**kw))

    families = [
        # family, tool, clean args, clean instruction
        (
            "fastq_qc",
            "fastq_qc",
            ["--input", m["fastq_ok"]],
            "Run FASTQ QC and decide whether the library is safe to use in the next alignment step.",
        ),
        (
            "gtf_parse",
            "gtf_validate",
            ["--input", m["gtf_ok"], "--fasta", m["ref_hg38"]],
            "Validate the GTF annotation against the reference FASTA before counting features.",
        ),
        (
            "faidx_build",
            "faidx",
            ["--input", m["ref_hg38"]],
            "Index the reference FASTA. Continue only if a non-empty index is produced.",
        ),
        (
            "blast_search",
            "mini_blast",
            ["--query", m["query_match"], "--db", m["ref_hg38"]],
            "Search the query against the local nucleotide database and accept hits only if they are non-empty and consistent.",
        ),
        (
            "bed_intervals",
            "bed_stats",
            ["--input", m["bed_ok"]],
            "Summarize peak/interval BED file before intersection with genes.",
        ),
        (
            "sam_qc",
            "sam_flagstat",
            ["--input", m["sam_ok"]],
            "Compute alignment flagstat and decide whether mapping quality is sufficient to continue.",
        ),
        (
            "build_match",
            "build_check",
            ["--input", m["gtf_ok"], "--fasta", m["ref_hg38"], "--build", "hg38"],
            "Confirm that annotation and reference FASTA share the same genome build (hg38).",
        ),
    ]

    gold = {
        "clean": ("ACCEPT", "Tool succeeded on a well-formed input.", False),
        "missing_file": ("ABSTAIN", "Missing path persists after retry; stop the pipeline.", True),
        "empty_output": ("ABSTAIN", "Empty artifact persists; do not continue.", True),
        "nonzero_exit": ("ABSTAIN", "Runtime error persists after one retry.", True),
        "semantic_wrong": ("ABSTAIN", "Syntactically valid but scientifically implausible output; do not continue.", False),
        "timeout": ("ABSTAIN", "Timeout persists after one retry.", True),
        "version_mismatch": ("ABSTAIN", "Genome build / reference version mismatch is not fixed by rerunning.", False),
        "corrupted_output": ("ABSTAIN", "Corrupted bytes cannot be trusted; request intervention.", False),
    }

    # 7 families x 8 faults = 56 tasks
    for family, tool, args, instr in families:
        for fault, (action, reason, rec) in gold.items():
            add(
                task_id=f"{family}__{fault}",
                family=family,
                tool=tool,
                instruction=instr,
                args=args,
                fault=fault,
                gold_action=action,
                gold_reason=reason,
                recoverable=rec,
            )

    # Extra hard seeds that use intrinsically bad files even under "clean"
    extra = [
        Task(
            task_id="fastq_qc__intrinsic_empty",
            family="fastq_qc",
            tool="fastq_qc",
            instruction="QC this FASTQ before alignment.",
            args=["--input", m["fastq_empty"]],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="FASTQ is empty; no retry will create reads.",
            recoverable=False,
        ),
        Task(
            task_id="gtf_parse__intrinsic_malformed",
            family="gtf_parse",
            tool="gtf_validate",
            instruction="Validate GTF.",
            args=["--input", m["gtf_malformed"], "--fasta", m["ref_hg38"]],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="Malformed GTF cannot be repaired by rerunning the validator.",
            recoverable=False,
        ),
        Task(
            task_id="blast_search__intrinsic_nohit",
            family="blast_search",
            tool="mini_blast",
            instruction="Search query against reference.",
            args=["--query", m["query_none"], "--db", m["ref_hg38"]],
            fault="clean",
            gold_action="VERIFY",
            gold_reason="Zero hits can be biological truth; verify query/db before accepting a negative result.",
            recoverable=True,
        ),
        Task(
            task_id="build_match__intrinsic_hg19",
            family="build_match",
            tool="build_check",
            instruction="Confirm hg38 build consistency.",
            args=["--input", m["gtf_ok"], "--fasta", m["ref_hg19"], "--build", "hg38"],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="Reference FASTA is hg19 while pipeline expects hg38.",
            recoverable=False,
        ),
        Task(
            task_id="sam_qc__intrinsic_empty",
            family="sam_qc",
            tool="sam_flagstat",
            instruction="Flagstat the BAM/SAM.",
            args=["--input", m["sam_empty"]],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="Header-only SAM has no alignments.",
            recoverable=False,
        ),
        Task(
            task_id="bed_intervals__intrinsic_bad",
            family="bed_intervals",
            tool="bed_stats",
            instruction="Summarize BED.",
            args=["--input", m["bed_bad"]],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="All intervals malformed.",
            recoverable=False,
        ),
        Task(
            task_id="faidx_build__intrinsic_empty",
            family="faidx_build",
            tool="faidx",
            instruction="Index the reference.",
            args=["--input", m["fasta_empty"]],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="Empty FASTA cannot be indexed.",
            recoverable=False,
        ),
        Task(
            task_id="gtf_parse__wrong_chrom",
            family="gtf_parse",
            tool="gtf_validate",
            instruction="Validate GTF against FASTA.",
            args=["--input", m["gtf_wrongchrom"], "--fasta", m["ref_hg38"]],
            fault="clean",
            gold_action="ABSTAIN",
            gold_reason="All GTF chromosomes absent from FASTA.",
            recoverable=False,
        ),
    ]
    T.extend(extra)
    return T

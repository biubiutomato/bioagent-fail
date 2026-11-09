#!/usr/bin/env python3
"""Generate small but syntactically real FASTQ / FASTA / GTF / BED / SAM fixtures."""
from __future__ import annotations

import random
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


DNA = "ACGT"


def _seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA) for _ in range(n))


def write_fastq(path: Path, n_reads: int, length: int, seed: int = 1, mean_q: int = 32) -> None:
    rng = random.Random(seed)
    recs = []
    for i in range(n_reads):
        s = _seq(length, rng)
        quals = [max(2, min(40, int(rng.gauss(mean_q, 4)))) for _ in range(length)]
        rec = SeqRecord(Seq(s), id=f"r{i+1}", description="")
        rec.letter_annotations["phred_quality"] = quals
        recs.append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(recs, str(path), "fastq")


def write_empty_fastq(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def write_corrupt_fastq(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("@r1\nACGTACGT\n+\n!!!!\n@r2\nNOTDNAXXX\n")


def write_fasta(path: Path, contigs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = [SeqRecord(Seq(seq), id=name, description="") for name, seq in contigs.items()]
    SeqIO.write(recs, str(path), "fasta")


def write_gtf(path: Path, records: list[tuple], header: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if header:
        lines.append(header)
    for chrom, source, feat, start, end, strand, gid, tid in records:
        attr = f'gene_id "{gid}"; transcript_id "{tid}";'
        lines.append(f"{chrom}\t{source}\t{feat}\t{start}\t{end}\t.\t{strand}\t.\t{attr}")
    path.write_text("\n".join(lines) + "\n")


def write_malformed_gtf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("chr1\tbad\nthis is not gtf\t1\t2\nchr1\tensembl\texon\tnot_int\t20\t.\t+\t.\tgene_id x;\n")


def write_bed(path: Path, intervals: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c}\t{s}\t{e}\t{name}" for c, s, e, name in intervals]
    path.write_text("\n".join(lines) + "\n")


def write_sam(path: Path, n: int, map_rate: float = 0.9, sq: str = "chr1", slen: int = 1000, seed: int = 7) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"@HD\tVN:1.6\tSO:unsorted", f"@SQ\tSN:{sq}\tLN:{slen}"]
    for i in range(n):
        mapped = rng.random() < map_rate
        flag = 0 if mapped else 4
        rname = sq if mapped else "*"
        pos = rng.randint(1, max(1, slen - 50)) if mapped else 0
        seq = _seq(50, rng)
        qual = "I" * 50
        cigar = "50M" if mapped else "*"
        lines.append(f"r{i+1}\t{flag}\t{rname}\t{pos}\t60\t{cigar}\t*\t0\t0\t{seq}\t{qual}")
    path.write_text("\n".join(lines) + "\n")


def generate_all(root: Path) -> dict:
    rng = random.Random(42)
    root.mkdir(parents=True, exist_ok=True)

    chr1 = _seq(800, rng)
    chr2 = _seq(600, rng)
    write_fasta(root / "ref_hg38.fa", {"chr1": chr1, "chr2": chr2})
    write_fasta(root / "ref_hg19.fa", {"chr1": chr1[:700], "chrM": _seq(200, rng)})
    write_fasta(root / "empty.fa", {})
    (root / "empty.fa").write_text("")

    # query that matches chr1 prefix
    write_fasta(root / "query_match.fa", {"q1": chr1[10:90]})
    write_fasta(root / "query_none.fa", {"q1": "T" * 80})
    write_fasta(root / "query_empty.fa", {})
    (root / "query_empty.fa").write_text("")

    write_fastq(root / "sample_ok.fq", n_reads=80, length=75, seed=1, mean_q=33)
    write_fastq(root / "sample_lowq.fq", n_reads=40, length=60, seed=2, mean_q=12)
    write_empty_fastq(root / "sample_empty.fq")
    write_corrupt_fastq(root / "sample_corrupt.fq")

    ok_gtf = []
    for i in range(12):
        start = 20 + i * 40
        ok_gtf.append(("chr1", "ensembl", "gene", start, start + 30, "+", f"G{i+1}", f"T{i+1}"))
        ok_gtf.append(("chr1", "ensembl", "exon", start, start + 30, "+", f"G{i+1}", f"T{i+1}"))
    write_gtf(root / "genes_hg38.gtf", ok_gtf, header="##genome-build GRCh38.p14")
    # syntactically valid but wrong chrom names vs hg38 fasta
    bad_build = [("chrX", "ensembl", "gene", 10, 50, "+", "GX", "TX")]
    write_gtf(root / "genes_wrongchrom.gtf", bad_build, header="##genome-build GRCh38")
    write_malformed_gtf(root / "genes_malformed.gtf")
    (root / "genes_empty.gtf").write_text("")

    write_bed(root / "peaks_ok.bed", [("chr1", 10, 40, "p1"), ("chr1", 100, 180, "p2"), ("chr2", 5, 25, "p3")])
    write_bed(root / "peaks_bad.bed", [("chr1", "x", "y", "p"), ("chr1", 80, 10, "neg")])
    (root / "peaks_empty.bed").write_text("")

    write_sam(root / "align_ok.sam", n=30, map_rate=0.85, sq="chr1", slen=800)
    write_sam(root / "align_unmapped.sam", n=20, map_rate=0.0, sq="chr1", slen=800)
    (root / "align_empty.sam").write_text("@HD\tVN:1.6\n")
    (root / "align_corrupt.sam").write_text("this is not SAM\n1\t2\n")

    manifest = {
        "fastq_ok": str(root / "sample_ok.fq"),
        "fastq_lowq": str(root / "sample_lowq.fq"),
        "fastq_empty": str(root / "sample_empty.fq"),
        "fastq_corrupt": str(root / "sample_corrupt.fq"),
        "ref_hg38": str(root / "ref_hg38.fa"),
        "ref_hg19": str(root / "ref_hg19.fa"),
        "fasta_empty": str(root / "empty.fa"),
        "query_match": str(root / "query_match.fa"),
        "query_none": str(root / "query_none.fa"),
        "gtf_ok": str(root / "genes_hg38.gtf"),
        "gtf_wrongchrom": str(root / "genes_wrongchrom.gtf"),
        "gtf_malformed": str(root / "genes_malformed.gtf"),
        "gtf_empty": str(root / "genes_empty.gtf"),
        "bed_ok": str(root / "peaks_ok.bed"),
        "bed_bad": str(root / "peaks_bad.bed"),
        "bed_empty": str(root / "peaks_empty.bed"),
        "sam_ok": str(root / "align_ok.sam"),
        "sam_unmapped": str(root / "align_unmapped.sam"),
        "sam_empty": str(root / "align_empty.sam"),
        "sam_corrupt": str(root / "align_corrupt.sam"),
    }
    return manifest

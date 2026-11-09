# BioAgent-Fail

Code, fixtures, and scored trajectories for:

> Ye Ye. **BioAgent-Fail: Benchmarking Verification, Recovery, and Abstention under Bioinformatics Tool Failures.**
> IEEE BIBM 2026 Workshops.

When should a tool-using bioinformatics agent accept a tool output, verify it, retry, or stop and hand the
step to a human? BioAgent-Fail answers this with 64 executable task cards over seven tool families
(FASTQ QC, GTF validation, FASTA indexing, a local BLAST-like search, BED statistics, SAM flagstat, and
genome-build checks) and eight observation conditions (clean, missing file, empty output, non-zero exit,
timeout, corrupted output, semantically wrong report, and build mismatch). Each card runs a real parser on
real files; a fault injector then rewrites what the agent observes.

## Layout

| Path | Contents |
|---|---|
| `src/tools.py` | Executable Biopython-based wrappers (CLI entry points) |
| `src/faults.py` | Fault injector over real tool observations |
| `src/tasks.py` | The 64 task cards and gold terminal actions |
| `src/agents.py` | Naive, retry-on-error, self-check, and verify-or-abstain policies |
| `src/run_experiment.py` | Scoring (exact match, unsafe acceptance, correct abstention, cost) |
| `src/run_fast.py` | Runs the four deterministic policies on all 64 cards |
| `src/run_llm_gpt41.py` | Function-calling runs for an OpenAI model (needs `OPENAI_API_KEY`) |
| `src/make_figures.py` | Regenerates the figures from `results/` |
| `src/datagen.py` | Generates the fixture files in `data/` |
| `data/` | Generated FASTQ, FASTA, GTF, BED, and SAM fixtures |
| `results/trial_results.csv` | 256 scored deterministic trajectories (4 policies × 64 cards) |
| `results/summary_by_policy.json`, `summary_by_fault.json` | Aggregate metrics |
| `results/llm_gpt-4.1_all64.json` | GPT-4.1 on all 64 cards (per-card decisions and reasons) |
| `results/llm_gpt41_slice.json`, `llm_gpt-4o_slice16.json` | GPT-4.1 and GPT-4o on the matched 16-card subset |
| `results/runs/` | Stored tool observations for each card |
| `figures/` | Figures in the paper |

## Reproduce

```bash
pip install -r requirements.txt
PYTHONPATH=src python src/run_fast.py      # deterministic policies -> results/
PYTHONPATH=src python src/make_figures.py  # figures/
```

LLM runs (optional; calls the OpenAI API):

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4.1   # or gpt-4o
PYTHONPATH=src python src/run_llm_gpt41.py
```

LLM runs were made at temperature 0 through the Chat Completions API; new runs may differ from the stored outputs.

## Headline numbers (64 cards)

| Policy | Exact | Unsafe accept | Exact (clean) | Unsafe (injected) |
|---|---|---|---|---|
| Naive | 0.109 | 0.891 | 0.467 | 1.000 |
| Retry-on-error | 0.109 | 0.891 | 0.467 | 1.000 |
| Verify-or-abstain | 0.969 | 0.031 | 1.000 | 0.041 |
| GPT-4.1 | 0.531 | 0.078 | 0.867 | 0.082 |

## Citation

```bibtex
@inproceedings{ye2026bioagentfail,
  author    = {Ye, Ye},
  title     = {BioAgent-Fail: Benchmarking Verification, Recovery, and Abstention under Bioinformatics Tool Failures},
  booktitle = {Proc. IEEE International Conference on Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).

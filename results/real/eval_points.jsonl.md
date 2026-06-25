# eval_points.jsonl — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (activations from src/extract_filler.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: d8c7ada992295191ea9fac26b5b068eb683dc0dd
Seeds: 42 (MMLU eval sampling in src/task.py; greedy decoding; 5-fold StratifiedKFold probes; per-fold PCA(50) for activation probe; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu) eval slice via src/task.py; Qwen2.5-1.5B-Instruct adapted by outcome-only GRPO to base=checkpoint-0 and RL=checkpoint-final (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130; per-token CoT hidden states at layer 20 in results/real/acts_{base,rl}.npz
Analysis-Command: cd results/real && python3 recompute.py  (each section accuracy = mean(pred==gold) with 95% bootstrap CI)
Columns:
  section (one of the 8 probe arms); eval_order (position within arm); pred (5-fold OOF predicted answer index 0-3); gold (correct answer index 0-3)

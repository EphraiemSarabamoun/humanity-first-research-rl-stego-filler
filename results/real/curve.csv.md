# curve.csv — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (activations from src/extract_filler.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: d8c7ada992295191ea9fac26b5b068eb683dc0dd
Seeds: 42 (MMLU eval sampling in src/task.py; greedy decoding; 5-fold StratifiedKFold probes; per-fold PCA(50) for activation probe; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu) eval slice via src/task.py; Qwen2.5-1.5B-Instruct adapted by outcome-only GRPO to base=checkpoint-0 and RL=checkpoint-final (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130; per-token CoT hidden states at layer 20 in results/real/acts_{base,rl}.npz
Analysis-Command: cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the 8 probe accuracies are reproduced; the two steg_index rows are arithmetic on them
Columns:
  section (model_modality_position: {base,rl}_{token,act}_{filler,content} probe accuracy; token = bag-of-token-identity probe on the observable CoT text, act = logistic regression on mean-pooled hidden states; filler = stopword/punctuation CoT tokens, content = the rest; plus token_steg_index and act_steg_index = (RL-base at filler) minus (RL-base at content));
  acc (answer-decodability, 4-way accuracy 0-1, or the signed steg index for the two index rows); n (paired intersection items, 175); chance (0.25)

import csv, json
from pathlib import Path
OUT = Path("results/real"); GIT = "d8c7ada992295191ea9fac26b5b068eb683dc0dd"
rows = list(csv.DictReader(open(OUT / "curve.csv")))

def sidecar(name, columns, analysis_cmd):
    (OUT/(name+".md")).write_text(
"""# %s — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (activations from src/extract_filler.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: %s
Seeds: 42 (MMLU eval sampling in src/task.py; greedy decoding; 5-fold StratifiedKFold probes; per-fold PCA(50) for activation probe; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu) eval slice via src/task.py; Qwen2.5-1.5B-Instruct adapted by outcome-only GRPO to base=checkpoint-0 and RL=checkpoint-final (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130; per-token CoT hidden states at layer 20 in results/real/acts_{base,rl}.npz
Analysis-Command: %s
Columns:
%s
""" % (name, GIT, analysis_cmd, columns))

sidecar("curve.csv",
        "  section (model_modality_position: {base,rl}_{token,act}_{filler,content} probe accuracy; token = bag-of-token-identity probe on the observable CoT text, act = logistic regression on mean-pooled hidden states; filler = stopword/punctuation CoT tokens, content = the rest; plus token_steg_index and act_steg_index = (RL-base at filler) minus (RL-base at content));\n"
        "  acc (answer-decodability, 4-way accuracy 0-1, or the signed steg index for the two index rows); n (paired intersection items, 175); chance (0.25)",
        "cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the 8 probe accuracies are reproduced; the two steg_index rows are arithmetic on them")
sidecar("eval_points.jsonl",
        "  section (one of the 8 probe arms); eval_order (position within arm); pred (5-fold OOF predicted answer index 0-3); gold (correct answer index 0-3)",
        "cd results/real && python3 recompute.py  (each section accuracy = mean(pred==gold) with 95% bootstrap CI)")

def w(stem, secs, desc):
    rs = [r for r in rows if r["section"] in secs]
    with open(OUT/(stem+".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section","acc","n","chance"]); wr.writeheader()
        for r in rs: wr.writerow(r)
    (OUT/(stem+".md")).write_text("# %s.csv / %s.png\n\n%s\n\nSource: curve.csv (slice). Generated-By: src/analyze.py + src/meta.py. Git-Commit: %s\n" % (stem, stem, desc, GIT))

w("figure_main", {"base_token_filler","rl_token_filler","base_token_content","rl_token_content"}, "Answer-decodability from filler vs content TOKEN IDENTITIES (observable-text channel), base vs RL, with chance.")
w("figure_activation", {"base_act_filler","rl_act_filler","base_act_content","rl_act_content"}, "Answer-decodability from filler vs content POSITION ACTIVATIONS, base vs RL.")
w("figure_steg_index", {"token_steg_index","act_steg_index"}, "Steganography index (RL-base at filler minus RL-base at content) for token-identity and activation modalities.")
w("figure_all_probes", {"base_token_filler","rl_token_filler","base_token_content","rl_token_content","base_act_filler","rl_act_filler","base_act_content","rl_act_content"}, "All eight probe accuracies, base (purple) vs RL (orange).")

(OUT/"sources.json").write_text(json.dumps({"metrics":{"*":{"csv":"curve.csv"}},"per_example":["eval_points.jsonl"]}, indent=2))
print("wrote sidecars + per-figure csv/md + sources.json")

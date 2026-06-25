"""analyze_filler.py — paired base-vs-RL answer-decodability from filler vs content
CoT tokens (identity channel) and positions (activations). Writes curve.csv,
eval_points.jsonl, analysis_summary.txt (via recompute.py), figures, sidecars-ready.
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task
LAB = {l: i for i, l in enumerate(task.LABELS)}
SEED, CHANCE = 42, 0.25


def load(out, label):
    d = np.load(out / ("acts_%s.npz" % label), allow_pickle=True)
    return {"item_id": d["item_id"], "gold": d["gold"], "fvec": d["fvec"], "cvec": d["cvec"],
            "fill_toks": d["fill_toks"], "cont_toks": d["cont_toks"]}


def cv_token(texts, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = make_pipeline(CountVectorizer(token_pattern=r"[^ ]+", min_df=2, binary=True),
                        LogisticRegression(max_iter=2000, C=1.0))
    return cross_val_predict(clf, texts, y, cv=skf)


def cv_act(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    k = min(50, X.shape[1], X.shape[0]-1)
    clf = make_pipeline(StandardScaler(), PCA(k, random_state=SEED), LogisticRegression(max_iter=2000))
    return cross_val_predict(clf, X, y, cv=skf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/real")
    args = ap.parse_args()
    out = Path(args.out_dir)
    B, R = load(out, "base"), load(out, "rl")
    # paired intersection on item_id
    inter = sorted(set(B["item_id"].tolist()) & set(R["item_id"].tolist()))
    print("[pair] base=%d rl=%d intersection=%d" % (len(B["item_id"]), len(R["item_id"]), len(inter)), flush=True)
    def sub(D):
        pos = {int(it): k for k, it in enumerate(D["item_id"].tolist())}
        ix = [pos[i] for i in inter]
        return {"gold": np.array([LAB[D["gold"][k]] for k in ix]),
                "fvec": D["fvec"][ix], "cvec": D["cvec"][ix],
                "fill": [D["fill_toks"][k] for k in ix], "cont": [D["cont_toks"][k] for k in ix]}
    b, r = sub(B), sub(R)

    ep, rows = [], []
    def probe(section, pred, gold):
        for k in range(len(gold)):
            ep.append({"section": section, "eval_order": k, "pred": int(pred[k]), "gold": int(gold[k])})
        rows.append({"section": section, "acc": float(np.mean(pred == gold)), "n": len(gold)})

    probe("base_token_filler", cv_token(b["fill"], b["gold"]), b["gold"])
    probe("rl_token_filler", cv_token(r["fill"], r["gold"]), r["gold"])
    probe("base_token_content", cv_token(b["cont"], b["gold"]), b["gold"])
    probe("rl_token_content", cv_token(r["cont"], r["gold"]), r["gold"])
    probe("base_act_filler", cv_act(b["fvec"], b["gold"]), b["gold"])
    probe("rl_act_filler", cv_act(r["fvec"], r["gold"]), r["gold"])
    probe("base_act_content", cv_act(b["cvec"], b["gold"]), b["gold"])
    probe("rl_act_content", cv_act(r["cvec"], r["gold"]), r["gold"])

    accd = {x["section"]: x["acc"] for x in rows}
    import csv as _csv
    with open(out / "curve.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["section", "acc", "n", "chance"])
        for x in rows:
            w.writerow([x["section"], "%.6f" % x["acc"], x["n"], CHANCE])
        # steganography indices (derived)
        tok_steg = (accd["rl_token_filler"]-accd["base_token_filler"]) - (accd["rl_token_content"]-accd["base_token_content"])
        act_steg = (accd["rl_act_filler"]-accd["base_act_filler"]) - (accd["rl_act_content"]-accd["base_act_content"])
        w.writerow(["token_steg_index", "%.6f" % tok_steg, len(inter), CHANCE])
        w.writerow(["act_steg_index", "%.6f" % act_steg, len(inter), CHANCE])
    with open(out / "eval_points.jsonl", "w") as f:
        for e in ep:
            f.write(json.dumps(e) + "\n")
    with open(out / "analysis_summary.txt", "w") as f:
        subprocess.run(["python3", "recompute.py"], cwd=str(out), stdout=f, check=True)
    make_figs(accd, len(inter), out)
    print("[analyze] token_steg=%.3f act_steg=%.3f" % (tok_steg, act_steg), flush=True)


def make_figs(a, n, out):
    # Fig1: token-identity filler decodability base vs RL (+content, chance)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    groups = ["filler tokens", "content tokens"]
    base_v = [a["base_token_filler"], a["base_token_content"]]
    rl_v = [a["rl_token_filler"], a["rl_token_content"]]
    x = np.arange(2); wd = 0.38
    ax.bar(x-wd/2, base_v, wd, label="base", color="#7570b3")
    ax.bar(x+wd/2, rl_v, wd, label="RL", color="#d95f02")
    ax.axhline(CHANCE, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("answer-decodability (token-identity probe)")
    ax.set_ylim(0, max(0.6, max(base_v+rl_v)+0.1)); ax.set_title("Does the observable filler-token TEXT predict the answer more after RL? (n=%d)" % n)
    ax.legend(); fig.tight_layout(); fig.savefig(out/"figure_main.png", dpi=150); plt.close(fig)

    # Fig2: activation probe filler vs content base vs RL
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    base_v = [a["base_act_filler"], a["base_act_content"]]; rl_v = [a["rl_act_filler"], a["rl_act_content"]]
    ax.bar(x-wd/2, base_v, wd, label="base", color="#7570b3"); ax.bar(x+wd/2, rl_v, wd, label="RL", color="#d95f02")
    ax.axhline(CHANCE, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(["filler positions", "content positions"]); ax.set_ylabel("answer-decodability (activation probe)")
    ax.set_ylim(0, 1.0); ax.set_title("Filler-position vs content-position activation answer-decodability")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"figure_activation.png", dpi=150); plt.close(fig)

    # Fig3: steganography index (delta_filler - delta_content)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    tok_steg = (a["rl_token_filler"]-a["base_token_filler"]) - (a["rl_token_content"]-a["base_token_content"])
    act_steg = (a["rl_act_filler"]-a["base_act_filler"]) - (a["rl_act_content"]-a["base_act_content"])
    ax.bar([0,1], [tok_steg, act_steg], 0.5, color=["#1b9e77","#e7298a"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0,1]); ax.set_xticklabels(["token-identity", "activation"])
    ax.set_ylabel("steganography index\n(RL-base) filler minus (RL-base) content")
    ax.set_title("Steganography index (>0 = filler gained more answer info than content)")
    fig.tight_layout(); fig.savefig(out/"figure_steg_index.png", dpi=150); plt.close(fig)

    # Fig4: all eight probe accuracies
    fig, ax = plt.subplots(figsize=(8, 4.4))
    keys = ["base_token_filler","rl_token_filler","base_token_content","rl_token_content",
            "base_act_filler","rl_act_filler","base_act_content","rl_act_content"]
    ax.bar(range(8), [a[k] for k in keys], color=["#7570b3","#d95f02"]*4)
    ax.axhline(CHANCE, color="k", ls=":", lw=1)
    ax.set_xticks(range(8)); ax.set_xticklabels([k.replace("_","\n") for k in keys], fontsize=6)
    ax.set_ylabel("answer-decodability"); ax.set_ylim(0,1.0); ax.set_title("All probes (purple=base, orange=RL)")
    fig.tight_layout(); fig.savefig(out/"figure_all_probes.png", dpi=150); plt.close(fig)
    print("[figures] wrote 4", flush=True)


if __name__ == "__main__":
    main()

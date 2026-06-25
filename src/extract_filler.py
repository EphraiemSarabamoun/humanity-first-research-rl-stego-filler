"""extract_filler.py — does outcome-only RL push answer info into filler CoT tokens?

For base (checkpoint-0) and RL (checkpoint-final) we let the model produce its own
CoT + answer, classify each CoT token as filler (stopword/punctuation) or content,
and save (a) the filler / content token-id strings for a token-identity channel
probe and (b) the mean-pooled hidden state over filler / content positions at a
fixed layer for a representation probe. A stable item_id lets the analysis pair
the two models on the items both keep.

Reuses src/task.py for prompt + answer parsing. Python 3.10 (system python3).
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_FRAC = 0.7

STOPWORDS = set("""a an the of to in on at for and or but if then so as is are was were be been being
this that these those it its it's we us our you your they them their he she his her i me my
let lets let's so thus hence therefore because since while which who whom whose what when where why how
will would can could should may might must shall do does did done have has had not no nor only just
than too very can't cannot also however thus first second third next finally now here there with by from
into over under about between each both either neither one two three four five option options answer
scenario step given consider determine analyze let's""".split())


def is_filler(tokstr):
    s = tokstr.strip().lower()
    if s == "":
        return True
    if not re.search(r"[a-z0-9]", s):  # pure punctuation / symbols
        return True
    return s in STOPWORDS


def load_model(adapter):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter and adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.no_grad()
def gen(model, tok, prompts, max_new, batch=16):
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


@torch.no_grad()
def run_model(label, adapter, items, out_dir, max_new_self=256):
    model, tok = load_model(adapter)
    n_layers = model.config.num_hidden_layers
    L = max(1, int(round(LAYER_FRAC * n_layers)))
    base_prompts = [task.make_prompt_text(tok, it) for it in items]
    resp = gen(model, tok, base_prompts, max_new_self)

    recs = []
    for idx, (it, prompt, r) in enumerate(zip(items, base_prompts, resp)):
        A0 = task.extract_answer(r)
        m = re.search(r"Answer:", r, re.IGNORECASE)
        cot = r[:m.start()] if m else r
        if A0 is None or len(cot.strip()) == 0:
            continue
        # re-encode prompt + cot, capture per-token hidden states over the cot span
        full = prompt + cot
        enc = tok(full, return_tensors="pt", truncation=True, max_length=1280).to(model.device)
        plen = tok(prompt, return_tensors="pt", truncation=True, max_length=1280)["input_ids"].shape[1]
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[L][0]  # [T, d]
        ids = enc["input_ids"][0].tolist()
        fill_pos, cont_pos, fill_toks, cont_toks = [], [], [], []
        for pos in range(plen, len(ids)):
            ts = tok.decode([ids[pos]])
            (fill_pos if is_filler(ts) else cont_pos).append(pos)
            (fill_toks if is_filler(ts) else cont_toks).append(ts.strip().lower())
        if len(fill_pos) < 2 or len(cont_pos) < 2:
            continue
        fvec = h[fill_pos].float().mean(0).cpu().numpy()
        cvec = h[cont_pos].float().mean(0).cpu().numpy()
        recs.append({"item_id": idx, "gold": it["gold_label"], "A0": A0,
                     "n_fill": len(fill_pos), "n_cont": len(cont_pos),
                     "fill_toks": " ".join(fill_toks), "cont_toks": " ".join(cont_toks),
                     "fvec": fvec.astype(np.float32), "cvec": cvec.astype(np.float32)})
    # save
    np.savez_compressed(out_dir / ("acts_%s.npz" % label),
                        item_id=np.array([r["item_id"] for r in recs]),
                        gold=np.array([r["gold"] for r in recs], dtype=object),
                        A0=np.array([r["A0"] for r in recs], dtype=object),
                        fvec=np.stack([r["fvec"] for r in recs]),
                        cvec=np.stack([r["cvec"] for r in recs]),
                        fill_toks=np.array([r["fill_toks"] for r in recs], dtype=object),
                        cont_toks=np.array([r["cont_toks"] for r in recs], dtype=object),
                        layer=L)
    acc = np.mean([1 if r["A0"] == r["gold"] else 0 for r in recs]) if recs else 0.0
    print("[%s] kept=%d layer=%d self-acc=%.3f mean_fill=%.1f mean_cont=%.1f" % (
        label, len(recs), L, acc, np.mean([r["n_fill"] for r in recs]), np.mean([r["n_cont"] for r in recs])), flush=True)
    del model; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_adapter", default="../rl-selfcot-causal/runs/rl/checkpoint-0")
    ap.add_argument("--rl_adapter", default="../rl-selfcot-causal/runs/rl/checkpoint-final")
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/real")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    items = task.load_items("mmlu", "eval", n=args.n_eval, seed=args.seed)
    print("[eval] %d MMLU items" % len(items), flush=True)
    run_model("base", args.base_adapter, items, out)
    run_model("rl", args.rl_adapter, items, out)
    print("[done] extraction", flush=True)


if __name__ == "__main__":
    main()

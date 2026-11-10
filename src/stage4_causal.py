"""aşama 4, neden-sonuç testi: ara katlarda ingilizce yönünü silince çeviri bozuluyor mu?

logit lens qwen'in orta katlarda ingilizceye yakın bir yerden geçtiğini gösterdi. bu sadece okuma.
bayazit ve arkadaşları (2026) bu okumanın ingilizceyi abarttığını, müdahale ile test edilmediğini söylüyor.
burada müdahale ediyorum.

1. her dil için bir yön buluyorum. modele sadece o dildeki kelimelerden oluşan listeler verip her katta iç
   durumun ortalamasını alıyorum. bir dilin yönü = o dilin ortalaması - dört dilin ortalaması.
   bu yönleri kelimelerin ilk yarısından çıkarıyorum, deneyi ikinci yarısında yapıyorum.
2. çeviri sırasında ara katlardan bu yönü siliyorum (h - (h·u)u).
3. son katta doğru cevaba verilen olasılığa ve doğru başlayan cevap sayısına bakıyorum.

beklenti: model ingilizceden geçiyorsa ingilizceyi silmek, görevle ilgisi olmayan bir dili ya da rastgele
bir yönü silmekten çok daha fazla zarar verir.

stage4_latent.py ile aynı klasörde çalışıyor.
"""
import argparse, json, os, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, LANG_NAME, start_tokens, build_prompt

LANGS = ["en", "tr", "fr", "de"]


@torch.no_grad()
def language_directions(model, tok, idx, device, n_shuffle=6):
    """her katta her dil için birim yön. listeyi 6 farklı sırayla verip bütün token'ların ortalamasını alıyorum."""
    means = {}
    for l in LANGS:
        acc, n = None, 0
        for s in range(n_shuffle):
            ws = [WORDS[i][COL[l]] for i in idx]
            random.Random(s).shuffle(ws)
            text = f"{LANG_NAME[l]}: " + ", ".join(f'"{w}"' for w in ws)
            enc = tok(text, return_tensors="pt").to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            H = torch.stack([h[0, 4:].float().mean(0) for h in hs])  # ilk birkaç token dil adı, onları atlıyorum
            acc = H if acc is None else acc + H
            n += 1
        means[l] = acc / n
    avg = sum(means.values()) / len(LANGS)
    dirs = {l: (means[l] - avg) / (means[l] - avg).norm(dim=-1, keepdim=True) for l in LANGS}
    g = torch.Generator().manual_seed(0)
    r = torch.randn(avg.shape, generator=g).to(avg.device)
    dirs["rastgele"] = r / r.norm(dim=-1, keepdim=True)
    return dirs  # her biri [katman+1, gizli boyut], indeks k = k. katmanın çıkışı (0 = gömme)


def ablate(model, layers, U):
    """verilen katmanların çıkışından o katmana ait yönü siliyorum."""
    hooks = []
    for k in layers:
        u = U[k + 1]

        def hook(mod, inp, out, u=u):
            h = out[0] if isinstance(out, tuple) else out
            uu = u.to(h.dtype)
            h = h - (h @ uu).unsqueeze(-1) * uu
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hooks.append(model.model.layers[k].register_forward_hook(hook))
    return hooks


@torch.no_grad()
def evaluate(model, tok, vocab_str, test_idx, src, tgt, device):
    """son katta hedef cevabın olasılığı, doğru başlayan cevap sayısı ve ara katlardaki ingilizce."""
    norm, head = model.model.norm, model.lm_head
    p_tgt, ok, en_mid, n = 0.0, 0, 0.0, 0
    for i in test_idx:
        w = WORDS[i]
        S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in {src, tgt, "en"}}
        if any(S[x] & S[y] for x in S for y in S if x < y):
            continue
        enc = tok(build_prompt(src, tgt, i), return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        L = len(hs)
        p = torch.softmax(out.logits[0, -1].float(), -1)
        p_tgt += float(p[list(S[tgt])].sum())
        ok += int(int(p.argmax()) in S[tgt])
        mids = range(int(L * .6), int(L * .9) + 1)
        en_mid += max(float(torch.softmax(head(norm(hs[k][0, -1])).float(), -1)[list(S["en"])].sum()) for k in mids)
        n += 1
    return {"n": n, "p_hedef": p_tgt / n, "dogru": ok, "ara_ingilizce": en_mid / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--pairs", default="fr-tr,de-tr,tr-fr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    n_layers = len(model.model.layers)
    layers = list(range(int(n_layers * .5), int(n_layers * .9)))  # ingilizcenin göründüğü ara katlar

    half = len(WORDS) // 2
    dir_idx, test_idx = list(range(half)), list(range(half, len(WORDS)))
    U = language_directions(model, tok, dir_idx, a.device)

    res = {"model": a.model, "layers": layers, "runs": []}
    name = a.model.split("/")[-1]
    print(f"{name}: {n_layers} katman, silinen katmanlar {layers[0]}-{layers[-1]}")
    print(f"{'yön':6s} {'silinen':9s} {'n':>3s} {'p(hedef)':>9s} {'doğru':>6s} {'ara ing.':>9s}")
    for pair in a.pairs.split(","):
        src, tgt = pair.split("-")
        for what in ["yok", "en", "tr", "fr", "de", "rastgele"]:
            hooks = ablate(model, layers, U[what]) if what != "yok" else []
            try:
                r = evaluate(model, tok, vocab_str, test_idx, src, tgt, a.device)
            finally:
                for h in hooks:
                    h.remove()
            r.update(pair=pair, silinen=what)
            res["runs"].append(r)
            print(f"{pair:6s} {what:9s} {r['n']:3d} {r['p_hedef']:9.3f} {r['dogru']:6d} {r['ara_ingilizce']:9.3f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/stage4_causal_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

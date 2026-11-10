"""aşama 4, yama testi: modelin içindeki kavram ingilizce biçimde mi duruyor?

dumas ve arkadaşlarının (2025) yama yönteminin küçük hali.
- verici: almancadan fransızcaya çeviri, kavram c (örneğin "Sonne" -> "soleil")
- alıcı: fransızcadan türkçeye çeviri, başka bir kavram c' (örneğin "chien" -> "köpek")
vericinin k. katmandaki son token durumunu alıcının aynı katmanına yapıştırıp alıcının çıkışına bakıyorum.

görevlerin hiçbirinde ingilizce yok. yama sonrası c'nin ingilizcesi ("sun") çıkıyorsa model kavramı
içeride ingilizce biçimde tutuyor demektir. c'nin türkçesi ("güneş") çıkıyorsa kavram dilden bağımsız
taşınmış, dili alıcının bağlamı belirlemiş demektir.

stage4_latent.py ile aynı klasörde çalışıyor.
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, start_tokens, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--donor", default="de-fr")
    ap.add_argument("--recv", default="fr-tr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ds, dt = a.donor.split("-")
    rs, rt = a.recv.split("-")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    n_layers = len(model.model.layers)

    # dört dilde de birbirinden ayrılabilen kelimeler
    langs = sorted({"en", ds, dt, rs, rt})
    S, good = {}, []
    for i, w in enumerate(WORDS):
        s = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in langs}
        if not any(s[x] & s[y] for x in langs for y in langs if x < y):
            S[i] = s; good.append(i)
    pairs = [(good[k], good[(k + 1) % len(good)]) for k in range(len(good))]  # (verici kavramı, alıcı kavramı)

    keys = ["c_en", "c_" + rt, "c_" + dt, "c2_" + rt]
    sums = [{k: 0.0 for k in keys} for _ in range(n_layers)]
    with torch.no_grad():
        for c, c2 in pairs:
            hs = model(**tok(build_prompt(ds, dt, c), return_tensors="pt").to(a.device), output_hidden_states=True).hidden_states
            enc = tok(build_prompt(rs, rt, c2), return_tensors="pt").to(a.device)
            for k in range(n_layers):
                donor = hs[k + 1][0, -1]

                def hook(mod, inp, out, donor=donor):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone(); h[0, -1] = donor.to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hd = model.model.layers[k].register_forward_hook(hook)
                try:
                    p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
                finally:
                    hd.remove()
                pr = lambda ids: float(p[list(ids)].sum())
                sums[k]["c_en"] += pr(S[c]["en"])
                sums[k]["c_" + rt] += pr(S[c][rt])
                sums[k]["c_" + dt] += pr(S[c][dt])
                sums[k]["c2_" + rt] += pr(S[c2][rt])

    n = len(pairs)
    res = [{k: v / n for k, v in s.items()} for s in sums]
    name = a.model.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.model, "donor": a.donor, "recv": a.recv, "n_pairs": n, "layers": res},
              open(f"{a.out}/stage4_patch_{name}_{a.donor}_into_{a.recv}.json", "w", encoding="utf-8"), indent=1)
    print(f"{name}: verici {a.donor}, alıcı {a.recv}, {n} çift")
    print("katman  " + "  ".join(f"{k:>8s}" for k in keys))
    for k, r in enumerate(res):
        print(f"{k:5d}   " + "  ".join(f"{r[x]:8.3f}" for x in keys))


if __name__ == "__main__":
    main()

"""aşama 4, kavram enjeksiyonu: modelin kavram alanı ingilizce biçimli mi?

model "chien" -> "köpek" çevirisi yaparken ara katmanlarda son token'a başka bir kavramın kelime vektörünü
ekliyorum (çıkış katmanındaki satırı, yani modelin o kelimeyi söylemek için kullandığı yönü).
üç seçenek, aynı uzunlukta: kavramın ingilizce ("sun"), almanca ("Sonne") ve türkçe ("güneş") kelimesi.
sonra modelin enjekte edilen kavramı türkçe söyleme olasılığına bakıyorum.

asıl karşılaştırma ingilizce ile almanca arasında. ikisi de türkçe çıkışa doğrudan katkı yapamaz, "güneş"
çıkması için modelin kavramı işlemesi gerekir. türkçe vektör ise "güneş" token'ını doğrudan yükseltir,
o yüzden sadece üst sınır olarak duruyor. ingilizce almancadan çok daha etkiliyse modelin kavram alanı
ingilizce kelimelere daha yakın demektir. adil olsun diye ingilizcesi ve almancası tek token olan kelimeleri kullanıyorum. türkçe vektör için kelimenin ilk
token'ını alıyorum. doz küçük tutuluyor, büyük dozda model enjekte edilen kelimeyi doğrudan tekrar ediyor.
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, start_tokens, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--src", default="fr")
    ap.add_argument("--alphas", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    n_layers = len(model.model.layers)
    W = model.lm_head.weight.float()

    one = lambda w: tok(w, add_special_tokens=False)["input_ids"]
    good, S = [], {}
    for i, w in enumerate(WORDS):
        if len(one(w[0])) == 1 and len(one(w[3])) == 1:
            s = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in ("en", "tr", a.src)}
            if not any(s[x] & s[y] for x in s for y in s if x < y):
                good.append(i); S[i] = s
    pairs = [(good[k], good[(k + 1) % len(good)]) for k in range(len(good))]  # (enjekte edilen, alıcı)

    # katman pencereleri: erken, orta, geç
    win = {"erken": range(int(n_layers * .2), int(n_layers * .45)),
           "orta": range(int(n_layers * .45), int(n_layers * .7)),
           "geç": range(int(n_layers * .7), n_layers)}

    @torch.no_grad()
    def run(c, c2, layers, vec, alpha):
        enc = tok(build_prompt(a.src, "tr", c2), return_tensors="pt").to(a.device)
        hooks = []
        for k in layers:
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[0, -1] += (alpha * h[0, -1].float().norm() * vec).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hooks.append(model.model.layers[k].register_forward_hook(hook))
        try:
            p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
        finally:
            for h in hooks:
                h.remove()
        return float(p[list(S[c]["tr"])].sum()), float(p[list(S[c]["en"])].sum()), float(p[list(S[c2]["tr"])].sum())

    name = a.model.split("/")[-1]
    out = {"model": a.model, "n_pairs": len(pairs), "words": [WORDS[i][1] for i in good], "runs": []}
    print(f"{name}: {len(pairs)} çift, kelimeler {[WORDS[i][1] for i in good]}")
    base = [run(c, c2, [], None, 0) for c, c2 in pairs]
    print(f"enjeksiyonsuz: p(c türkçe)={sum(b[0] for b in base)/len(base):.3f}  p(alıcı cevabı)={sum(b[2] for b in base)/len(base):.3f}")
    print(f"{'pencere':7s} {'alfa':>5s} {'vektör':7s} {'p(c tr)':>8s} {'p(c en)':>8s} {'p(alıcı)':>9s} {'c tr kazandı':>13s}")
    for wname, layers in win.items():
        for alpha in map(float, a.alphas.split(",")):
            for vname, col in [("ingilizce", 0), ("almanca", 3), ("türkçe", 1)]:
                rs = []
                for c, c2 in pairs:
                    t = one(WORDS[c][col])[0]
                    v = W[t] / W[t].norm()
                    rs.append(run(c, c2, list(layers), v, alpha))
                n = len(rs)
                r = {"pencere": wname, "alfa": alpha, "vektor": vname,
                     "p_c_tr": sum(x[0] for x in rs) / n, "p_c_en": sum(x[1] for x in rs) / n,
                     "p_alici": sum(x[2] for x in rs) / n, "kazandi": sum(x[0] > x[2] for x in rs)}
                out["runs"].append(r)
                print(f"{wname:7s} {alpha:5.1f} {vname:9s} {r['p_c_tr']:8.3f} {r['p_c_en']:8.3f} {r['p_alici']:9.3f} {r['kazandi']:6d}/{n}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/stage4_inject_{name}_{a.src}-tr.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""aşama 4, kaynak kelime kontrolü: 4b'de ingilizceyi silince bozulan şey gerçekten ingilizce mi?

stage4_bridge.py'de qwen3.5-4b'nin ara katlarından o kavramın ingilizce kelimesinin çıkış vektörünü silince türkçe
çeviri bozuldu (fr→tr 77'den 64'e). ama ingilizce kelime çoğu zaman kaynak kelimeye benziyor (almanca "haus" ile
"house"). model çeviri yaparken kaynak kelimenin bilgisini taşıyor olmalı. ingilizce vektörü silerken o bilgiyi de
silmiş olabilirim. burada bunu ayırıyorum.

her kelime için ara katlardan şunları ayrı ayrı siliyorum:
- hedef: türkçe cevabın kendisi (olumlu kontrol, yöntemin bir şey bozabildiğini gösteriyor)
- kaynak: çevrilen kelimenin kendisi (fransızca ya da almanca)
- en: kavramın ingilizcesi, olduğu gibi
- en_dik2: ingilizcenin hedefe ve kaynağa ait payı çıkarılmış hali. asıl test bu.
- kontrol_dik2: görevde olmayan bir dilin (almanca ya da fransızca) kelimesi, aynı şekilde arındırılmış
- baska_dik2: ilgisiz bir kavramın ingilizcesi, aynı şekilde arındırılmış

en_dik2 hâlâ zarar veriyorsa etki kaynak kelimeden gelmiyor, ingilizcenin kendisinden geliyor.

kelime kelime sonuç tutuyorum, böylece farkın şans eseri olup olmadığını da test edebiliyorum:
- doğru/yanlış için mcnemar testi (sadece iki koşulda farklı sonuç veren kelimelere bakıyor)
- olasılık farkı için eşleştirilmiş bootstrap (kelimeleri yeniden örnekleyip ortalama farkın güven aralığı)

stage4_latent.py ve stage4_bridge.py ile aynı klasörde çalışıyor.
"""
import argparse, json, math, os, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, start_tokens, build_prompt
from stage4_bridge import first_ids, word_hooks

CONDS = ["yok", "hedef", "kaynak", "en", "en_dik2", "kontrol_dik2", "baska_dik2"]


def mcnemar(a, b):
    """iki koşulun doğru/yanlış listeleri. kesin (binom) iki yönlü p değeri."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)  # a doğru, b yanlış
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def bootstrap(a, b, iters=10000, seed=0):
    """ortalama(a - b) ve %95 güven aralığı, kelimeler yeniden örneklenerek."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(d) / n, means[int(iters * .025)], means[int(iters * .975)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--pairs", default="fr-tr,de-tr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    n_layers = len(model.model.layers)
    layers = list(range(int(n_layers * .5), int(n_layers * .9)))
    norm, head = model.model.norm, model.lm_head
    Wg = head.weight.float() * norm.weight.float()
    vec = lambda word: Wg[first_ids(tok, word)].T  # [d, 1 ya da 2]
    name = a.model.split("/")[-1]
    res = {"model": a.model, "layers": layers, "pairs": {}}
    print(f"{name}: silinen katmanlar {layers[0]}-{layers[-1]}")

    for pair in a.pairs.split(","):
        src, tgt = pair.split("-")
        ctrl = "de" if src != "de" else "fr"
        rows = []
        for i, w in enumerate(WORDS):
            S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in {src, tgt, "en", ctrl}}
            if any(S[x] & S[y] for x in S for y in S if x < y):
                continue
            other = WORDS[(i + len(WORDS) // 2) % len(WORDS)][COL["en"]]
            enc = tok(build_prompt(src, tgt, i), return_tensors="pt").to(a.device)
            T, Sv = vec(w[COL[tgt]]), vec(w[COL[src]])
            base = torch.cat([T, Sv], 1)
            nb = torch.linalg.matrix_rank(base).item()

            def clean(X):
                """X'ten hedefin ve kaynağın kapsadığı alanı çıkar, kalan payın oranını da döndür."""
                Q, _ = torch.linalg.qr(torch.cat([base, X], 1))
                V = Q[:, nb:]
                Qb, _ = torch.linalg.qr(base)
                left = (X - Qb @ (Qb.T @ X)).norm() / X.norm()
                return V, float(left)

            Vs, kalan = {}, {}
            Vs["hedef"] = torch.linalg.qr(T)[0]
            Vs["kaynak"] = torch.linalg.qr(Sv)[0]
            Vs["en"] = torch.linalg.qr(vec(w[COL["en"]]))[0]
            Vs["en_dik2"], kalan["en"] = clean(vec(w[COL["en"]]))
            Vs["kontrol_dik2"], kalan["kontrol"] = clean(vec(w[COL[ctrl]]))
            Vs["baska_dik2"], kalan["baska"] = clean(vec(other))
            r = {"word": w, "kalan": kalan}
            for c in CONDS:
                hooks = word_hooks(model, layers, Vs[c]) if c != "yok" else []
                try:
                    p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
                finally:
                    for h in hooks:
                        h.remove()
                r[c] = {"p": float(p[list(S[tgt])].sum()), "dogru": int(int(p.argmax()) in S[tgt])}
            rows.append(r)

        n = len(rows)
        stats = {}
        print(f"\n{pair} ({n} kelime, kontrol dili {ctrl})")
        print(f"  {'silinen':13s} {'doğru':>7s} {'p(hedef)':>9s}   yoka göre p düşüşü [%95 aralık]   mcnemar (yok doğru/bu doğru)  p")
        for c in CONDS:
            dog = [r[c]["dogru"] for r in rows]
            pp = [r[c]["p"] for r in rows]
            line = f"  {c:13s} {sum(dog):4d}/{n:<3d} {sum(pp) / n:9.3f}"
            if c != "yok":
                m, lo, hi = bootstrap([r["yok"]["p"] for r in rows], pp)
                n01, n10, pm = mcnemar([r["yok"]["dogru"] for r in rows], dog)
                stats[c] = {"p_farki": m, "aralik": [lo, hi], "mcnemar": [n01, n10, pm]}
                line += f"   {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   {n01:3d} / {n10:<3d}  {pm:.4f}"
            print(line)
        # asıl karşılaştırma: arındırılmış ingilizce, arındırılmış kontrollere karşı
        for c in ("kontrol_dik2", "baska_dik2"):
            m, lo, hi = bootstrap([r[c]["p"] for r in rows], [r["en_dik2"]["p"] for r in rows])
            n01, n10, pm = mcnemar([r[c]["dogru"] for r in rows], [r["en_dik2"]["dogru"] for r in rows])
            stats[f"en_dik2_vs_{c}"] = {"p_farki": m, "aralik": [lo, hi], "mcnemar": [n01, n10, pm]}
            print(f"  en_dik2, {c}'ye göre: p düşüşü {m:+.3f} [{lo:+.3f}, {hi:+.3f}], mcnemar {n01}/{n10} p={pm:.4f}")
        for k in ("en", "kontrol", "baska"):
            print(f"  {k} vektörünün hedef+kaynak dışında kalan payı: ort. {sum(r['kalan'][k] for r in rows) / n:.3f}")
        res["pairs"][pair] = {"n": n, "ctrl": ctrl, "stats": stats, "rows": rows}

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/stage4_source_{name}.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

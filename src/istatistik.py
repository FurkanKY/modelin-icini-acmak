"""eleştiriye cevap veren istatistikler, gpu gerekmiyor.

1. aşama 4: göreli etki (p düşüşü / taban p) için kelimeler yeniden örneklenerek %95 güven aralığı.
   en_dik2 ve kontrol_dik2'nin göreli etkisi arasındaki fark da aynı örneklemle.
2. aşama 3: takılma geçişleri için kesin mcnemar. her deneme kendi tabanıyla, aynı 100 soru.

çalıştırma: python src/istatistik.py
"""
import json, math, os, random

def mcnemar(a, b):
    """stage4_source.mcnemar ile aynı (torch'suz çalışsın diye burada)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n
    return n01, n10, min(1.0, p)

D = "data"


def rel_boot(rows, cond, ref="kontrol_dik2", iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(rows)
    b = [r["yok"]["p"] for r in rows]
    c = [r[cond]["p"] for r in rows]
    k = [r[ref]["p"] for r in rows]

    def stat(idx):
        sb = sum(b[i] for i in idx)
        return (sb - sum(c[i] for i in idx)) / sb, (sum(k[i] for i in idx) - sum(c[i] for i in idx)) / sb

    full = stat(range(n))
    samp = [stat([rng.randrange(n) for _ in range(n)]) for _ in range(iters)]
    out = []
    for j in range(2):
        s = sorted(x[j] for x in samp)
        out.append((full[j], s[int(iters * .025)], s[int(iters * .975)]))
    return out


def main():
    print("## aşama 4: göreli etki (p düşüşü / taban p), %95 bootstrap aralığı\n")
    print("| model | çift | n | en_dik2 göreli | kontrol_dik2 göreli | en_dik2 − kontrol_dik2 |")
    print("|---|---|---|---|---|---|")
    for m in ("0.8B", "2B", "4B"):
        d = json.load(open(f"{D}/stage4_source_Qwen3.5-{m}.json", encoding="utf-8"))
        for pair, v in d["pairs"].items():
            rows = v["rows"]
            (e, elo, ehi), (df, dlo, dhi) = rel_boot(rows, "en_dik2")
            (k, klo, khi), _ = rel_boot(rows, "kontrol_dik2")
            f = lambda a, lo, hi: f"%{100 * a:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"
            print(f"| {m} | {pair} | {len(rows)} | {f(e, elo, ehi)} | {f(k, klo, khi)} | {f(df, dlo, dhi)} puan |")

    print("\n## aşama 3: takılma geçişleri, kesin mcnemar\n")
    print("| deneme | taban | takılan (taban → deneme) | takıldı→bitti | bitti→takıldı | p |")
    print("|---|---|---|---|---|---|")
    jobs = [("steer03", "base"), ("steer06", "base"),
            ("rep_a10", "base_v2"), ("rep_a15", "base_v2"), ("rep_p50c", "base_v2"),
            ("rand1_ters", "base_v2"), ("rand2_ters", "base_v2"), ("rand1_guclu", "base_v2"), ("rand2_guclu", "base_v2"),
            ("think_newrep_a10", "think_base_v2"), ("think_newrep_a15", "think_base_v2"),
            ("ext_steer03", "ext_base")]
    for j, b in jobs:
        pj, pb = f"{D}/stage3_{j}.json", f"{D}/stage3_{b}.json"
        if not (os.path.exists(pj) and os.path.exists(pb)):
            continue
        J, B = json.load(open(pj, encoding="utf-8")), json.load(open(pb, encoding="utf-8"))
        sb, sj = [not r["finished"] for r in B], [not r["finished"] for r in J]
        n01, n10, p = mcnemar(sb, sj)  # n01: tabanda takılıp denemede biten
        print(f"| {j} | {b} | {sum(sb)} → {sum(sj)} | {n01} | {n10} | {p:.4f} |")


if __name__ == "__main__":
    main()

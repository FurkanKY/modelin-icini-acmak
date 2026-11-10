"""aşama 4, kaynak kelime kontrolünün ikinci ayağı: etki sadece birbirine benzeyen kelimelerde mi var?

stage4_source.py'nin kelime kelime sonuçlarını okuyor (gpu gerekmiyor). kelimeleri kaynak kelime ile ingilizcesinin
yazılışı ne kadar benziyor diye ikiye ayırıyorum:
- benzer: almanca "haus" / ingilizce "house", fransızca "table" / "table" gibi
- benzemeyen: fransızca "chien" / ingilizce "dog", almanca "baum" / "tree" gibi

ingilizceyi silmenin zararı sadece benzer kelimelerde varsa, silinen şey büyük ihtimalle kaynak kelimenin bilgisi.
benzemeyen kelimelerde de aynı zarar varsa kaynak şüphesi düşüyor, çünkü orada ingilizce vektör kaynağa hiç benzemiyor.

benzerlik ölçüsü python'un difflib oranı (0 hiç benzemiyor, 1 aynı yazılış), aksanları silerek.
"""
import json, sys, unicodedata, random
from difflib import SequenceMatcher
from pathlib import Path

COL = {"en": 0, "tr": 1, "fr": 2, "de": 3}
ESIK = 0.55  # "main"/"hand" (0.5) benzer sayılmasın


def sade(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def benzerlik(a, b):
    return SequenceMatcher(None, sade(a), sade(b)).ratio()


def bootstrap(d, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(d) / n, m[int(iters * .025)], m[int(iters * .975)]


def main(path):
    res = json.load(open(path, encoding="utf-8"))
    out = {}
    for pair, P in res["pairs"].items():
        src = pair.split("-")[0]
        rows = P["rows"]
        print(f"\n{pair}: {len(rows)} kelime")
        out[pair] = {}
        for grup, sec in (("benzer", lambda r: benzerlik(r["word"][COL[src]], r["word"][0]) >= ESIK),
                          ("benzemeyen", lambda r: benzerlik(r["word"][COL[src]], r["word"][0]) < ESIK)):
            R = [r for r in rows if sec(r)]
            if not R:
                continue
            ornek = ", ".join(f"{r['word'][COL[src]]}/{r['word'][0]}" for r in R[:5])
            g = {"n": len(R), "ornek": ornek}
            print(f"  {grup} ({len(R)} kelime, örnek: {ornek})")
            for c in ("hedef", "kaynak", "en", "en_dik2", "kontrol_dik2", "baska_dik2"):
                d = [r["yok"]["p"] - r[c]["p"] for r in R]
                m, lo, hi = bootstrap(d)
                dog = sum(r[c]["dogru"] for r in R)
                g[c] = {"p_dususu": m, "aralik": [lo, hi], "dogru": dog, "yok_dogru": sum(r["yok"]["dogru"] for r in R)}
                print(f"    {c:13s} doğru {dog:3d}/{len(R)} (silmeden {g[c]['yok_dogru']}), p düşüşü {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            out[pair][grup] = g
    dst = Path(path).with_name(Path(path).stem + "_benzerlik.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nkaydedildi:", dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/stage4_source_Qwen3.5-4B.json")

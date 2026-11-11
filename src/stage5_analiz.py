"""aşama 5 analizi. gpu gerekmiyor, sadece tokenizer (colab'da ya da transformers kurulu bir yerde).

1. doğruluk: her koşul, soru başına yeniden puanlanıyor (math-verify). eşli karşılaştırmalar için kesin mcnemar
   ve doğru oranı farkına soruları yeniden örnekleyerek %95 aralık.
2. döngü: token bazlı (stage5_tr.dongu_token) oran ve başlangıç. tr_on ve stage3_think_base_v2 (ingilizce).
3. düşünce dili: düşünme açık cevaplar 100 token'lık parçalara bölünüyor, latex atılıyor, her parça
   türkçe / ingilizce / çince / belirsiz diye sınıflanıyor.
4. token şişmesi: aynı soru türkçe ve ingilizce kaç token.
5. ayrı raporlananlar: sayısı türkçe biçimli sorular (242, 263, 273) ve metin cevaplılar (227, 255, 257, 296).

çalıştırma: python src/stage5_analiz.py --data data
"""
import argparse, json, math, os, random, re
from math_verify import parse, verify

KOSULLAR = ["en_off", "tr_off", "trq_eninst_off", "geri_off", "tr_on", "zorla_tr", "zorla_en"]
CIFTLER = [("en_off", "tr_off"), ("en_off", "geri_off"), ("tr_off", "trq_eninst_off"), ("tr_off", "tr_on"),
           ("tr_on", "zorla_tr"), ("zorla_en", "zorla_tr"), ("en_off", "zorla_tr")]
OZEL = {"sayi_bicimi": [242, 263, 273], "metin_cevap": [227, 255, 257, 296]}

TR_KAR = re.compile(r"[çğışöüÇĞİŞÖÜ]")
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
TR_SOZ = set("ve bir bu için ile olarak de da ise ki şimdi yani değil olan sonra eğer çünkü göre kadar her "
             "olur olduğu olduğunu bulmak buluyoruz cevap sonuç soru adım toplam sayı sayısı tane veya ama "
             "hem şu o biz bunu bunun burada".split())
EN_SOZ = set("the and is of to we so let that this for with if then are be it in can which but now "
             "wait maybe since because first next answer number numbers total check need find".split())


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def fark_ci(a, b, iters=10000, seed=0):
    """b - a doğru sayısı farkı (puan), sorular yeniden örneklenerek."""
    rng, n = random.Random(seed), len(a)
    d = [int(y) - int(x) for x, y in zip(a, b)]
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) for _ in range(iters))
    return sum(d), s[int(iters * .025)], s[int(iters * .975)]


def dogru(answer, text):
    try:
        return bool(verify(parse("$" + answer + "$"), parse(text)))
    except Exception:
        return False


def latexsiz(t):
    t = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\$[^$]*\$", " ", t, flags=re.S)
    return re.sub(r"\\[a-zA-Z]+|[{}_^\\]", " ", t)


def parca_dili(t):
    if len(CJK.findall(t)) >= 5:
        return "çince"
    t = latexsiz(t)
    w = re.findall(r"[a-zA-ZçğışöüÇĞİŞÖÜ]+", t.lower())
    tr = sum(x in TR_SOZ for x in w) + len(TR_KAR.findall(t)) / 3
    en = sum(x in EN_SOZ for x in w)
    if tr + en < 3:
        return "belirsiz"
    return "türkçe" if tr > en else "ingilizce"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from stage5_tr import dongu_token
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    S = json.load(open(f"{a.data}/stage5_tr_200_299.json", encoding="utf-8"))
    ids = [r["i"] for r in S]
    D, C = {}, {}
    for k in KOSULLAR:
        p = f"{a.data}/stage5_{k}.json"
        if not os.path.exists(p):
            continue
        D[k] = json.load(open(p, encoding="utf-8"))
        C[k] = [r["correct"] if "correct" in r else dogru(r["answer"], r["text"]) for r in D[k]]
    out = {"dogru": {k: sum(v) for k, v in C.items()}, "ciftler": {}, "ozel": {}}

    print("## doğruluk\n")
    print("| koşul | doğru | sınıra dayanan |\n|---|---|---|")
    for k, v in C.items():
        tak = sum(not r["finished"] for r in D[k])
        print(f"| {k} | {sum(v)} | {tak} |")
    print("\n## eşli karşılaştırmalar (b − a)\n")
    print("| a | b | a doğru b yanlış | b doğru a yanlış | fark [%95] | mcnemar p |\n|---|---|---|---|---|---|")
    for x, y in CIFTLER:
        if x in C and y in C:
            n01, n10, p = mcnemar(C[x], C[y])
            f, lo, hi = fark_ci(C[x], C[y])
            out["ciftler"][f"{x}|{y}"] = dict(a_dogru_b_yanlis=n01, b_dogru_a_yanlis=n10, fark=f, ci=[lo, hi], p=p)
            print(f"| {x} | {y} | {n01} | {n10} | {f:+d} [{lo:+d}, {hi:+d}] | {p:.2g} |")

    print("\n## ayrı raporlanan sorular\n")
    for grup, liste in OZEL.items():
        sat = {i: {k: C[k][ids.index(i)] for k in C} for i in liste}
        out["ozel"][grup] = sat
        print(grup, json.dumps(sat))

    print("\n## döngü (token bazlı: 8 token, 256 pencere, %90)\n")
    out["dongu"] = {}
    kaynak = {"tr_on": f"{a.data}/stage5_tr_on.json", "en_on (think_base_v2)": f"{a.data}/stage3_think_base_v2.json"}
    for ad, p in kaynak.items():
        if not os.path.exists(p):
            continue
        rows = json.load(open(p, encoding="utf-8"))
        ons = [dongu_token(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
        dg = [o for o in ons if o is not None]
        med = sorted(dg)[len(dg) // 2] if dg else None
        out["dongu"][ad] = dict(n=len(rows), dongu=len(dg), medyan_baslangic=med,
                                sinir=sum(not r["finished"] for r in rows))
        print(ad, out["dongu"][ad])

    print("\n## düşünce dili (100 token'lık parçalar)\n")
    out["dil"] = {}
    for ad, p in kaynak.items():
        if not os.path.exists(p):
            continue
        rows = json.load(open(p, encoding="utf-8"))
        say, soru_cogunluk = {}, {}
        for r in rows:
            t = r["text"].split("</think>")[0]
            ii = tok(t, add_special_tokens=False)["input_ids"]
            o = dongu_token(ii)
            ii = ii[:o] if o else ii  # döngü tekrarı oranı şişirmesin
            yerel = {}
            for s in range(0, len(ii), 100):
                d = parca_dili(tok.decode(ii[s:s + 100]))
                say[d] = say.get(d, 0) + 1
                yerel[d] = yerel.get(d, 0) + 1
            if yerel:
                m = max(yerel, key=yerel.get)
                soru_cogunluk[m] = soru_cogunluk.get(m, 0) + 1
        top = sum(say.values())
        out["dil"][ad] = dict(parca=say, oran={k: round(v / top, 3) for k, v in say.items()},
                              soru_basina_cogunluk=soru_cogunluk)
        print(ad, out["dil"][ad])

    print("\n## token şişmesi (soru metni)\n")
    te = [len(tok(r["en"])["input_ids"]) for r in S]
    tt = [len(tok(r["tr"])["input_ids"]) for r in S]
    out["sisme"] = dict(en_ort=round(sum(te) / 100, 1), tr_ort=round(sum(tt) / 100, 1),
                        oran=round(sum(tt) / sum(te), 2))
    print(out["sisme"])
    for k in ("en_off", "tr_off", "tr_on"):
        if k in D:
            out["sisme"][f"{k}_cevap_ort_token"] = round(sum(r["n_tok"] for r in D[k]) / len(D[k]))

    json.dump(out, open(f"{a.data}/stage5_analiz.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nANALIZ_BITTI")


if __name__ == "__main__":
    main()

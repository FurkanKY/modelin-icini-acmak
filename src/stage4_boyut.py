"""aşama 4, boyut karşılaştırması: qwen3.5 0.8b, 2b ve 4b aynı ölçülerle yan yana.

gpu gerekmiyor, sadece data/ klasöründeki sonuçları okuyor ve markdown tablo yazıyor.
- çeviri: son katmanda doğru başlayan cevap sayısı (stage4_source, silme yokken)
- görünen ingilizce: ara katlardaki en yüksek ortalama olasılık, logit lens ve tuned lens (stage4_tuned)
- kullanılan ingilizce: hedef ve kaynaktan arındırılmış ingilizceyi silince doğru cevabın olasılığındaki düşüş,
  hem mutlak hem başlangıç olasılığına oranla. kontrol dili ve ilgisiz ingilizce kelime yanında.
- ekli kelimeler: tam doğru yazılan ifade sayısı (stage4_suffix)
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
MODELS = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B"]


def load(name):
    p = D / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def mid_max(rows, kind, lang, L):
    mids = range(int(L * .6), int(L * .9) + 1)
    return max(sum(r[kind][k][lang] for r in rows) / len(rows) for k in mids)


def main():
    out = ["| model | yön | doğru (silmeden) | görünen ingilizce, logit / tuned | görünen hedef, tuned | "
           "arındırılmış ingilizceyi silince düşüş | oranı | kontrol dili | ilgisiz ingilizce |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        S, T = load(f"stage4_source_{m}.json"), load(f"stage4_tuned_{m}.json")
        for pair in ("fr-tr", "de-tr"):
            if not S or pair not in S["pairs"]:
                continue
            P = S["pairs"][pair]
            R = P["rows"]
            base = sum(r["yok"]["p"] for r in R) / len(R)
            dog = sum(r["yok"]["dogru"] for r in R)
            st = P["stats"]
            seen = "?"
            tgt_seen = "?"
            if T and pair in T["pairs"]:
                tr = T["pairs"][pair]
                L = len(tr[0]["logit"])
                seen = f"{mid_max(tr, 'logit', 'en', L):.3f} / {mid_max(tr, 'tuned', 'en', L):.3f}"
                tgt_seen = f"{mid_max(tr, 'tuned', 'tr', L):.3f}"
            e = st["en_dik2"]
            out.append(f"| {m} | {pair} | {dog}/{len(R)} | {seen} | {tgt_seen} | {e['p_farki']:.3f} "
                       f"[{e['aralik'][0]:.3f}, {e['aralik'][1]:.3f}] | %{100 * e['p_farki'] / base:.0f} | "
                       f"%{100 * st['kontrol_dik2']['p_farki'] / base:.0f} | %{100 * st['baska_dik2']['p_farki'] / base:.0f} |")
    out.append("")
    out.append("| model | ekli kelimeler, tam doğru |")
    out.append("|---|---|")
    for m in MODELS:
        s = load(f"stage4_suffix_{m}.json")
        if s:
            out.append(f"| {m} | {sum(r['exact'] for r in s['rows'])}/{len(s['rows'])} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()

"""aşama 3, düşünme modu: cevaplar sınıra neden dayanıyor?

gpu gerekmiyor, data/stage3_*.json dosyalarını okuyor.
"takılan" sadece 4096 token'da bitmeyen demek. düşünme modunda model uzun düşündüğü için de bitmeyebilir.
o yüzden ayrıca birebir döngüye bakıyorum. bir kelime "tekrar" sayılıyor eğer onunla biten 8 kelimelik parça
metnin daha önceki bir yerinde geçmişse. 200 kelimelik bir pencerenin en az 180'i tekrarsa cevap orada
döngüye girmiş sayılıyor ve o pencerenin başı döngünün başladığı kelime.

çalıştırma: python src/stage3_think_analiz.py think_base think_steer03 think_rep_a15 base
"""
import json, sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"


def tekrar_etiketleri(text, n=8):
    w = text.split()
    seen, lab = set(), []
    for i in range(len(w)):
        g = tuple(w[max(0, i - n + 1): i + 1])
        lab.append(len(g) == n and g in seen)
        seen.add(g)
    return lab


def dongu_baslangici(text, pencere=200, esik=180):
    lab = tekrar_etiketleri(text)
    run = sum(lab[:pencere])
    for i in range(len(lab) - pencere):
        if run >= esik:
            return i
        run += lab[i + pencere] - lab[i]
    return None


def main(names):
    print("| deneme | doğru | sınıra dayanan | birebir döngüye giren | döngünün başladığı kelime (medyan) |")
    print("|---|---|---|---|---|")
    summary = {}
    for p in (D / "stage3_think_summary.json", D / "stage3_summary.json", D / "stage3_kontrol_summary.json"):
        if p.exists():
            summary.update(json.load(open(p, encoding="utf-8")))
    for n in names:
        R = json.load(open(D / f"stage3_{n}.json", encoding="utf-8"))
        ons = sorted(o for o in (dongu_baslangici(r["text"]) for r in R) if o is not None)
        dog = summary.get(n, {}).get("dogru", "?")
        print(f"| {n} | {dog} | {sum(not r['finished'] for r in R)}/{len(R)} | {len(ons)} | {ons[len(ons) // 2] if ons else '-'} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "think_base", "think_steer03", "think_rep_a15"])

"""aşama 4, ekli kelimeler: model "evlerimizden" derken ara katlarda hangi dilin düzeninde?

fransızca "de nos maisons" (ingilizcesi "from our houses") türkçede tek kelime, "evlerimizden".
türkçe cevap kökle ("ev") başlıyor, ingilizce karşılığı edatla ("from"). ara katlarda üç şeye bakıyorum:
- türkçe kökün ilk parçası ("ev")
- ingilizce cümlenin ilk kelimesi ("from", "in", "to")
- ingilizce isim ("house")
ara katlarda "from" öne çıkarsa model ingilizce kelime sırasıyla plan yapıyor gibi görünür.
"house" öne çıkarsa sıradan bağımsız bir içerik kavramı taşıyor. "ev" öne çıkarsa türkçe düzende.

ayrıca modelin cevabı baştan sona doğru yazıp yazamadığını da kontrol ediyorum (greedy, 12 token).
stage4_tunedlens.py'nin eğittiği çevirici varsa aynı okumayı tuned lens ile de yapıyorum.
stage4_latent.py ile aynı klasörde çalışıyor.
"""
import argparse, json, os, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import start_tokens

# (ingilizce tekil, fransızca çoğul, türkçe kök, türkçe çoğul+bizim kökü)
NOUNS = [
    ("house", "maisons", "ev", "evlerimiz"), ("tree", "arbres", "ağaç", "ağaçlarımız"),
    ("dog", "chiens", "köpek", "köpeklerimiz"), ("book", "livres", "kitap", "kitaplarımız"),
    ("bird", "oiseaux", "kuş", "kuşlarımız"), ("child", "enfants", "çocuk", "çocuklarımız"),
    ("friend", "amis", "arkadaş", "arkadaşlarımız"), ("hand", "mains", "el", "ellerimiz"),
    ("door", "portes", "kapı", "kapılarımız"), ("window", "fenêtres", "pencere", "pencerelerimiz"),
    ("key", "clés", "anahtar", "anahtarlarımız"), ("horse", "chevaux", "at", "atlarımız"),
    ("fish", "poissons", "balık", "balıklarımız"), ("apple", "pommes", "elma", "elmalarımız"),
    ("knife", "couteaux", "bıçak", "bıçaklarımız"), ("cat", "chats", "kedi", "kedilerimiz"),
    ("cow", "vaches", "inek", "ineklerimiz"), ("garden", "jardins", "bahçe", "bahçelerimiz"),
    ("city", "villes", "şehir", "şehirlerimiz"), ("ship", "navires", "gemi", "gemilerimiz"),
]


def harmony(stem, back, front):
    """büyük ünlü uyumu: kökteki son ünlü kalınsa back, inceyse front eki."""
    for ch in reversed(stem):
        if ch in "aıou":
            return back
        if ch in "eiöü":
            return front
    return front


# (fransızca edat, ingilizce edat, türkçe ek kalın, ince)
CASES = [("de nos", "from", "dan", "den"), ("dans nos", "in", "da", "de"), ("à nos", "to", "a", "e")]


def items():
    out = []
    for en, fr, root, base in NOUNS:
        for fr_p, en_p, back, front in CASES:
            tr = base + harmony(base, back, front)
            out.append({"fr": f"{fr_p} {fr}", "en": f"{en_p} our {en}s", "en_first": en_p, "en_noun": en,
                        "tr": tr, "tr_root": root})
    return out


def prompt(it, demos):
    lines = [f'Français: "{d["fr"]}" - Türkçe: "{d["tr"]}"' for d in demos]
    lines.append(f'Français: "{it["fr"]}" - Türkçe: "')
    return "\n".join(lines)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="data")
    ap.add_argument("--lens", default="", help="stage4_tunedlens.py'nin kaydettiği çevirici")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    norm, head = model.model.norm, model.lm_head
    lens = None
    if a.lens and os.path.exists(a.lens):
        from stage4_tunedlens import Lens
        sd = torch.load(a.lens, map_location=a.device)
        n_l = len(model.model.layers)
        lens = Lens(n_l, sd["t.0.weight"].shape[0]).to(a.device)
        lens.load_state_dict(sd)
    readers = {"logit": lambda k, h: h}
    if lens is not None:
        readers["tuned"] = lambda k, h: lens(k, h.float()) if k < len(model.model.layers) else h
    IT = items()
    rows, skipped = [], 0
    for i, it in enumerate(IT):
        S = {"tr_root": start_tokens(tok, vocab_str, it["tr_root"]),
             "en_first": start_tokens(tok, vocab_str, it["en_first"]),
             "en_noun": start_tokens(tok, vocab_str, it["en_noun"])}
        if any(S[x] & S[y] for x in S for y in S if x < y):
            skipped += 1; continue
        rng = random.Random(i)
        demos = rng.sample([d for d in IT if d["tr_root"] != it["tr_root"]], 4)
        enc = tok(prompt(it, demos), return_tensors="pt").to(a.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        layers = {}
        for kind, rd in readers.items():
            layers[kind] = []
            for k, h in enumerate(hs):
                hh = rd(k, h[0, -1]).to(norm.weight.dtype)
                p = torch.softmax(head(norm(hh)).float(), -1)
                layers[kind].append({x: float(p[list(v)].sum()) for x, v in S.items()})
        gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).split('"')[0]
        rows.append({"item": it, "layers": layers, "gen": text, "exact": text.strip() == it["tr"]})

    name = a.model.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.model, "skipped": skipped, "rows": rows},
              open(f"{a.out}/stage4_suffix_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n, L = len(rows), len(rows[0]["layers"]["logit"])
    print(f"{name}: {n} ifade, atlanan {skipped}, tam doğru yazılan {sum(r['exact'] for r in rows)}/{n}")
    print("örnekler:", [(r["item"]["tr"], r["gen"]) for r in rows[:6]])
    for kind in readers:
        print(f"\n{kind} lens\nkatman   tr kök  en ilk  en isim")
        for k in range(L):
            print(f"{k:5d}   " + "  ".join(f"{sum(r['layers'][kind][k][x] for r in rows) / n:6.3f}" for x in ("tr_root", "en_first", "en_noun")))


if __name__ == "__main__":
    main()

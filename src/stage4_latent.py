"""aşama 4: model türkçeye çevirirken içeride ingilizceden geçiyor mu?

wendler ve arkadaşlarının (2024) çeviri deneyinin küçük hali. modele birkaç örnekle
'Français: "fleur" - Türkçe: "çiçek"' gibi bir çeviri listesi veriyorum ve son satırı yarım bırakıyorum.
model bir sonraki token'da türkçe kelimeyi yazmalı.

her katmanda modelin o anki ara durumunu doğrudan çıkış katmanına verip (logit lens) "şu an söyleseydin
ne söylerdin" diye soruyorum. sonra üç olasılığa bakıyorum:
- türkçe cevabın ilk parçasına verilen olasılık
- aynı kavramın ingilizcesinin ilk parçasına verilen olasılık
- kaynak dildeki kelimenin ilk parçasına verilen olasılık

orta katlarda ingilizce türkçeden önde çıkarsa model ingilizceden geçiyor gibi görünüyor.
bayazit ve arkadaşları (2026) bu yöntemin ingilizceyi abarttığını söylüyor, o yüzden sonucu tek başına kanıt saymıyorum.

çalıştırma: python src/stage4_latent.py --model Qwen/Qwen3.5-0.8B --src fr --tgt tr
"""
import argparse, json, os, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LANG_NAME = {"fr": "Français", "de": "Deutsch", "en": "English", "tr": "Türkçe"}

# somut, sık kullanılan isimler. dillerin arasında ortak kökü olanları (taksi, radyo gibi) bilerek almadım.
WORDS = [
    # en, tr, fr, de
    ("book", "kitap", "livre", "Buch"), ("water", "su", "eau", "Wasser"), ("tree", "ağaç", "arbre", "Baum"),
    ("house", "ev", "maison", "Haus"), ("dog", "köpek", "chien", "Hund"), ("bird", "kuş", "oiseau", "Vogel"),
    ("moon", "ay", "lune", "Mond"), ("sun", "güneş", "soleil", "Sonne"), ("flower", "çiçek", "fleur", "Blume"),
    ("bread", "ekmek", "pain", "Brot"), ("horse", "at", "cheval", "Pferd"), ("mountain", "dağ", "montagne", "Berg"),
    ("cloud", "bulut", "nuage", "Wolke"), ("door", "kapı", "porte", "Tür"), ("window", "pencere", "fenêtre", "Fenster"),
    ("key", "anahtar", "clé", "Schlüssel"), ("heart", "kalp", "cœur", "Herz"), ("eye", "göz", "œil", "Auge"),
    ("hand", "el", "main", "Hand"), ("snow", "kar", "neige", "Schnee"), ("fish", "balık", "poisson", "Fisch"),
    ("apple", "elma", "pomme", "Apfel"), ("milk", "süt", "lait", "Milch"), ("road", "yol", "route", "Straße"),
    ("child", "çocuk", "enfant", "Kind"), ("friend", "arkadaş", "ami", "Freund"), ("star", "yıldız", "étoile", "Stern"),
    ("river", "nehir", "rivière", "Fluss"), ("egg", "yumurta", "œuf", "Ei"), ("knife", "bıçak", "couteau", "Messer"),
]
# ikinci tur: 76 kelime daha. iki dilde aynı yazılanlar (table, train, village) otomatik atlanıyor.
WORDS += [
    ("cat", "kedi", "chat", "Katze"), ("cow", "inek", "vache", "Kuh"), ("sheep", "koyun", "mouton", "Schaf"),
    ("chicken", "tavuk", "poulet", "Huhn"), ("mouse", "fare", "souris", "Maus"), ("bear", "ayı", "ours", "Bär"),
    ("wolf", "kurt", "loup", "Wolf"), ("snake", "yılan", "serpent", "Schlange"), ("bee", "arı", "abeille", "Biene"),
    ("tooth", "diş", "dent", "Zahn"), ("head", "baş", "tête", "Kopf"), ("foot", "ayak", "pied", "Fuß"),
    ("nose", "burun", "nez", "Nase"), ("ear", "kulak", "oreille", "Ohr"), ("mouth", "ağız", "bouche", "Mund"),
    ("hair", "saç", "cheveux", "Haar"), ("blood", "kan", "sang", "Blut"), ("bone", "kemik", "os", "Knochen"),
    ("skin", "deri", "peau", "Haut"), ("stone", "taş", "pierre", "Stein"), ("sea", "deniz", "mer", "Meer"),
    ("lake", "göl", "lac", "See"), ("rain", "yağmur", "pluie", "Regen"), ("wind", "rüzgar", "vent", "Wind"),
    ("fire", "ateş", "feu", "Feuer"), ("ice", "buz", "glace", "Eis"), ("sky", "gök", "ciel", "Himmel"),
    ("night", "gece", "nuit", "Nacht"), ("day", "gün", "jour", "Tag"), ("year", "yıl", "année", "Jahr"),
    ("city", "şehir", "ville", "Stadt"), ("village", "köy", "village", "Dorf"), ("bridge", "köprü", "pont", "Brücke"),
    ("wall", "duvar", "mur", "Mauer"), ("roof", "çatı", "toit", "Dach"), ("bed", "yatak", "lit", "Bett"),
    ("chair", "sandalye", "chaise", "Stuhl"), ("table", "masa", "table", "Tisch"), ("cup", "fincan", "tasse", "Tasse"),
    ("spoon", "kaşık", "cuillère", "Löffel"), ("salt", "tuz", "sel", "Salz"), ("sugar", "şeker", "sucre", "Zucker"),
    ("honey", "bal", "miel", "Honig"), ("cheese", "peynir", "fromage", "Käse"), ("meat", "et", "viande", "Fleisch"),
    ("rice", "pirinç", "riz", "Reis"), ("wine", "şarap", "vin", "Wein"), ("shoe", "ayakkabı", "chaussure", "Schuh"),
    ("shirt", "gömlek", "chemise", "Hemd"), ("hat", "şapka", "chapeau", "Hut"), ("bag", "çanta", "sac", "Tasche"),
    ("money", "para", "argent", "Geld"), ("gold", "altın", "or", "Gold"), ("iron", "demir", "fer", "Eisen"),
    ("ship", "gemi", "navire", "Schiff"), ("car", "araba", "voiture", "Auto"), ("train", "tren", "train", "Zug"),
    ("wheel", "tekerlek", "roue", "Rad"), ("letter", "mektup", "lettre", "Brief"), ("song", "şarkı", "chanson", "Lied"),
    ("word", "kelime", "mot", "Wort"), ("name", "isim", "nom", "Name"), ("king", "kral", "roi", "König"),
    ("woman", "kadın", "femme", "Frau"), ("man", "adam", "homme", "Mann"), ("mother", "anne", "mère", "Mutter"),
    ("father", "baba", "père", "Vater"), ("brother", "kardeş", "frère", "Bruder"), ("girl", "kız", "fille", "Mädchen"),
    ("garden", "bahçe", "jardin", "Garten"), ("forest", "orman", "forêt", "Wald"), ("field", "tarla", "champ", "Feld"),
    ("island", "ada", "île", "Insel"), ("leaf", "yaprak", "feuille", "Blatt"), ("seed", "tohum", "graine", "Samen"),
    ("grass", "çimen", "herbe", "Gras"),
]
COL = {"en": 0, "tr": 1, "fr": 2, "de": 3}


def start_tokens(tok, vocab_str, word):
    """kelimeyi başlatabilecek bütün token'lar. kelimenin kendi ilk token'ı ve kelimenin başıyla aynı olan
    en az 2 harflik token'lar (önünde boşluk olsa da olmasa da). tek harfli parçaları almıyorum, yoksa
    "e" hem "ekmek"in hem "egg"in başı sayılır."""
    ids = {tok(word, add_special_tokens=False)["input_ids"][0], tok(" " + word, add_special_tokens=False)["input_ids"][0]}
    for i, s in enumerate(vocab_str):
        s = s.lstrip()
        if len(s) >= min(2, len(word)) and word.startswith(s):
            ids.add(i)
    return ids


def build_prompt(src, tgt, i, n_shot=4, seed=0):
    rng = random.Random(seed + i)
    demos = rng.sample([j for j in range(len(WORDS)) if j != i], n_shot)
    lines = [f'{LANG_NAME[src]}: "{WORDS[j][COL[src]]}" - {LANG_NAME[tgt]}: "{WORDS[j][COL[tgt]]}"' for j in demos]
    lines.append(f'{LANG_NAME[src]}: "{WORDS[i][COL[src]]}" - {LANG_NAME[tgt]}: "')
    return "\n".join(lines)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--src", default="fr")
    ap.add_argument("--tgt", default="tr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dtype", default="float32", help="ram yetmezse bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    inner = model.model if hasattr(model.model, "norm") else model.model.language_model
    norm, head = inner.norm, model.lm_head

    langs = sorted({a.src, a.tgt, "en"})
    rows, skipped = [], []
    for i, w in enumerate(WORDS):
        S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in langs}
        # türkçe ve ingilizce aynı token'la başlayabiliyorsa bu kelimede ikisini ayıramam, atlıyorum
        if any(S[x] & S[y] for x in langs for y in langs if x < y):
            skipped.append(w[0]); continue
        enc = tok(build_prompt(a.src, a.tgt, i), return_tensors="pt").to(a.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        layers = []
        for h in hs:
            p = torch.softmax(head(norm(h[0, -1])).float(), -1)  # son katmanda norm iki kez uygulanıyor, etkisi küçük
            layers.append({l: float(p[list(S[l])].sum()) for l in langs})
        top = tok.decode(int(torch.argmax(head(norm(hs[-1][0, -1])))))
        rows.append({"word": w, "layers": layers, "top": top})

    os.makedirs(a.out, exist_ok=True)
    name = a.model.split("/")[-1]
    path = f"{a.out}/stage4_{name}_{a.src}-{a.tgt}.json"
    json.dump({"model": a.model, "src": a.src, "tgt": a.tgt, "skipped": skipped, "rows": rows}, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    n, L = len(rows), len(rows[0]["layers"])
    ok = sum(r["top"].strip() and r["word"][COL[a.tgt]].startswith(r["top"].strip()) for r in rows)
    print(f"{name} {a.src}->{a.tgt}: {n} kelime, atlanan {len(skipped)} {skipped}")
    print(f"son katmanda doğru başlayan cevap: {ok}/{n}")
    print("katman  " + "  ".join(f"{l:>6s}" for l in langs))
    for k in range(L):
        print(f"{k:5d}   " + "  ".join(f"{sum(r['layers'][k][l] for r in rows) / n:6.3f}" for l in langs))
    print("kaydedildi:", path)


if __name__ == "__main__":
    main()

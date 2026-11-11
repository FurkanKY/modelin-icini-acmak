"""aşama 4, çince kontrolü: etki ingilizceye mi özgü, yoksa modelin en çok gördüğü dile mi?

ingilizceyi silmek türkçe çeviriyi bozuyordu, almancayı silmek bozmuyordu. ama qwen'in eğitim verisinde ingilizce
almancadan çok daha fazla. belki model kavramı "en çok gördüğü dilin" kelimesine yakın tutuyor ve almanca sadece
yeterince sık olmadığı için işe yaramıyor. qwen çok miktarda çince metinle de eğitilmiş bir model. o yüzden aynı
testi çince kelimeyle yapıyorum.

- çinceyi silmek de bozuyorsa: model kavramı "çok gördüğü dillerin" yakınında tutuyor, ingilizceye özel bir şey yok
- çinceyi silmek bozmuyor, ingilizceyi silmek bozuyorsa: etki ingilizceye özgü

stage4_source.py ile aynı kelimeler, aynı katmanlar ve aynı arındırma. koşullar:
- yok, hedef (olumlu kontrol), en_dik2 (karşılaştırma için yeniden)
- zh: kavramın çincesi, olduğu gibi
- zh_dik2: çincenin hedefe ve kaynağa ait payı çıkarılmış hali. asıl test bu.
- kontrol_dik2: almanca ya da fransızca, stage4_source ile aynı

çalıştırma: python src/stage4_zh.py --model Qwen/Qwen3.5-4B --dtype float16
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, start_tokens, build_prompt
from stage4_bridge import first_ids, word_hooks
from stage4_source import mcnemar, bootstrap

ZH = {
    "book": "书", "water": "水", "tree": "树", "house": "房子", "dog": "狗", "bird": "鸟", "moon": "月亮", "sun": "太阳",
    "flower": "花", "bread": "面包", "horse": "马", "mountain": "山", "cloud": "云", "door": "门", "window": "窗户",
    "key": "钥匙", "heart": "心", "eye": "眼睛", "hand": "手", "snow": "雪", "fish": "鱼", "apple": "苹果", "milk": "牛奶",
    "road": "路", "child": "孩子", "friend": "朋友", "star": "星星", "river": "河", "egg": "鸡蛋", "knife": "刀",
    "cat": "猫", "cow": "牛", "sheep": "羊", "chicken": "鸡", "mouse": "老鼠", "bear": "熊", "wolf": "狼", "snake": "蛇",
    "bee": "蜜蜂", "tooth": "牙齿", "head": "头", "foot": "脚", "nose": "鼻子", "ear": "耳朵", "mouth": "嘴",
    "hair": "头发", "blood": "血", "bone": "骨头", "skin": "皮肤", "stone": "石头", "sea": "海", "lake": "湖", "rain": "雨",
    "wind": "风", "fire": "火", "ice": "冰", "sky": "天空", "night": "夜晚", "day": "白天", "year": "年", "city": "城市",
    "village": "村庄", "bridge": "桥", "wall": "墙", "roof": "屋顶", "bed": "床", "chair": "椅子", "table": "桌子",
    "cup": "杯子", "spoon": "勺子", "salt": "盐", "sugar": "糖", "honey": "蜂蜜", "cheese": "奶酪", "meat": "肉",
    "rice": "米饭", "wine": "葡萄酒", "shoe": "鞋", "shirt": "衬衫", "hat": "帽子", "bag": "包", "money": "钱", "gold": "金",
    "iron": "铁", "ship": "船", "car": "汽车", "train": "火车", "wheel": "轮子", "letter": "信", "song": "歌", "word": "词",
    "name": "名字", "king": "国王", "woman": "女人", "man": "男人", "mother": "母亲", "father": "父亲", "brother": "兄弟",
    "girl": "女孩", "garden": "花园", "forest": "森林", "field": "田野", "island": "岛", "leaf": "叶子", "seed": "种子",
    "grass": "草",
}
CONDS = ["yok", "hedef", "en_dik2", "zh", "zh_dik2", "kontrol_dik2"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--pairs", default="fr-tr,de-tr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    assert all(w[0] in ZH for w in WORDS), [w[0] for w in WORDS if w[0] not in ZH]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    n_layers = len(model.model.layers)
    layers = list(range(int(n_layers * .5), int(n_layers * .9)))
    norm, head = model.model.norm, model.lm_head
    Wg = head.weight.float() * norm.weight.float()
    vec = lambda word: Wg[first_ids(tok, word)].T
    name = a.model.split("/")[-1]
    res = {"model": a.model, "layers": layers, "pairs": {}}

    for pair in a.pairs.split(","):
        src, tgt = pair.split("-")
        ctrl = "de" if src != "de" else "fr"
        rows, atlanan = [], []
        for i, w in enumerate(WORDS):
            S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in {src, tgt, "en", ctrl}}
            if any(S[x] & S[y] for x in S for y in S if x < y):
                continue  # stage4_source ile aynı kelimeler
            zh = ZH[w[0]]
            if set(first_ids(tok, zh)) & S[tgt]:
                atlanan.append(w[0]); continue
            enc = tok(build_prompt(src, tgt, i), return_tensors="pt").to(a.device)
            T, Sv = vec(w[COL[tgt]]), vec(w[COL[src]])
            base = torch.cat([T, Sv], 1)
            nb = torch.linalg.matrix_rank(base).item()
            Qb, _ = torch.linalg.qr(base)

            def clean(X):
                Q, _ = torch.linalg.qr(torch.cat([base, X], 1))
                return Q[:, nb:], float((X - Qb @ (Qb.T @ X)).norm() / X.norm())

            Vs, kalan = {}, {}
            Vs["hedef"] = torch.linalg.qr(T)[0]
            Vs["en_dik2"], kalan["en"] = clean(vec(w[COL["en"]]))
            Vs["zh"] = torch.linalg.qr(vec(zh))[0]
            Vs["zh_dik2"], kalan["zh"] = clean(vec(zh))
            Vs["kontrol_dik2"], kalan["kontrol"] = clean(vec(w[COL[ctrl]]))
            r = {"word": list(w) + [zh], "kalan": kalan}
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
        base_p = sum(r["yok"]["p"] for r in rows) / n
        stats = {}
        print(f"\n{name} {pair} ({n} kelime, atlanan {atlanan}, kontrol dili {ctrl}, p(hedef) {base_p:.3f})")
        for c in CONDS[1:]:
            m, lo, hi = bootstrap([r["yok"]["p"] for r in rows], [r[c]["p"] for r in rows])
            n01, n10, pm = mcnemar([r["yok"]["dogru"] for r in rows], [r[c]["dogru"] for r in rows])
            stats[c] = {"p_farki": m, "aralik": [lo, hi], "oran": m / base_p, "mcnemar": [n01, n10, pm]}
            print(f"  {c:13s} doğru {sum(r[c]['dogru'] for r in rows):3d}/{n}  p düşüşü {m:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"(%{100 * m / base_p:.0f})  mcnemar {n01}/{n10} p={pm:.4f}")
        m, lo, hi = bootstrap([r["zh_dik2"]["p"] for r in rows], [r["en_dik2"]["p"] for r in rows])
        stats["en_dik2_vs_zh_dik2"] = {"p_farki": m, "aralik": [lo, hi]}
        print(f"  en_dik2, zh_dik2'ye göre fazladan düşüş: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for k in ("en", "zh", "kontrol"):
            print(f"  {k} vektörünün hedef+kaynak dışında kalan payı: {sum(r['kalan'][k] for r in rows) / n:.3f}")
        res["pairs"][pair] = {"n": n, "ctrl": ctrl, "atlanan": atlanan, "stats": stats, "rows": rows}

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/stage4_zh_{name}.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

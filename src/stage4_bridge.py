"""aşama 4, açık soru: kumru türkçeden fransızcaya çevirirken ingilizceyi köprü olarak mı kullanıyor?

stage4_causal.py'de kumru tr→fr'de ingilizce yönünü silince doğru cevap 11'den 5'e düştü. ama aynı silme ara
katlardaki ingilizceyi azaltmadı, tersine artırdı (0.10'dan 0.32'ye). fransızca yönünü silmek de aynısını yaptı.
bu, qwen'de gördüğüm tuzağa benziyor. "dil eksi bütün dillerin ortalaması" diye bulunan yönler birbirine karışıyor.
burada üç yoldan bakıyorum.

1. dik yönler. her dilin yönünden öteki üç dilin yönlerini çıkarıyorum (gram-schmidt), sonra siliyorum.
   ingilizcenin kendine özgü kısmını silmek yine zarar veriyorsa köprü ihtimali güçlenir.
2. kelimeye özel silme. her kelime için ara katlardan o kelimenin ingilizcesinin çıkış vektörünü siliyorum.
   kontrol olarak aynısını almancası için yapıyorum. model ingilizce kelimeden geçiyorsa ingilizceyi silmek
   almancayı silmekten çok daha fazla zarar verir. yöntemin işe yaradığını görmek için hedef kelimenin
   kendi vektörünü de siliyorum (olumlu kontrol). bu zarar vermiyorsa yöntem hiçbir şey ölçmüyor demektir.
   ingilizce vektör hedef kelimeye biraz benzeyebilir (fransızca ile ortak kökler). o yüzden "_dik" sürümünde
   ingilizce ve almanca vektörden önce hedef kelimenin vektörlerini çıkarıyorum, kalan kısmı siliyorum.
3. tuned lens ile ara katlarda ingilizce ve almanca yan yana. almanca görevle ilgisiz, taban çizgisi o.

stage4_latent.py ile aynı klasörde çalışıyor.
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, start_tokens, build_prompt
from stage4_causal import language_directions, ablate, evaluate, LANGS


def perp(U, keep, others):
    """keep yönünden others yönlerinin kapsadığı alanı çıkarıyorum, her katta ayrı."""
    out = []
    for k in range(U[keep].shape[0]):
        Q, _ = torch.linalg.qr(torch.stack([U[o][k] for o in others], 1))
        v = U[keep][k] - Q @ (Q.T @ U[keep][k])
        out.append(v / v.norm())
    return torch.stack(out)


def first_ids(tok, word):
    """kelimenin modelde ilk token'ı, boşluklu ve boşluksuz yazılışlarıyla."""
    return sorted({tok(word, add_special_tokens=False)["input_ids"][0], tok(" " + word, add_special_tokens=False)["input_ids"][0]})


def word_hooks(model, layers, V):
    """V (birim, dik sütunlar) alanını verilen katmanların çıkışından siliyorum."""
    hooks = []
    for k in layers:
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            Vv = V.to(h.dtype)
            h = h - (h @ Vv) @ Vv.T
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hooks.append(model.model.layers[k].register_forward_hook(hook))
    return hooks


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vngrs-ai/Kumru-2B-Base")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--pairs", default="tr-fr,fr-tr")
    ap.add_argument("--lens", default="")
    ap.add_argument("--out", default="data")
    ap.add_argument("--parts", default="dik,kelime,okuma")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    n_layers = len(model.model.layers)
    layers = list(range(int(n_layers * .5), int(n_layers * .9)))
    norm, head = model.model.norm, model.lm_head
    Wg = head.weight.float() * norm.weight.float()  # çıkışın gerçekte okuduğu yön: w_u * norm ağırlığı
    name = a.model.split("/")[-1]
    res = {"model": a.model, "layers": layers}

    parts = a.parts.split(",")
    # 1. dik yönler
    half = len(WORDS) // 2
    dir_idx, test_idx = list(range(half)), list(range(half, len(WORDS)))
    U = language_directions(model, tok, dir_idx, a.device)
    mid = layers[len(layers) // 2] + 1
    cos = {f"{x}-{y}": float(U[x][mid] @ U[y][mid]) for i, x in enumerate(LANGS) for y in LANGS[i + 1:]}
    print(f"{name}: silinen katmanlar {layers[0]}-{layers[-1]}, katman {mid - 1} çıkışında kosinüsler", {k: round(v, 2) for k, v in cos.items()})
    P = {f"{l}_dik": perp(U, l, [o for o in LANGS if o != l]) for l in LANGS}
    U.update(P)
    res["cos"], res["dik"] = cos, []
    print(f"{'yön':6s} {'silinen':9s} {'n':>3s} {'p(hedef)':>9s} {'doğru':>6s} {'ara ing.':>9s}")
    for pair in (a.pairs.split(",") if "dik" in parts else []):
        src, tgt = pair.split("-")
        for what in ["yok", "en", "en_dik", "tr_dik", "fr_dik", "de_dik", "rastgele"]:
            hooks = ablate(model, layers, U[what]) if what != "yok" else []
            try:
                r = evaluate(model, tok, vocab_str, test_idx, src, tgt, a.device)
            finally:
                for h in hooks:
                    h.remove()
            r.update(pair=pair, silinen=what)
            res["dik"].append(r)
            print(f"{pair:6s} {what:9s} {r['n']:3d} {r['p_hedef']:9.3f} {r['dogru']:6d} {r['ara_ingilizce']:9.3f}")

    # 2. kelimeye özel silme: ingilizce kelimenin çıkış vektörü mü, almancası mı?
    res["kelime"] = {}
    for pair in (a.pairs.split(",") if "kelime" in parts else []):
        src, tgt = pair.split("-")
        stats = {w: {"p": 0.0, "dogru": 0} for w in ("yok", "en", "de", "hedef", "en_dik", "de_dik")}
        n, cos_t = 0, {"en": 0.0, "de": 0.0, "hedef": 1.0, "en_dik": 0.0, "de_dik": 0.0}
        for i, w in enumerate(WORDS):
            S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in {src, tgt, "en", "de"}}
            if any(S[x] & S[y] for x in S for y in S if x < y):
                continue
            enc = tok(build_prompt(src, tgt, i), return_tensors="pt").to(a.device)
            tgt_v = Wg[first_ids(tok, w[COL[tgt]])].mean(0)
            T = Wg[first_ids(tok, w[COL[tgt]])].T
            for what in ("yok", "en", "de", "hedef", "en_dik", "de_dik"):
                hooks = []
                if what.endswith("_dik"):
                    Q, _ = torch.linalg.qr(torch.cat([T, Wg[first_ids(tok, w[COL[what[:2]]])].T], 1))
                    V = Q[:, T.shape[1]:]  # hedefin kapsadığı alan çıkmış hali
                elif what != "yok":
                    V, _ = torch.linalg.qr(Wg[first_ids(tok, w[COL[tgt if what == "hedef" else what]])].T)
                if what != "yok":
                    cos_t[what] += float(torch.nn.functional.cosine_similarity(V.mean(1), tgt_v, 0))
                    hooks = word_hooks(model, layers, V)
                try:
                    p = torch.softmax(model(**enc).logits[0, -1].float(), -1)
                finally:
                    for h in hooks:
                        h.remove()
                stats[what]["p"] += float(p[list(S[tgt])].sum())
                stats[what]["dogru"] += int(int(p.argmax()) in S[tgt])
            n += 1
        for what in stats:
            stats[what]["p"] /= n
        res["kelime"][pair] = {"n": n, "sonuc": stats, "hedefle_kosinus": {k: v / n for k, v in cos_t.items()}}
        print(f"\n{pair} kelimeye özel silme, {n} kelime")
        for what, s in stats.items():
            print(f"  {what:4s} silindi: p(hedef) {s['p']:.3f}, doğru {s['dogru']}/{n}")
        print("  silinen vektörün hedef kelimeyle kosinüsü:", {k: round(v / n, 3) for k, v in cos_t.items()})

    # 3. tuned lens ile ingilizce ve almanca yan yana
    if "okuma" in parts and a.lens and os.path.exists(a.lens):
        from stage4_tunedlens import Lens
        sd = torch.load(a.lens, map_location=a.device)
        lens = Lens(n_layers, sd["t.0.weight"].shape[0]).to(a.device)
        lens.load_state_dict(sd)
        res["okuma"] = {}
        for pair in a.pairs.split(","):
            src, tgt = pair.split("-")
            langs = sorted({src, tgt, "en", "de"})
            rows = []
            for i, w in enumerate(WORDS):
                S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in langs}
                if any(S[x] & S[y] for x in S for y in S if x < y):
                    continue
                enc = tok(build_prompt(src, tgt, i), return_tensors="pt").to(a.device)
                hs = model(**enc, output_hidden_states=True).hidden_states
                r = {"word": w, "logit": [], "tuned": []}
                for k in range(n_layers + 1):
                    h = hs[k][0, -1].float()
                    for kind in ("logit", "tuned"):
                        hh = lens(k, h) if (kind == "tuned" and k < n_layers) else h
                        pp = torch.softmax(norm(hh.to(norm.weight.dtype)).float() @ head.weight.float().T, -1)
                        r[kind].append({l: float(pp[list(S[l])].sum()) for l in langs})
                rows.append(r)
            res["okuma"][pair] = rows
            L = n_layers + 1
            mids = range(int(L * .6), int(L * .9) + 1)
            print(f"\n{pair} ara katlar, {len(rows)} kelime (en yüksek ort. olasılık, ingilizce > almanca sayısı)")
            for kind in ("logit", "tuned"):
                avg = {l: max(sum(r[kind][k][l] for r in rows) / len(rows) for k in mids) for l in langs}
                win = sum(max(r[kind][k]["en"] for k in mids) > max(r[kind][k]["de"] for k in mids) for r in rows)
                print(f"  {kind:6s} " + "  ".join(f"{l} {avg[l]:.3f}" for l in langs) + f"   en>de {win}/{len(rows)}")

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/stage4_bridge_{name}{'' if a.parts == 'dik,kelime,okuma' else '_' + a.parts.replace(',', '-')}.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

"""aşama 4, tuned lens: qwen'in ara katlarındaki ingilizce ne kadar gerçek?

logit lens ara katmanı doğrudan çıkış katmanına veriyor. ama ara katmanlar çıkış katmanının "dilini" konuşmak
zorunda değil, o yüzden logit lens bazı şeyleri abartabilir. tuned lens (belrose ve ark. 2023) her katman için
küçük bir çevirici (d x d matris + sapma) öğreniyor. çevirici, o katmanın durumunu son katmanın tahminine en çok
benzeyecek biçimde dönüştürüyor. bayazit ve ark. (2026) tuned lens'in logit lens'ten daha az ingilizce gördüğünü buldu.

1. çevirici: dört dilde (en, tr, fr, de) eşit miktarda vikipedi metni üzerinde, son katmanın dağılımına kl
   uzaklığını küçültecek şekilde eğitiliyor. modelin kendi ağırlıkları değişmiyor.
2. aynı çeviri denemelerini (stage4_latent.py) tuned lens ile tekrar okuyorum.

stage4_latent.py ile aynı klasörde çalışıyor.
"""
import argparse, json, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage4_latent import WORDS, COL, start_tokens, build_prompt

WIKI = {"en": "20231101.en", "tr": "20231101.tr", "fr": "20231101.fr", "de": "20231101.de"}


def wiki_texts(n_per_lang, min_chars=400, seed=0):
    """her dilden n paragraf, akış halinde (bütün vikipediyi indirmeden)."""
    from datasets import load_dataset
    out = {}
    for l, cfg in WIKI.items():
        ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
        texts = []
        for row in ds:
            t = row["text"]
            if len(t) >= min_chars:
                texts.append(t[:3000])
            if len(texts) >= n_per_lang:
                break
        out[l] = texts
    return out


class Lens(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.t = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        for m in self.t:
            nn.init.eye_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, k, h):
        return self.t[k](h)  # birim matristen başlıyor, yani başta logit lens ile aynı


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--pos", type=int, default=96, help="her adımda kaç konumda kayıp hesaplanacak")
    ap.add_argument("--n_text", type=int, default=400)
    ap.add_argument("--pairs", default="fr-tr,de-tr,tr-fr")
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(model.model.layers)
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    norm = model.model.norm
    Wu = model.lm_head.weight.float()
    head = lambda h: norm(h.to(norm.weight.dtype)).float() @ Wu.T

    # 1. veri: dilleri sırayla değiştirerek
    t0 = time.time()
    texts = wiki_texts(a.n_text)
    seqs = []
    for i in range(a.n_text):
        for l in WIKI:
            ids = tok(texts[l][i], add_special_tokens=False)["input_ids"][: a.seq]
            if len(ids) >= 64:
                seqs.append(ids)
    print(f"veri: {len(seqs)} parça, {time.time()-t0:.0f} sn")

    # 2. eğitim. son katman (n_layers) için çevirici gerekmiyor, o zaten modelin kendi çıkışı.
    lens = Lens(n_layers, d).to(a.device)
    opt = torch.optim.AdamW(lens.parameters(), lr=1e-3, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0 = time.time()
    for step in range(a.steps):
        ids = torch.tensor([seqs[step % len(seqs)]], device=a.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states
            T = ids.shape[1]
            pos = torch.randperm(T, device=a.device)[: a.pos]
            target = F.log_softmax(out.logits[0, pos].float(), -1)
        opt.zero_grad()
        total = 0.0
        for k in range(n_layers):  # hs[k] = k. katmanın girişi, hs[0] gömme
            h = hs[k][0, pos].float()
            logp = F.log_softmax(head(lens(k, h)), -1)
            loss = F.kl_div(logp, target, log_target=True, reduction="batchmean")
            loss.backward()
            total += float(loss)
        opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"adım {step:4d}  ortalama kl {total / n_layers:.3f}  {time.time()-t0:.0f} sn", flush=True)

    name = a.model.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    torch.save(lens.state_dict(), f"/content/tunedlens_{name}.pt")  # büyük, depoya girmiyor

    # 3. çeviri denemelerini hem logit lens hem tuned lens ile oku
    vocab_str = [tok.decode([i]) for i in range(len(tok))]
    res = {"model": a.model, "steps": a.steps, "pairs": {}}
    print(f"\n{'yön':6s} {'mercek':7s}  ara katlarda en yüksek ort. olasılık (en / kaynak / hedef)   en>hedef")
    for pair in a.pairs.split(","):
        src, tgt = pair.split("-")
        langs = sorted({src, tgt, "en"})
        rows = []
        with torch.no_grad():
            for i, w in enumerate(WORDS):
                S = {l: start_tokens(tok, vocab_str, w[COL[l]]) for l in langs}
                if any(S[x] & S[y] for x in langs for y in langs if x < y):
                    continue
                enc = tok(build_prompt(src, tgt, i), return_tensors="pt").to(a.device)
                hs = model(**enc, output_hidden_states=True).hidden_states
                r = {"word": w, "logit": [], "tuned": []}
                for k in range(n_layers + 1):
                    h = hs[k][0, -1].float()
                    for kind in ("logit", "tuned"):
                        hh = lens(k, h) if (kind == "tuned" and k < n_layers) else h
                        p = torch.softmax(head(hh), -1)
                        r[kind].append({l: float(p[list(S[l])].sum()) for l in langs})
                rows.append(r)
        res["pairs"][pair] = rows
        L = n_layers + 1; mid = range(int(L * .6), int(L * .9) + 1)
        for kind in ("logit", "tuned"):
            avg = {l: max(sum(r[kind][k][l] for r in rows) / len(rows) for k in mid) for l in ("en", src, tgt)}
            win = sum(max(r[kind][k]["en"] for k in mid) > max(r[kind][k][tgt] for k in mid) for r in rows)
            print(f"{pair:6s} {kind:7s}  {avg['en']:.3f} / {avg[src]:.3f} / {avg[tgt]:.3f}{'':30s} {win}/{len(rows)}")
    json.dump(res, open(f"{a.out}/stage4_tuned_{name}.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

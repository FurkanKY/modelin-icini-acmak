"""aşama 3: 2. aşamada bulduğum yönü modelin ağırlıklarından silip takılma azalıyor mu, ona bakıyorum.

2. aşamadaki yön şimdilik sadece bir tahmin sinyali. takılacak cevaplarda modelin içi o yöne
kayıyor ama bundan o yönün takılmaya sebep olduğu sonucu çıkmaz. ateş hastalığı haber verir ama
ateşi düşürmek hastalığı geçirmez. anlamanın yolu yönü silip ne olduğuna bakmak. takılma azalırsa yön
gerçekten işin içinde, hiçbir şey değişmezse sadece bir belirtiymiş.

her katman 1024 sayılık ortak bir vektöre (residual stream) bir şey ekliyor.
bu vektöre yazan iki matris var: dikkat çıkışı (linear_attn.out_proj ya da self_attn.o_proj) ve
mlp çıkışı (mlp.down_proj). bu matrislerin çıktısından u yönündeki payı çıkarıyorum:
    W' = W - alpha * u (u^T W)
alpha = 1 o yönü tamamen siliyor. mit 1.5 kullandı, yani bir adım daha ileri gidip ters yöne itti.

yönü 0-199. sorulardan buluyorum, denemeyi 200-299. sorularda yapıyorum. aynı sorularda
deneseydim sınav sorularını önceden görmüş gibi olurdu.

2. aşamanın colab oturumunda çalışıyor, oradaki model, tok, build_prompt ve IM_END'i kullanıyor.
"""
import json, os, time
import numpy as np
import torch
from datasets import load_dataset
from math_verify import parse, verify


# ---------- 1. yönü bulmak ----------

def stuck_labels(results):
    """takıldı demek 4096 token içinde cevabı bitiremedi demek."""
    return np.array([not r["finished"] for r in results])


def loo_acc(X, y):
    """her cevabı sırayla dışarıda bırakıp yönü kalanlardan buluyorum, dışarıdakini tahmin ettiriyorum."""
    ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        mu1, mu0 = X[m & y].mean(0), X[m & ~y].mean(0)
        d = mu1 - mu0
        ok += (X[i] @ d > ((mu1 + mu0) / 2) @ d) == y[i]
    return ok / len(y)


def direction(feats, y, layer):
    """takılanların ortalamasından bitenlerin ortalamasını çıkarıyorum. sadece yön lazım, o yüzden boyunu 1 yapıyorum."""
    d = feats[y, layer].mean(0) - feats[~y, layer].mean(0)
    return d / np.linalg.norm(d)


# ---------- 2. ağırlıkları düzenlemek ----------

def writers(model, layers, parts=("attn", "mlp")):
    """verilen katmanlarda residual stream'e yazan matrisler."""
    out = []
    for i in layers:
        L = model.model.layers[i]
        if "attn" in parts:
            out.append(L.linear_attn.out_proj if hasattr(L, "linear_attn") else L.self_attn.o_proj)
        if "mlp" in parts:
            out.append(L.mlp.down_proj)
    return out


@torch.no_grad()
def edit(model, u, layers, alpha=1.0, parts=("attn", "mlp")):
    """W' = W - alpha * u (u^T W). eski ağırlıkları saklıyorum, undo() ile geri alıp başka bir ayarı deneyebiliyorum."""
    u = torch.tensor(u, dtype=torch.float32, device="cuda")
    saved = []
    for lin in writers(model, layers, parts):
        W = lin.weight
        saved.append((lin, W.detach().clone()))
        Wf = W.float()
        W.copy_((Wf - alpha * torch.outer(u, u @ Wf)).to(W.dtype))
    return saved


@torch.no_grad()
def undo(saved):
    for lin, W in saved:
        lin.weight.copy_(W)


# ---------- 3. denemek ve puanlamak ----------

def run(model, tok, build_prompt, im_end, rows, path, batch_size=25, max_new=4096, log=print):
    """soruları sorup her partiden sonra diske yazıyorum. colab koparsa kaldığı yerden devam ediyor."""
    res = json.load(open(path)) if os.path.exists(path) else []
    for start in range(len(res), len(rows), batch_size):
        batch = rows.select(range(start, min(start + batch_size, len(rows))))
        enc = tok([build_prompt(q) for q in batch["problem"]], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, row in enumerate(batch):
            ids = out[i, enc["input_ids"].shape[1]:].tolist()
            fin = im_end in ids or tok.eos_token_id in ids
            ids = [t for t in ids if t != tok.pad_token_id]
            res.append({"problem": row["problem"], "answer": row["answer"], "level": row["level"],
                        "text": tok.decode(ids, skip_special_tokens=True), "finished": fin, "n_tok": len(ids)})
        json.dump(res, open(path, "w"))
        log(f"{os.path.basename(path)}: {len(res)}/{len(rows)}")
        del out, enc; torch.cuda.empty_cache()
    return res


def grade(res):
    """math-verify ile puanlıyorum, mit de bunu kullandı. "1/2" ile "0.5"i aynı sayıyor."""
    for r in res:
        try:
            r["correct"] = bool(verify(parse("$" + r["answer"] + "$"), parse(r["text"])))
        except Exception:
            r["correct"] = False
    n = len(res)
    stuck = sum(not r["finished"] for r in res)
    return {"n": n, "dogru": sum(r["correct"] for r in res), "takilan": stuck,
            "takilip_dogru": sum(r["correct"] and not r["finished"] for r in res)}

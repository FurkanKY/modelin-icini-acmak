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
def edit(model, u, layers, alpha=1.0, parts=("attn", "mlp"), top_cols=None):
    """W' = W - alpha * u (u^T W). eski ağırlıkları saklıyorum, undo() ile geri alıp başka bir ayarı deneyebiliyorum.

    top_cols verilirse bütün matrise değil, u yönüne en çok yazan o kadar sütuna dokunuyorum.
    mit'nin p50c düzenlemesi buna benziyor. 15. ve 19. katmanın mlp'sinde 64 sütun.
    """
    u = torch.tensor(u, dtype=torch.float32, device=model.device)
    saved = []
    for lin in writers(model, layers, parts):
        W = lin.weight
        saved.append((lin, W.detach().clone()))
        Wf = W.float()
        proj = u @ Wf  # her sütunun u yönüne yazdığı miktar
        if top_cols:
            mask = torch.zeros_like(proj)
            mask[proj.abs().topk(top_cols).indices] = 1
            proj = proj * mask
        W.copy_((Wf - alpha * torch.outer(u, proj)).to(W.dtype))
    return saved


@torch.no_grad()
def undo(saved):
    for lin, W in saved:
        lin.weight.copy_(W)


def steer(model, u, layer=15, c=0.3):
    """ağırlıklara dokunmadan, verilen katmanın çıkışına her token'da -c*u ekliyorum. yani modeli "biten" tarafa itiyorum.
    c = 0.3 aşağı yukarı takılanlarla bitenlerin ortalaması arasındaki mesafe (0.297).
    dönen hook'u h.remove() ile kaldırınca model eski haline dönüyor."""
    U = torch.tensor(u, dtype=torch.float16, device=model.device)

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] - c * U,) + tuple(out[1:])
        return out - c * U
    return model.model.layers[layer].register_forward_hook(hook)


# ---------- 2b. tekrar yönü ----------

@torch.no_grad()
def repeat_direction(model, tok, build_prompt, results, n=8, start=512, hidden=1024):
    """model kendini tekrar ederken içi nasıl, yeni bir şey yazarken nasıl, ona bakıyorum.

    bir token "tekrar" sayılıyor eğer onunla biten 8 token'lık parça cevabın daha önceki bir yerinde geçmişse.
    konumdan gelen farkı karıştırmamak için iki grubu da 512. token'dan sonraki konumlardan alıyorum.
    her katman için yön, tekrar token'larının ortalamasından yeni token'ların ortalamasını çıkarınca çıkıyor.
    stage2_200_raw.json ile 126.111 tekrar ve 185.213 yeni token buldum.
    """
    nl = model.config.num_hidden_layers + 1  # gömme katmanı da sayılıyor
    sum_rep = torch.zeros(nl, hidden, device=model.device, dtype=torch.float64)
    sum_new = torch.zeros_like(sum_rep)
    n_rep = n_new = 0
    for r in results:
        ids = r["ids"]
        if len(ids) <= start + n:
            continue
        seen, lab = set(), []
        for t in range(len(ids)):
            g = tuple(ids[max(0, t - n + 1): t + 1])
            lab.append(len(g) == n and g in seen)
            seen.add(g)
        lab = torch.tensor(lab[start:], device=model.device)
        p = tok(build_prompt(r["problem"]))["input_ids"]
        hs = model(torch.tensor([p + ids], device=model.device), output_hidden_states=True).hidden_states
        H = torch.stack([h[0, len(p) + start:].double() for h in hs])  # [katman, token, 1024]
        sum_rep += H[:, lab].sum(1)
        sum_new += H[:, ~lab].sum(1)
        n_rep += int(lab.sum())
        n_new += int((~lab).sum())
        del hs, H
    D = (sum_rep / n_rep - sum_new / n_new).float().cpu().numpy()
    return D / np.linalg.norm(D, axis=1, keepdims=True), n_rep, n_new


# tekrar yönüyle yaptığım üç deneme. hepsinde yön 15. katmandan, u_rep = U_rep[15].
REPEAT_RUNS = [
    ("rep_a10", dict(layers=list(range(8, 23)), alpha=1.0)),
    ("rep_a15", dict(layers=list(range(8, 23)), alpha=1.5)),                               # mit p10c'nin karşılığı
    ("rep_p50c", dict(layers=[15, 19], alpha=1.5, parts=("mlp",), top_cols=64)),           # mit p50c
]


# ---------- 3. denemek ve puanlamak ----------

def run(model, tok, build_prompt, im_end, rows, path, batch_size=25, max_new=4096, log=print):
    """soruları sorup her partiden sonra diske yazıyorum. colab koparsa kaldığı yerden devam ediyor."""
    res = json.load(open(path)) if os.path.exists(path) else []
    for start in range(len(res), len(rows), batch_size):
        batch = rows.select(range(start, min(start + batch_size, len(rows))))
        enc = tok([build_prompt(q) for q in batch["problem"]], return_tensors="pt", padding=True).to(model.device)
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
    """math-verify ile puanlıyorum, mit de bunu kullandı. "1/2" ile "0.5"i aynı sayıyor.
    ana thread'de çağırmak gerekiyor. math-verify signal kullanıyor ve arka plandaki bir thread'de hepsini yanlış sayıyor."""
    for r in res:
        try:
            r["correct"] = bool(verify(parse("$" + r["answer"] + "$"), parse(r["text"])))
        except Exception:
            r["correct"] = False
    n = len(res)
    stuck = sum(not r["finished"] for r in res)
    return {"n": n, "dogru": sum(r["correct"] for r in res), "takilan": stuck,
            "takilip_dogru": sum(r["correct"] and not r["finished"] for r in res)}

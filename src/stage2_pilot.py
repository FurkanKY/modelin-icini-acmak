"""aşama 2 pilotu: takılan ve takılmayan cevaplarda modelin içi farklı mı, ona bakıyorum.

stage1_generate.py ile aynı colab oturumunda çalışıyor, oradaki model, tok, build_prompt ve
results'ı kullanıyor.

her cevabın ilk 512 token'ını modele geri veriyorum ve her katmanda 1024 sayılık ortalama
aktivasyonu alıyorum. bir katmanda takılanların ortalamasından bitenlerin ortalamasını çıkarınca
bir yön çıkıyor (difference of means). bu yön takılmayı tahmin edebiliyor mu diye her cevabı
sırayla dışarıda bırakıp kalanlarla yönü bulup dışarıdakini tahmin ettiriyorum (leave-one-out).
"""
import numpy as np
import torch

WIN = 512


def collect_feats(results, model, tok, build_prompt):
    feats = []
    for r in results:
        p = tok(build_prompt(r["problem"]))["input_ids"]
        ids = torch.tensor([p + r["ids"][:WIN]], device="cuda")
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        feats.append(torch.stack([h[0, len(p):].float().mean(0) for h in hs]).cpu().numpy())
    return np.stack(feats)  # boyut: [cevap, 25 (gömme + 24 katman), 1024]


def loo_acc(X, y):
    ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        mu1, mu0 = X[m & y].mean(0), X[m & ~y].mean(0)
        d = mu1 - mu0
        ok += ((X[i] @ d > ((mu1 + mu0) / 2) @ d) == y[i])
    return ok / len(y)

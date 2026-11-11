"""aşama 5: türkçe matematik sorularında aynı döngü var mı, döngüyü kesmek türkçede de işe yarıyor mu?

sorular: math-500 test 200-299. türkçesi bezir/MATH-500-multilingual'dan, unique_id ile eşleştirilip 5 soruda
düzeltildi (data/stage5_tr_200_299.json, data/stage5_tr_duzeltmeler.json). tahminler deneyden önce
results/stage5_tahminler.md'ye yazıldı.

işler:
- en_off, tr_off, tr_on: soru dili / talimat dili / düşünme modu
- geri_off: türkçe soruların ingilizceye geri çevrilmişi (data/stage5_geri.json). çeviri soruyu zorlaştırdı mı?
- trq_eninst_off: türkçe soru, ingilizce talimat
- zorla_tr / zorla_en: düşünme açık cevabı token bazlı döngü başında kesip "</think>" ekleyip 1024 token cevap.
  zorla_en, stage3_think_base_v2 (aynı sorular, ingilizce, düşünme açık) cevaplarını kullanıyor.

token bazlı döngü: 8 token'lık parça daha önce geçtiyse o token "tekrar". 256 token'lık pencerenin %90'ı tekrarsa
döngü orada başlıyor. kelime bazlı ölçü türkçede (eklemeli dil, daha uzun kelimeler) adil olmayabilir.

çalıştırma: python src/stage5_tr.py --data data --jobs en_off,tr_off,tr_on,zorla_tr,zorla_en,trq_eninst_off,geri_off
"""
import argparse, json, os, time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify
from stage3_ablate import run, grade

TALIMAT = {"en": "Please reason step by step, and put your final answer within \\boxed{}.",
           "tr": "Adım adım düşün ve son cevabını \\boxed{} içine yaz."}
JOBS = {
    "en_off": dict(soru="en", talimat="en", think=False),
    "tr_off": dict(soru="tr", talimat="tr", think=False),
    "tr_on": dict(soru="tr", talimat="tr", think=True),
    "trq_eninst_off": dict(soru="tr", talimat="en", think=False),
    "geri_off": dict(soru="geri", talimat="en", think=False),
    "zorla_tr": dict(zorla="stage5_tr_on", soru="tr", talimat="tr"),
    "zorla_en": dict(zorla="stage3_think_base_v2", soru="en", talimat="en"),
}
KAPAT = "\n</think>\n\n"


def dongu_token(ids, n=8, pencere=256, esik=0.9):
    seen, lab = set(), []
    for i in range(len(ids)):
        g = tuple(ids[max(0, i - n + 1): i + 1])
        lab.append(len(g) == n and g in seen)
        seen.add(g)
    run_ = sum(lab[:pencere])
    for i in range(len(lab) - pencere):
        if run_ >= esik * pencere:
            return i
        run_ += lab[i + pencere] - lab[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--jobs", default="en_off,tr_off,tr_on,zorla_tr,zorla_en,trq_eninst_off,geri_off")
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()

    name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda").eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    S = json.load(open(f"{a.data}/stage5_tr_200_299.json", encoding="utf-8"))
    geri_p = f"{a.data}/stage5_geri.json"
    if os.path.exists(geri_p):
        g = {r["i"]: r["geri"] for r in json.load(open(geri_p, encoding="utf-8"))}
        for r in S:
            r["geri"] = g[r["i"]]

    def prompt_fn(talimat, think):
        def f(q):
            msg = [{"role": "user", "content": q + "\n\n" + TALIMAT[talimat]}]
            kw = {"enable_thinking": True} if think else {}
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **kw)
        return f

    sp = f"{a.data}/stage5_summary.json"
    summary = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    for j in a.jobs.split(","):
        cfg, t0 = JOBS[j], time.time()
        pf = prompt_fn(cfg["talimat"], cfg.get("think", True))
        if "zorla" not in cfg:
            if cfg["soru"] == "geri" and not os.path.exists(geri_p):
                print(f"{j}: {geri_p} yok, atlanıyor", flush=True); continue
            ds = Dataset.from_list([{"problem": r[cfg["soru"]], "answer": r["answer"], "level": r["level"]} for r in S])
            print(f"{j} örnek istem:\n{pf(ds[0]['problem'])}", flush=True)
            res = run(model, tok, pf, im_end, ds, f"{a.data}/stage5_{j}.json", batch_size=a.batch, max_new=4096,
                      log=lambda s: print(s, flush=True))
            g = grade(res)
            g["ort_token"] = round(sum(r["n_tok"] for r in res) / len(res))
        else:
            src = json.load(open(f"{a.data}/{cfg['zorla']}.json", encoding="utf-8"))
            path = f"{a.data}/stage5_{j}.json"
            res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
            for s in range(len(res), len(src), 10):
                rows, qs = src[s:s + 10], S[s:s + 10]
                pre, ons = [], []
                for r in rows:
                    ids = tok(r["text"], add_special_tokens=False)["input_ids"]
                    o = dongu_token(ids)
                    ons.append(o)
                    pre.append(tok.decode(ids[:o]) if o is not None else r["text"])
                enc = tok([pf(q[cfg["soru"]]) + p + KAPAT for q, p in zip(qs, pre)], return_tensors="pt",
                          padding=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=1024, do_sample=False, pad_token_id=tok.pad_token_id)
                for i, r in enumerate(rows):
                    ids = out[i, enc["input_ids"].shape[1]:].tolist()
                    fin = im_end in ids or tok.eos_token_id in ids
                    cevap = tok.decode([t for t in ids if t != tok.pad_token_id], skip_special_tokens=True)
                    try:
                        ok = bool(verify(parse("$" + r["answer"] + "$"), parse(cevap)))
                    except Exception:
                        ok = False
                    res.append({"i": qs[i]["i"], "answer": r["answer"], "dongu_token": ons[i], "cevap": cevap,
                                "finished": fin, "correct": ok})
                json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"{j}: {len(res)}/{len(src)}", flush=True)
                del out, enc; torch.cuda.empty_cache()
            g = {"n": len(res), "dogru": sum(r["correct"] for r in res), "biten": sum(r["finished"] for r in res),
                 "dongusuz": sum(r["dongu_token"] is None for r in res)}
        summary[j] = g
        print(f"{j}: {g}  {(time.time() - t0) / 60:.0f} dk", flush=True)
        json.dump(summary, open(sp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("STAGE5_BITTI", flush=True)


if __name__ == "__main__":
    main()

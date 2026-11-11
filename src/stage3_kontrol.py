"""aşama 3, eleştiriye cevap veren kontroller.

1. rastgele yön kontrolü. bitirme yönünü 1.5 katıyla silmek takılmayı 33'ten 93'e çıkarıyordu, itmek 21'e indiriyordu.
   ama 15 katmandan herhangi bir yönü silmek modeli genel olarak bozabilir. aynı işlemleri iki rastgele yönle yapıyorum.
   - rand{1,2}_ters: 8-22. katlarda W - 1.5 u(uᵀW), bitirme yönündeki "ters çevir" ile aynı
   - rand{1,2}_guclu: 15. kat çıkışına -0.6 u, bitirme yönündeki "güçlü it" ile aynı
2. döngüyü kırmanın faydası. düşünme modunda 100 cevabın 95'i birebir döngüye giriyordu. çözme sırasında tekrar cezası
   (repetition_penalty 1.1) döngüyü dışarıdan kırıyor. doğru sayısı artarsa döngüyü kırmak gerçekten kazanç getiriyor.
   - think_reppen
3. düşünme modundan çıkarılmış tekrar yönü. önceki yönler düşünme kapalı cevaplardan bulunmuştu.
   - think_train: düşünme açık, sorular 0-99 (sadece yön için, test sorularıyla karışmasın)
   - think_newrep_a10 / think_newrep_a15: bu cevaplardan bulunan tekrar yönünü 200-299'da 8-22. katlardan silmek
4. sınır kontrolü. düşünme açık, 200-224. sorular, 16.384 token. birebir döngüye giren cevap daha uzun sınırda kurtuluyor mu?
   - think_16k
5. daha fazla soru. "doğru sayısı artmadı" sonucu 100 soruda zayıf. 300-399 sorularında taban ve hafif itme.
   - ext_base, ext_steer03
6. colab oturumu sıfırlanınca kütüphane sürümü değişti. newrep denemeleri think_base_v2 ile karşılaştırılıyor.
   - think_base_v2

her deneme data/stage3_{isim}.json'a her partide yazılıyor, özet data/stage3_kontrol_summary.json.
yarıda kalırsa aynı komut kaldığı yerden devam ediyor.
"""
import argparse, json, os, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage3_ablate import edit, undo, steer, run, grade, repeat_direction

L8_22 = list(range(8, 23))
JOBS = {
    "rand1_ters": dict(rand=1, edit=dict(layers=L8_22, alpha=1.5)),
    "rand1_guclu": dict(rand=1, steer=0.6),
    "rand2_ters": dict(rand=2, edit=dict(layers=L8_22, alpha=1.5)),
    "rand2_guclu": dict(rand=2, steer=0.6),
    "think_reppen": dict(think=True, gen=dict(repetition_penalty=1.1)),
    "think_16k": dict(think=True, rows=(200, 225), max_new=16384, batch=13),
    "think_train": dict(think=True, rows=(0, 100)),
    "think_newrep_a10": dict(think=True, newrep=dict(layers=L8_22, alpha=1.0)),
    "think_newrep_a15": dict(think=True, newrep=dict(layers=L8_22, alpha=1.5)),
    "ext_base": dict(rows=(300, 400)),
    "ext_steer03": dict(rows=(300, 400), steer_u15=0.3),
    "think_base_v2": dict(think=True),  # aynı oturumda düşünme modu tabanı (kütüphane sürümü değişince)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(JOBS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()

    name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda").eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    hidden = model.config.hidden_size

    def prompt_fn(think):
        def build_prompt(q):
            msg = [{"role": "user", "content": q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            kw = {"enable_thinking": True} if think else {}
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **kw)
        return build_prompt

    u15 = np.load(f"{a.data}/u_layer15.npy")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def rand_dir(seed):
        v = np.random.default_rng(seed).standard_normal(hidden).astype(np.float32)
        return v / np.linalg.norm(v)

    sp = f"{a.data}/stage3_kontrol_summary.json"
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    u_newrep = None
    orig_generate = model.generate

    for j in a.jobs.split(","):
        cfg = JOBS[j]
        think = cfg.get("think", False)
        lo, hi = cfg.get("rows", (200, 300))
        rows = ds.select(range(lo, hi))
        saved, hook = [], None
        t0 = time.time()

        if "rand" in cfg:
            u = rand_dir(cfg["rand"])
            print(f"{j}: rastgele yön, bitirme yönüyle kosinüs {float(u @ u15):.3f}", flush=True)
            if "edit" in cfg:
                saved = edit(model, u, **cfg["edit"])
            if "steer" in cfg:
                hook = steer(model, u, layer=15, c=cfg["steer"])
        if "steer_u15" in cfg:
            hook = steer(model, u15, layer=15, c=cfg["steer_u15"])
        if "newrep" in cfg:
            if u_newrep is None:
                train = json.load(open(f"{a.data}/stage3_think_train.json"))
                for r in train:
                    r["ids"] = tok(r["text"], add_special_tokens=False)["input_ids"]
                U, n_rep, n_new = repeat_direction(model, tok, prompt_fn(True), train, hidden=hidden)
                u_newrep = U[15]
                np.save(f"{a.data}/u_rep_think_layer15.npy", u_newrep)
                print(f"düşünme modu tekrar yönü: tekrar {n_rep}, yeni {n_new}, bitirme yönüyle kosinüs "
                      f"{float(u_newrep @ u15):.3f}", flush=True)
            saved = edit(model, u_newrep, **cfg["newrep"])
        if "gen" in cfg:
            model.generate = lambda *args, **kw: orig_generate(*args, **kw, **cfg["gen"])

        try:
            res = run(model, tok, prompt_fn(think), im_end, rows, f"{a.data}/stage3_{j}.json",
                      batch_size=cfg.get("batch", a.batch), max_new=cfg.get("max_new", 4096),
                      log=lambda s: print(s, flush=True))
        finally:
            undo(saved)
            if hook is not None:
                hook.remove()
            model.generate = orig_generate
        g = grade(res)
        g["ort_token"] = round(sum(r["n_tok"] for r in res) / len(res))
        summary[j] = g
        print(f"{j}: {g}  {(time.time() - t0) / 60:.0f} dk", flush=True)
        json.dump(summary, open(sp, "w"), indent=1)
    print("KONTROL_BITTI", flush=True)


if __name__ == "__main__":
    main()

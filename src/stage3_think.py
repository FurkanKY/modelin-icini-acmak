"""aşama 3, eksik denemeler: düşünme modu açık ve tekrar yönü denemelerinin yeniden çalıştırılması.

1. düşünme modu. mit qwen3.5-0.8b'yi düşünme modu açık kullandı ve cevapların %93'ü takılıyordu. bizde düşünme modu
   kapalıydı ve %34'ü takılıyordu. bu fark mit'nin büyük kazancını bizim neden göremediğimizi açıklayabilir.
   burada aynı 100 soruyu (200-299) düşünme modu açık soruyorum:
   - think_base: düzenlenmemiş model
   - think_steer03: 15. katmanda "biten" tarafa hafif itme (3. aşamada takılmayı en çok azaltan ayar)
   - think_rep_a15: tekrar yönünü 8-22. katmanlardan 1.5 katsayıyla silmek (mit p10c'nin karşılığı)
   yönler düşünme modu kapalı cevaplardan bulundu. düşünme modunda da işe yarıyorlarsa bu ayrıca bir bilgi.
2. tekrar yönü denemelerinin ham cevapları colab sıfırlanınca kaybolmuştu. aynı üç denemeyi ve düzenlenmemiş
   tabanı (base_v2, eskisinin üstüne yazmamak için ayrı isimle) yeniden çalıştırıyorum.

stage3_ablate.py'deki edit, undo, steer, run, grade ve repeat_direction'ı kullanıyor.
her deneme data/stage3_{isim}.json'a her partide yazılıyor. yarıda kalırsa aynı komut kaldığı yerden devam ediyor.
"""
import argparse, json, os, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from stage3_ablate import edit, undo, steer, run, grade, repeat_direction, REPEAT_RUNS

JOBS = {
    "think_base": dict(think=True),
    "think_steer03": dict(think=True, steer=0.3),
    "think_rep_a15": dict(think=True, rep=dict(REPEAT_RUNS)["rep_a15"]),
    "base_v2": dict(think=False),
    **{name: dict(think=False, rep=kw) for name, kw in REPEAT_RUNS},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(JOBS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--max_new", type=int, default=4096)
    a = ap.parse_args()

    name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda").eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def prompt_fn(think):
        def build_prompt(q):
            msg = [{"role": "user", "content": q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            kw = {"enable_thinking": True} if think else {}
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **kw)
        return build_prompt

    print("düşünme kapalı prompt:", repr(prompt_fn(False)("2+2?")[-80:]))
    print("düşünme açık prompt:  ", repr(prompt_fn(True)("2+2?")[-80:]))

    u15 = np.load(f"{a.data}/u_layer15.npy")
    res200 = json.load(open(f"{a.data}/stage2_200_raw.json"))
    test = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(200, 300))

    u_rep = None
    jobs = a.jobs.split(",")
    if any("rep" in JOBS[j] for j in jobs):
        t0 = time.time()
        U_rep, n_rep, n_new = repeat_direction(model, tok, prompt_fn(False), res200)
        u_rep = U_rep[15]
        print(f"tekrar yönü {time.time()-t0:.0f} sn, tekrar {n_rep}, yeni {n_new} (eskisi 126111 / 185213), "
              f"bitirme yönüyle kosinüs {float(u_rep @ u15):.3f} (eskisi 0.14)", flush=True)

    sp = f"{a.data}/stage3_think_summary.json"
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    for j in jobs:
        cfg = JOBS[j]
        t0 = time.time()
        saved, hook = [], None
        if "rep" in cfg:
            saved = edit(model, u_rep, **cfg["rep"])
        if "steer" in cfg:
            hook = steer(model, u15, layer=15, c=cfg["steer"])
        try:
            res = run(model, tok, prompt_fn(cfg["think"]), im_end, test, f"{a.data}/stage3_{j}.json",
                      batch_size=a.batch, max_new=a.max_new, log=lambda s: print(s, flush=True))
        finally:
            undo(saved)
            if hook is not None:
                hook.remove()
        g = grade(res)
        g["ort_token"] = round(sum(r["n_tok"] for r in res) / len(res))
        summary[j] = g
        print(f"{j}: {g}  {(time.time()-t0)/60:.0f} dk", flush=True)
        json.dump(summary, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main()

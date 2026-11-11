"""aşama 3, düşünme modu: döngü kırılırsa ne kazanılır?

düşünme modunda 100 cevabın 95'i birebir döngüye giriyor ve hiçbiri düşünme bloğunu kapatmıyor. ama döngüye
giren cevapların bir kısmında doğru cevap döngüden önce metinde geçiyor. döngüyü dışarıdan keserek modeli
cevap vermeye zorluyorum ("budget forcing"). modelin içine dokunmuyorum.

- dongu: think_base_v2 cevabını döngünün başladığı kelimede kesip "</think>" ekliyorum, model son cevabı yazıyor
- son: cevabın tamamının (4096 token) sonuna "</think>" ekliyorum. döngüde kesmekle sona kadar beklemek arasındaki fark
  döngünün kendisinin bir şey kaybettirip kaybettirmediğini gösteriyor

döngüye girmeyen cevaplarda iki koşul aynı (cevabın tamamı). puanlama sadece "</think>"ten sonra yazılan kısımda.

çalıştırma: python src/stage3_zorla.py --data data
"""
import argparse, json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify
from stage3_think_analiz import dongu_baslangici

KAPAT = "\n</think>\n\n"


def kes(text, kelime):
    """metni kelime numarasından kesiyor, boşlukları koruyarak."""
    if kelime is None:
        return text
    m = list(re.finditer(r"\S+", text))
    return text[:m[kelime].start()] if kelime < len(m) else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--src", default="think_base_v2")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max_new", type=int, default=1024)
    a = ap.parse_args()

    name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda").eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def prompt(q):
        msg = [{"role": "user", "content": q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    src = json.load(open(f"{a.data}/stage3_{a.src}.json"))
    summary = {}
    for kosul in ("dongu", "son"):
        path = f"{a.data}/stage3_zorla_{kosul}.json"
        res = json.load(open(path)) if os.path.exists(path) else []
        for s in range(len(res), len(src), a.batch):
            rows = src[s:s + a.batch]
            pre = [kes(r["text"], dongu_baslangici(r["text"]) if kosul == "dongu" else None) for r in rows]
            enc = tok([prompt(r["problem"]) + p + KAPAT for r, p in zip(rows, pre)], return_tensors="pt",
                      padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for i, r in enumerate(rows):
                ids = out[i, enc["input_ids"].shape[1]:].tolist()
                fin = im_end in ids or tok.eos_token_id in ids
                ids = [t for t in ids if t != tok.pad_token_id]
                cevap = tok.decode(ids, skip_special_tokens=True)
                try:
                    ok = bool(verify(parse("$" + r["answer"] + "$"), parse(cevap)))
                except Exception:
                    ok = False
                res.append({"problem": r["problem"], "answer": r["answer"], "kesilen_kelime": len(pre[i].split()),
                            "cevap": cevap, "finished": fin, "correct": ok})
            json.dump(res, open(path, "w"))
            print(f"{kosul}: {len(res)}/{len(src)}", flush=True)
            del out, enc; torch.cuda.empty_cache()
        summary[kosul] = {"n": len(res), "dogru": sum(r["correct"] for r in res), "biten": sum(r["finished"] for r in res),
                          "ort_kesilen_kelime": round(sum(r["kesilen_kelime"] for r in res) / len(res))}
        print(kosul, summary[kosul], flush=True)
    json.dump(summary, open(f"{a.data}/stage3_zorla_summary.json", "w"), indent=1)
    print("ZORLA_BITTI", flush=True)


if __name__ == "__main__":
    main()

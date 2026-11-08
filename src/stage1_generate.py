"""aşama 1: qwen3.5-0.8b'ye math-500'den 50 soru soruyorum ve kaç cevabın takıldığını sayıyorum.

colab'da, t4 gpu ile hücre hücre çalıştırdım. ayarlar:
- greedy decoding (do_sample=False): model her adımda en olası token'ı seçiyor, zar atmıyor.
- düşünme modu kapalı. chat şablonunun varsayılanı bu.
- ilk denemede sınır 1536 token'dı, sonra bitiremeyenler için 4096'ya çıkardım. burada direkt 4096.
"""
import json, re, time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(50))


def build_prompt(q):
    msgs = [{"role": "user", "content": q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(max_new=4096, batch_size=25):
    results = []
    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        enc = tok([build_prompt(q) for q in batch["problem"]], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, row in enumerate(batch):
            ids = out[i, enc["input_ids"].shape[1]:].tolist()
            finished = IM_END in ids or tok.eos_token_id in ids
            ids = [t for t in ids if t != tok.pad_token_id]
            results.append({"idx": start + i, "problem": row["problem"], "answer": row["answer"],
                            "ids": ids, "text": tok.decode(ids, skip_special_tokens=True), "finished": finished})
    return results


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7; depth = 1; j = i
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i:j - 1] if depth == 0 else None


def norm(x):
    return re.sub(r"\\left|\\right|\\!|\s|\$|\\dfrac|\\tfrac",
                  lambda m: "\\frac" if "frac" in m.group() else "", x or "")


def is_verbatim_loop(ids, n=20, tail=400):
    """son 400 token'daki 20'lik parçaların yarısından fazlası tekrarsa model birebir aynı şeyi yazıyordur.
    takılmaların çoğu böyle olmuyor, model genelde aynı şüpheyi farklı cümlelerle döndürüyor. o yüzden
    asıl ölçüm "4096 token'da bitti mi" sorusu, bu fonksiyon sadece ek bilgi veriyor."""
    t = ids[-tail:]
    grams = [tuple(t[k:k + n]) for k in range(len(t) - n)]
    return bool(grams) and len(set(grams)) / len(grams) < 0.5


if __name__ == "__main__":
    results = generate()
    for r in results:
        r["pred"] = last_boxed(r["text"])
        r["correct"] = r["pred"] is not None and norm(r["pred"]) == norm(r["answer"])
        r["verbatim_loop"] = (not r["finished"]) and is_verbatim_loop(r["ids"])
    json.dump(results, open("stage1_raw_4096.json", "w"))

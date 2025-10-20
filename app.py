import re, json, torch
from flask import Flask, request, render_template, jsonify
from langdetect import detect
from banglish_transliterate import to_bangla
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. English grammar fix (T5-small is fast & free)
GRAMMAR_MODEL = "pszemraj/flan-t5-small-grammar-synthesis"
g_tok = AutoTokenizer.from_pretrained(GRAMMAR_MODEL)
g_mod = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL).to(device)

# 2. English ↔ Bengali MT (Helsinki OPUS small)
EN2BN_NAME = "Helsinki-NLP/opus-mt-en-bn"
BN2EN_NAME = "Helsinki-NLP/opus-mt-bn-en"
en2bn_tok = AutoTokenizer.from_pretrained(EN2BN_NAME)
en2bn_mod = AutoModelForSeq2SeqLM.from_pretrained(EN2BN_NAME).to(device)
bn2en_tok = AutoTokenizer.from_pretrained(BN2EN_NAME)
bn2en_mod = AutoModelForSeq2SeqLM.from_pretrained(BN2EN_NAME).to(device)

# ---------- helpers ----------
def correct_english(sent: str) -> str:
    prompt = "grammar: " + sent
    tokens = g_tok(prompt, return_tensors="pt").input_ids.to(device)
    out = g_mod.generate(tokens, max_length=128, num_beams=3, early_stopping=True)
    return g_tok.decode(out[0], skip_special_tokens=True)

def translate(text, tokenizer, model):
    batch = tokenizer(text, return_tensors="pt", padding=True).to(device)
    gen = model.generate(**batch, max_length=128, num_beams=3, early_stopping=True)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

def is_banglish(txt: str) -> bool:
    # very light heuristic
    return bool(re.search(r'(?i)\b(ami|tumi|valo|bhalobashi|kemon|acho|tomay)\b', txt))

# ---------- routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    txt = request.form.get("text", "").strip()
    if not txt:
        return jsonify({"error": "Empty input"}), 400

    lang = detect(txt)          # 'en' or 'bn'
    if is_banglish(txt):
        lang = "banglish"

    reply = {"detected": lang, "original": txt}

    # branch 1: Banglish → Bengali script + English translation
    if lang == "banglish":
        bn = to_bangla(txt)
        reply["bengali"] = bn
        reply["english"] = translate(bn, bn2en_tok, bn2en_mod)
        return jsonify(reply)

    # branch 2: English grammar fix + Bengali translation
    if lang == "en":
        fixed = correct_english(txt)
        reply["corrected"] = fixed
        reply["bengali"]   = translate(fixed, en2bn_tok, en2bn_mod)
        return jsonify(reply)

    # branch 3: Bengali input → English translation
    if lang == "bn":
        reply["english"] = translate(txt, bn2en_tok, bn2en_mod)
        return jsonify(reply)

    # fallback
    return jsonify(reply)

# ---------- run ----------
if __name__ == "__main__":
    app.run(debug=True)

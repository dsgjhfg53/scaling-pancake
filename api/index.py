from flask import Flask, request, jsonify
from langdetect import detect
from banglish_transliterate import to_bangla
import requests, os

app = Flask(__name__)

HF_API = "https://api-inference.huggingface.co/models"
HEADERS = {"Authorization": f'Bearer {os.getenv("HF_TOKEN")}'}

def hf_infer(model: str, payload: str) -> str:
    """Call HF Inference API (free)"""
    r = requests.post(f"{HF_API}/{model}", headers=HEADERS, json={"inputs": payload})
    if r.status_code != 200:
        return payload          # fallback
    return r.json()[0]["generated_text"]

@app.route("/", methods=["POST"])
def process():
    txt = request.get_json(force=True).get("text", "").strip()
    if not txt:
        return jsonify({"error": "empty"}), 400

    lang = detect(txt)
    reply = {"detected": lang, "original": txt}

    # Banglish → Bengali
    if lang == "bn" or "ami" in txt.lower():
        bn = to_bangla(txt)
        reply["bengali"] = bn
        reply["english"] = hf_infer("Helsinki-NLP/opus-mt-bn-en", bn)
        return jsonify(reply)

    # English grammar fix
    if lang == "en":
        fixed = hf_infer("pszemraj/flan-t5-small-grammar-synthesis", "grammar: " + txt)
        reply["corrected"] = fixed
        reply["bengali"]   = hf_infer("Helsinki-NLP/opus-mt-en-bn", fixed)
        return jsonify(reply)

    return jsonify(reply)

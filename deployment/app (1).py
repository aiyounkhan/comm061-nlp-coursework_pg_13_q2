"""
Q5.1 -- BESSTIE Variety-Aware Classifier
Gradio deployment app

Run with: python app.py
Requires: pip install gradio datasets scikit-learn
"""
import gradio as gr
import pandas as pd
import time
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -- Load dataset and train variety-specific models at startup --
print("Loading dataset...")
ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
train_df = pd.DataFrame(ds["train"])

variety_models = {}
for variety in ["en-AU", "en-IN", "en-UK"]:
    v = train_df[train_df["variety"] == variety].reset_index(drop=True)
    ts = TfidfVectorizer(max_features=5000)
    ls = LogisticRegression(max_iter=1000)
    ls.fit(ts.fit_transform(v["text"]), v["Sentiment"])
    tr = TfidfVectorizer(max_features=5000)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(tr.fit_transform(v["text"]), v["Sarcasm"])
    variety_models[variety] = {"sentiment": (ts, ls), "sarcasm": (tr, lr)}
    print(f"  Loaded model for {variety}")

print("All models ready.")

# -- LoRA adapter loading (reference -- use once adapters are on HuggingFace) --
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ADAPTERS = {
#     "en-AU": "<hf-username>/tinyllama-lora-sarcasm-en-au",
#     "en-IN": "<hf-username>/tinyllama-lora-sarcasm-en-in",
#     "en-UK": "<hf-username>/tinyllama-lora-sarcasm-en-uk",
# }
# base_model = AutoModelForCausalLM.from_pretrained(BASE)
# adapters = {v: PeftModel.from_pretrained(base_model, r) for v, r in ADAPTERS.items()}


def predict(text, variety):
    if not text.strip():
        return "Please enter text.", "", "", "", ""
    t0 = time.time()

    ts, ls = variety_models[variety]["sentiment"]
    sent_pred = ls.predict(ts.transform([text]))[0]
    sent_prob = ls.predict_proba(ts.transform([text]))[0]
    sent_out  = "Positive" if sent_pred == 1 else "Negative"
    sent_conf = f"{max(sent_prob)*100:.1f}%"

    tr, lr = variety_models[variety]["sarcasm"]
    sarc_pred = lr.predict(tr.transform([text]))[0]
    sarc_prob = lr.predict_proba(tr.transform([text]))[0]
    sarc_out  = "Sarcastic" if sarc_pred == 1 else "Not Sarcastic"
    sarc_conf = f"{max(sarc_prob)*100:.1f}%"

    ms = f"{(time.time()-t0)*1000:.1f} ms"
    return sent_out, sent_conf, sarc_out, sarc_conf, ms


EXAMPLES = [
    ["Absolute legend, parked his ute right across my driveway. Good onya, mate.", "en-AU"],
    ["Yeah the wifi is sooo reliable here, totally worth 3 hours waiting.", "en-AU"],
    ["Coz we all have free internet.", "en-IN"],
    ["Yaar this place is just too good, totally worth the 2 hour wait!", "en-IN"],
    ["Traditional friendly pub. Excellent beer.", "en-UK"],
    ["Oh brilliant, another 45-minute Tube delay. Just what I needed.", "en-UK"],
]

with gr.Blocks(title="BESSTIE Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# BESSTIE -- Variety-Aware Sentiment & Sarcasm Classifier\n"
        "**COMM061 NLP Coursework | University of Surrey | PG13**\n\n"
        "Select the English variety matching your text, then click Classify. "
        "The backend switches to the model trained specifically on that variety."
    )
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="Input Text", lines=4,
                              placeholder="e.g. Oh great, another delay on the Tube...")
            var = gr.Radio(["en-AU", "en-IN", "en-UK"], value="en-UK",
                            label="English Variety",
                            info="Australian | Indian | British")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            so = gr.Textbox(label="Sentiment",            interactive=False)
            sc = gr.Textbox(label="Sentiment Confidence", interactive=False)
            ro = gr.Textbox(label="Sarcasm",              interactive=False)
            rc = gr.Textbox(label="Sarcasm Confidence",   interactive=False)
            to = gr.Textbox(label="Inference Time",       interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=[txt, var],
                label="Try these examples (covers all 3 varieties)")
    btn.click(predict, inputs=[txt, var], outputs=[so, sc, ro, rc, to])
    gr.Markdown(
        "---\nVariety switch = dict lookup only. "
        "Same efficiency principle as LoRA adapter swap: base model stays "
        "in memory, only adapter weights change per request."
    )

if __name__ == "__main__":
    demo.launch(share=True)

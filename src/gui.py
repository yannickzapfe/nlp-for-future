import torch
import gradio as gr

from transformers import AutoModelForSequenceClassification, AutoTokenizer

review = "Absolutely ridiculous. Should be listed under fantasy or fiction."

model_names = [
    'bert-base-cased',
    'distilbert-base-cased',
    'distilbert-base-uncased',
    'distilroberta-base'
]

tokenizers = {}
models = {}

for model_name in model_names:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except EnvironmentError:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizers[model_name] = tokenizer

    model_folder_name = f'{model_name}-fine_tuned-14000(20000)_early_ne10'
    base_path = f'./local_{model_name}/{model_folder_name}'
    model_path = f'{base_path}/final'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load pre-trained model from harddrive
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        problem_type='multi_label_classification'
    ).to(device)

    models[model_name] = model


def int_to_stars(i):
    i = int(i)
    return i * "\u2605" + (5 - i) * "\u2606"


def classify_review(review, model_name):
    tokenizer = tokenizers[model_name]
    review_encoded = tokenizer(review, return_tensors="pt", truncation=True).to(device)

    model = models[model_name]
    with torch.no_grad():
        logits = model(**review_encoded).logits

    prediction = float(logits.argmax().item() + 1)
    return int_to_stars(prediction)


review_interface = gr.Interface(
    fn=classify_review,
    theme="default",
    css=".footer {display:none !important}",
    inputs=["text", gr.Dropdown(model_names)],
    outputs=["text"]
)

review_interface.launch(share=True)

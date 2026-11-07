#!/usr/bin/env python3

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def get_word_embedding(model, tokenizer, sentence: str, target_word: str, use_last_4_layers: bool = False):
    lower_sent = sentence.lower()
    lower_target = target_word.lower()
    
    start_char = lower_sent.find(lower_target)
    if start_char == -1:
        raise ValueError(f"Slovo '{target_word}' sa nenachádza vo vete: '{sentence}'")
    end_char = start_char + len(lower_target)

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    offset_mapping = encoded.pop("offset_mapping")[0]
    
    target_token_indices = []
    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue
        if max(start_char, tok_start.item()) < min(end_char, tok_end.item()):
            target_token_indices.append(idx)
            
    if not target_token_indices:
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        for idx, tok in enumerate(tokens):
            clean_tok = tok.replace("##", "").replace(" ", "").lower()
            if clean_tok in lower_target or lower_target in clean_tok:
                target_token_indices.append(idx)

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        
    if use_last_4_layers:
        hidden_states = outputs.hidden_states
        last_4_layers = torch.stack(hidden_states[-4:], dim=0) # [4, batch, seq_len, hidden]
        layer_mean = last_4_layers.mean(dim=0) # [batch, seq_len, hidden]
        token_embeddings = layer_mean[0] # [seq_len, hidden]
    else:
        token_embeddings = outputs.last_hidden_state[0] # [seq_len, hidden]

    target_embeddings = token_embeddings[target_token_indices]
    word_embedding = target_embeddings.mean(dim=0).cpu().numpy()
    
    return word_embedding, target_token_indices


def run_experiment(model_name: str, examples: list, target_word: str, lang_code: str = "sk"):
    print(f"\n{'='*75}")
    print(f"Načítavam model: {model_name}")
    print(f"Cieľové slovo (polysémia): '{target_word}'")
    print(f"{'='*75}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    labels = []
    
    for i, ex in enumerate(examples, 1):
        sent = ex["sentence"]
        sense = ex["sense"]
        emb, tokens_idx = get_word_embedding(model, tokenizer, sent, target_word)
        embeddings.append(emb)
        labels.append(f"Veta {i} ({sense})")
        
        print(f"[{i}] Zmysel: {sense}")
        print(f"    Veta: \"{sent}\"")
        print(f"    Token indexy v sekvencii: {tokens_idx}")
        print(f"    Rozmer embeddingu: {emb.shape[0]} dimenzií\n")

    sim_matrix = cosine_similarity(embeddings) # type: ignore
    
    print("-" * 75)
    print("Matica kosínusovej podobnosti kontextových embeddingov:")
    print("-" * 75)
    
    header = f"{'Veta':<25}" + "".join([f"{f'V{j+1}':>10}" for j in range(len(examples))])
    print(header)
    print("-" * len(header))
    for i in range(len(examples)):
        row_str = f"Veta {i+1} ({examples[i]['sense'][:12]}):"
        row_str = f"{row_str:<25}" + "".join([f"{sim_matrix[i, j]:>10.4f}" for j in range(len(examples))])
        print(row_str)
        
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Kosínusová podobnosť'}
    )
    plt.title(f"Kosínusová podobnosť kontextových embeddingov slova '{target_word}'\nModel: {model_name}", fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_img = f"similarity_matrix_{lang_code}.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"\nGraf podobností bol uložený do: {output_img}")


def main():
    # 1. Príklad v slovenčine (slovo 'zámok' - hrad / stavba vs. mechanický uzáver dverí)
    slovak_examples = [
        {
            "sentence": "Starobylý zámok sa týči na vysokom skalnatom kopci.",
            "sense": "architektúra (hrad)"
        },
        {
            "sentence": "Kráľovský zámok a jeho nádvorie navštevujú tisíce turistov.",
            "sense": "architektúra (hrad)"
        },
        {
            "sentence": "Zámočník rýchlo vymenil poškodený zámok na vchodových dverách.",
            "sense": "mechanizmus (dvere)"
        },
        {
            "sentence": "Na bráne visel ťažký oceľový zámok proti vlámaniu.",
            "sense": "mechanizmus (visiaci)"
        }
    ]

    # 2. Príklad v angličtine (slovo 'bank' - finančná inštitúcia vs. breh rieky)
    english_examples = [
        {
            "sentence": "I deposited all my savings into a commercial bank account.",
            "sense": "finances (institution)"
        },
        {
            "sentence": "The bank approved our mortgage loan with low interest rates.",
            "sense": "finances (institution)"
        },
        {
            "sentence": "We sat on the muddy river bank fishing until the sunset.",
            "sense": "nature (river edge)"
        },
        {
            "sentence": "During the flood, the water overflowed the river bank.",
            "sense": "nature (river edge)"
        }
    ]

    run_experiment(
        model_name="bert-base-multilingual-cased",
        examples=slovak_examples,
        target_word="zámok",
        lang_code="sk"
    )

    run_experiment(
        model_name="bert-base-uncased",
        examples=english_examples,
        target_word="bank",
        lang_code="en"
    )


if __name__ == "__main__":
    main()

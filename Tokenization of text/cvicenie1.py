import os
import re

def parse_conllu(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    #for example #text = Nema peniaze, ale ja mám
    #blocks = ["# sent_id = 1\n# text = Nema peniaze, ale ja mám.\n1-2\tNema\t...\n1\tne\t...\n2\tma\t...\n3\tpeniaze\t...\n4\t,\t...\n5\tale\t...\n6\tja\t...\n7\tmám\t...\tSpaceAfter=No\n8\t.\t..."]

    sentences = []
    for block in blocks:
        orig_text = ""
        surface_tokens, syntactic_words, reconstructed_parts = [], [], [] #surface - two word in one, syntactic - one word
        skip_syntactic_until = -1

        for line in block.strip().split("\n"):
            line = line.strip()
            if not line: continue
            
            if line.startswith("# text ="):
                orig_text = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"): continue

            parts = line.split("\t")
            if len(parts) < 10: continue

            token_id, form, misc = parts[0], parts[1], parts[9]

            if "." in token_id: continue 

            if token_id.isdigit():
                syntactic_words.append(form)

            if "-" in token_id:
                skip_syntactic_until = int(token_id.split("-")[1])
                surface_tokens.append(form)
                reconstructed_parts.append(form)
                if "SpaceAfter=No" not in misc: reconstructed_parts.append(" ")
            else:
                if int(token_id) <= skip_syntactic_until: continue
                surface_tokens.append(form)
                reconstructed_parts.append(form)
                if "SpaceAfter=No" not in misc: reconstructed_parts.append(" ")

        sentences.append({
            "orig_text": orig_text,
            "reconstructed_text": "".join(reconstructed_parts).strip(),
            "surface_tokens": surface_tokens,
            "syntactic_words": syntactic_words
        })
    return sentences

def custom_tokenizer(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def evaluate_corpus(sentences, tokenizer_func, token_type="surface"):
    total_matched, total_mismatched, total_gold = 0, 0, 0

    for sent in sentences:
        gold = sent["surface_tokens"] if token_type == "surface" else sent["syntactic_words"]
        pred = tokenizer_func(sent["reconstructed_text"])
        
        total_gold += len(gold)
        matched = sum(1 for i in range(min(len(gold), len(pred))) if gold[i] == pred[i])
        mismatched = (min(len(gold), len(pred)) - matched) + abs(len(gold) - len(pred))
        total_matched += matched
        total_mismatched += mismatched

    accuracy = (total_matched / total_gold * 100) if total_gold > 0 else 0
    return total_matched, total_mismatched, accuracy

def main():
    files = ["sk_snk-ud-dev.conllu", "sk_snk-ud-test.conllu"]
    for f in files:
        if os.path.exists(f):
            data = parse_conllu(f)
            matched, mismatched, acc = evaluate_corpus(data, custom_tokenizer, "surface")
            print(f"File: {f}")
            print(f"  - Zhodujúce sa tokeny (Matched)    : {matched}")
            print(f"  - Rozdelené inak (Mismatched)      : {mismatched}")
            print(f"  - Úspešnosť (Accuracy)             : {acc:.2f}%\n")

if __name__ == "__main__":
    main()

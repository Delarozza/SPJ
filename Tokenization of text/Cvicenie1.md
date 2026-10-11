# 🧠 Cvičenie 1: Rekonštrukcia textu a tokenizácia v CoNLL-U

> **Oblasť:** Spracovanie prirodzeného jazyka (NLP)  
> **Jazyk:** Slovenský (Slovak UD - Slovak National Corpus)  
> **Cieľ:** Pochopiť štruktúru formátu CoNLL-U, rekonštruovať pôvodný súvislý text so zachovaním medzier a vytvoriť vlastný tokenizer pomocou regulárnych výrazov bez externých NLP knižníc.

---

## 📌 1. Čo je formát CoNLL-U?
CoNLL-U je štandardizovaný textový formát projektu **Universal Dependencies (UD)** používaný na anotáciu morfológie a závislostnej syntaxe.

### Štruktúra súboru:
- Vety sú od seba oddelené **prázdnym riadkom** (`\n\n`).
- Komentáre a metadáta začínajú mriežkou `#` (napr. `# sent_id = 1`, `# text = Veta...`).
- Každý token/slovo je na samostatnom riadku a má presne **10 stĺpcov** oddelených tabulátorom (`\t`):

| Stĺpec # | Názov poľa | Význam | Príklad |
|:---:|:---|:---|:---|
| **1** | `ID` | Index tokenu (číslo, rozsah alebo desatinné číslo) | `1`, `10-11`, `1.1` |
| **2** | `FORM` | Slovná forma (text tak, ako je napísaný) | `pes`, `aby`, `,` |
| **3** | `LEMMA` | Základný tvar slova | `pes`, `byť` |
| **4** | `UPOS` | Univerzálny slovný druh | `NOUN`, `VERB`, `PUNCT` |
| **5** | `XPOS` | Jazykovo špecifická morfologická značka | `SSms1` |
| **6** | `FEATS` | Morfologické kategórie (pád, rod, číslo...) | `Case=Nom\|Gender=Masc` |
| **7** | `HEAD` | ID riadiaceho uzla v strome syntaxe | `2` |
| **8** | `DEPREL` | Typ syntaktického vzťahu | `nsubj`, `obj` |
| **9** | `DEPS` | Rozšírené závislosti | `_` |
| **10** | `MISC` | Dodatočné informácie (kľúčové pre medzery) | `SpaceAfter=No` |

---

## 🧩 2. Kľúčové koncepty a rozdiely v tokenoch

### A. Surface Tokens (Povrchové tokeny) vs. Syntactic Words (Syntaktické slová)
V textoch existujú tzv. **viacslovné tokeny (Multi-Word Tokens - MWT)**. V slovenčine je typickým príkladom spojka **„aby“** alebo zámeno **„naňho“**:

```text
10-11   aby        _      _      _      _
10      aby        aby    SCONJ  _      _
11      by         by     AUX    _      _
```
- **Surface token (`10-11 aby`):** Fyzické slovo v texte tak, ako ho vidí čitateľ.
- **Syntaktické slová (`10 aby` + `11 by`):** Gramaticky rozdelené prvky (spojka + kondicionálna častica).

> 💡 **Prečo vzniká rozdiel v úspešnosti tokenizéra:**  
> - Pri porovnaní s **povrchovými tokenmi (surface)** dosahuje regulárny výraz `r"\w+|[^\w\s]"` úspešnosť **100.00%**.  
> - Pri porovnaní so **syntaktickými slovami (syntactic)** je úspešnosť cca **99.64%**, pretože jednoduchý regex nedokáže bez slovníka rozdeliť jedno písané slovo `aby` na dve samostatné slová `aby` a `by`.

### B. Riešenie medzier (`SpaceAfter=No`)
V pôvodnom texte sa medzery nevyskytujú všade (napr. pred čiarkou alebo bodkou medzera nie je). V CoNLL-U to riadi stĺpec č. 10 (`MISC`):
- Ak obsahuje príznak `SpaceAfter=No`, za daným tokenom sa **nepridáva medzera**.
- V opačnom prípade sa pridá medzera `' '`.

---

## 💻 3. Implementácia v Pythone

### Rekonštrukcia textu (bez externých knižníc):
```python
def parse_conllu(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    sentences = []
    for block in blocks:
        orig_text = ""
        surface_tokens, syntactic_words, reconstructed_parts = [], [], []
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
            if "." in token_id: continue # vynechanie elipsy

            if token_id.isdigit():
                syntactic_words.append(form)

            if "-" in token_id: # MWT (napr. 10-11 aby)
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
```

### Vlastný regex tokenizer:
```python
import re

def custom_tokenizer(text):
    # \w+      -> zachytí celé slová / alfanumerické znaky vrátane diakritiky
    # [^\w\s]  -> zachytí každý jednotlivý interpunkčný znak (čiarky, bodky, zátvorky)
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
```

---

## 📊 4. Hlavné zistenia a zhrnutie
1. **Regulárne výrazy sú silné, ale majú limity:** Regex dokáže perfektne segmentovať povrchový text na slová a interpunkciu, ale nedokáže vykonávať morfo-syntaktickú dekompozíciu (napr. rozpoznať skryté morfémy v `aby` -> `aby` + `by`).
2. **Rekonštrukcia textu:** Vďaka štítku `SpaceAfter=No` je možné spätne rekonštruovať 100% zhodný pôvodný text z tabuľkových anotačných dát bez straty formátovania.

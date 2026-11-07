# Cvičenie: Kontextové embeddingy a Transformer (SPj)

Tento priečinok obsahuje vypracovanie cvičenia z predmetu **Spracovanie prirodzeného jazyka (SPj)** zameraného na **kontextové embeddingy, architektúru ELMo a Transformer (enkóderové a dekóderové modely)**.

---

## 📂 Štruktúra súborov

* **[`Instruction`](./Instruction)** — pôvodné zadanie a otázky k cvičeniu.
* **[`Cvicenie_Transformer.md`](./Cvicenie_Transformer.md)** — kompletná teoretická správa a odpovede na všetky otázky zo zadania v **slovenčine**.
* **[`Cvicenie_Transformer_RU.md`](./Cvicenie_Transformer_RU.md)** — podrobný teoretický rozbor a vysvetlenia v **ruštine** (pre jednoduchšie štúdium).
* **[`task.py`](./task.py)** — spustiteľný Python skript realizujúci praktickú časť cvičenia (extrakcia embeddingov pomocou Transformer modelov, výpočet kosínusovej podobnosti, vizualizácia heatmapy).
* **`similarity_matrix_sk.png`** — matica podobností pre slovenské polysémne slovo *„zámok“*.
* **`similarity_matrix_en.png`** — matica podobností pre anglické polysémne slovo *„bank“*.

---

## Ako spustiť praktickú časť

Najjednoduchší spôsob priamo z priečinka `Transformer`:

```bash
./run.sh
```

Alebo cez Python z virtuálneho prostredia:

```bash
# Priame spustenie:
.venv/bin/python task.py

# Alebo cez aktiváciu:
source .venv/bin/activate
python task.py
```

Skript automaticky:
1. Načíta model `bert-base-multilingual-cased` a vyhodnotí kontextové embeddingy pre slovo **„zámok“** v 4 rôznych vetách (architektonická pamiatka vs. mechanizmus zámku na dverách).
2. Načíta model `bert-base-uncased` a vyhodnotí kontextové embeddingy pre slovo **„bank“** (finančná inštitúcia vs. breh rieky).
3. Vypočíta maticu kosínusovej podobnosti a porovná ju s teoretickým výstupom statických embeddingov.
4. Uloží grafické heatmapy do súborov `similarity_matrix_sk.png` a `similarity_matrix_en.png`.

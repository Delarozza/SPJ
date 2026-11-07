# Kontextové embeddingy a Transformer architektúra


## 1. Kontextové vs. Statické embeddingy

### 1.1 Princíp a obmedzenia statických embeddingov
V klasických modeloch distribuovanej sémantiky (**Word2Vec (Skip-Gram, CBOW)**, **GloVe**, **FastText**) má každé slovo $w$ zo slovníka $V$ priradený **práve jeden statický vektor** $\vec{v}_w \in \mathbb{R}^d$.
* **Statické priradenie:** $f(w) = \vec{v}_w$ bez ohľadu na to, v akej vete, gramatickom tvare či význame sa slovo nachádza.
* **Problém polysémie a homonymie:** Slová s viacerými významami (napr. *„zámok“* ako historická stavba vs. *„zámok“* na dverách, alebo v angličtine *„bank“* ako finančná inštitúcia vs. breh rieky) sú nútené zdieľať jeden vektor. Tento vektor je v praxi váženým priemerom všetkých jeho významov v trénovacom korpuse, čo vedie k strate presného sémantického rozlíšenia.

### 1.2 Princíp kontextových embeddingov
Kontextové modely reprezentujú slovo ako funkciu celej vstupnej postupnosti:
$$\vec{h}_i = f(w_i \mid w_1, w_2, \dots, w_n)$$
* Vektor slova $w_i$ sa počíta dynamicky na základe celého okolitého kontextu (ľavého aj pravého).
* To umožňuje, aby to isté slovo v dvoch rôznych kontextoch získalo odlišné súradnice v latentnom priestore, presne odrážajúce jeho aktuálny význam.

| Vlastnosť | Statické embeddingy (Word2Vec, GloVe) | Kontextové embeddingy (ELMo, BERT, RoBERTa) |
| :--- | :--- | :--- |
| **Závislosť na kontexte** | Žiadna (1 slovo = 1 fixný vektor) | Dynamická (vektor závisí od celej vety) |
| **Polysémia** | Nedokážu rozlíšiť významy | Úspešne rozlišujú významy podľa kontextu |
| **OOV (Out-Of-Vocabulary)** | Neschopnosť reprezentovať neznáme slová (rieši čiastočne subword FastText) | Použitie subword tokenizácie (BPE, WordPiece) |
| **Výpočtová náročnosť** | Nízka (priame vyhľadanie v lookup tabuľke $O(1)$) | Vyššia (prechod neurónovou sieťou pri každom vstupe) |

---

## 2. ELMo (Embeddings from Language Models)

Architektúra **ELMo** (Peters et al., 2018) bola jedným z prvých úspešných modelov hlbokých kontextových reprezentácií, postavená na obojsmerných rekurentných sieťach (**biLM - Bidirectional Language Model**).

```
        Výstupný kombinovaný embedding: e_k = γ * Σ s_j * h_{k,j}
                             ▲
              ┌──────────────┴──────────────┐
              │  Vážený súčet vrstiev (s_j) │
              └──────────────▲──────────────┘
                             │
     [Vrstva 2: BiLSTM] ──► h_{k,2}^{LM} (Komplexná sémantika, pragmatika)
              ▲
     [Vrstva 1: BiLSTM] ──► h_{k,1}^{LM} (Syntax, slovné druhy, morfológia)
              ▲
     [Vrstva 0: Char-CNN] ─► h_{k,0}     (Znaková reprezentácia, sub-word)
              ▲
        Vstupné tokeny: w_1, w_2, ..., w_k
```

### 2.1 Úloha siete BiLSTM pri spracovaní textu
* Štandardný dopredný jazykový model odhaduje pravdepodobnosť tokenu $t_k$ na základe histórie: $P(t_k \mid t_1, \dots, t_{k-1})$.
* Spätný jazykový model odhaduje pravdepodobnosť na základe budúcich tokenov: $P(t_k \mid t_{k+1}, \dots, t_N)$.
* **BiLSTM** spája dopredný (forward) a spätný (backward) LSTM prechod:
  $$\vec{h}_{k, j}^{LM} = \left[ \vec{h}_{k, j}^{\text{forward}} \,;\, \vec{h}_{k, j}^{\text{backward}} \right]$$
  Vďaka tomu má model v každom bode $k$ informácie z oboch strán vety.

### 2.2 Aké informácie zachytávajú jednotlivé vrstvy modelu?
1. **Nultá vrstva (Char-CNN / Vstupné embeddingy):**
   * Vstup spracováva konvolučná sieť nad znakmi (Character-level CNN).
   * Zachytáva **morfologické vlastnosti** (predpony, prípony, korene slov) a je robustná voči preklepom a OOV slovám.
2. **Prvá vrstva BiLSTM:**
   * Špecializuje sa na **syntaktické informácie** (slovné druhy, vetné členy, závislosti a gramatické väzby).
3. **Druhá (vyššia) vrstva BiLSTM:**
   * Špecializuje sa na **vysokoúrovňovú sémantiku a kontext**, rozlišuje významy polysémnych slov a zachytáva dlhé závislosti vo vete.

### 2.3 Kombinácia reprezentácií z jednotlivých vrstiev
ELMo nekombinuje vrstvy obyčajným výberom poslednej vrstvy. Namiesto toho vytvára úlohe-špecifickú (task-specific) lineárnu kombináciu:

$$\mathbf{ELMo}_k^{\text{task}} = E(R_k; \Theta^{\text{task}}) = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} \mathbf{h}_{k,j}^{LM}$$

* $L$ je počet vrstiev BiLSTM (zvyčajne $L=2$, spolu s nultou vrstvou sú to 3 reprezentácie).
* $s_j^{\text{task}}$ sú softvérovo normalizované váhy (cez Softmax, teda $\sum s_j = 1$), ktoré sa učia počas trénovania konkrétnej cieľovej úlohy (napr. klasifikácia, QA).
* $\gamma^{\text{task}}$ je škálovací parameter, ktorý umožňuje cieľovému modelu upraviť celkovú magnitúdu ELMo vektora.

---

## 3. Transformer architektúra - Základné princípy

Transformer (Vaswani et al., 2017: *„Attention Is All You Need“*) úplne odstránil rekurenciu (RNN/LSTM) aj konvolúciu (CNN) a nahradil ich mechanizmom pozornosti (**Self-Attention**).

### 3.1 Základné komponenty Transformer bloku

Každý transformer blok sa skladá z dvoch hlavných podvrstiev:

#### A. Mechanizmus pozornosti (Multi-Head Self-Attention)
1. **Scaled Dot-Product Attention:**
   Pre vstupnú maticu $X$ sa vypočítajú projekcie $Q$ (Query), $K$ (Key), $V$ (Value):
   $$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$
   Pozornosť sa počíta vzorcom:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   Faktor $\frac{1}{\sqrt{d_k}}$ zabraňuje tomu, aby skalárny súčin rástol do extrémnych hodnôt pri vysokých dimenziách, čo by viedlo k saturácii softmax funkcie s miznúcimi gradientmi.

2. **Multi-Head Attention:**
   Namiesto jedinej pozornosti sa projekcie rozdelia do $h$ hláv. Každá hlava sa môže zamerať na iný typ vzťahu (napr. jedna na zhodu prísudku s podmetom, iná na zámenné odkazy):
   $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

#### B. Dopredná neurónová sieť (Position-wise Feed-Forward Network - FFN)
Za pozornosťou nasleduje plne prepojená dvojvrstvová sieť aplikovaná na každú pozíciu nezávisle a identicky:
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2 \quad (\text{alebo s aktiváciou GELU})$$
Zvyčajne vnútorná vrstva FFN expanduje dimenziu na $4 \times d_{\text{model}}$ a následne ju vracia na $d_{\text{model}}$.

Okolo oboch podvrstiev sú **reziduálne spojenia (Residual Connections)** s vrstvovou normalizáciou (**LayerNorm**):
$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

---

### 3.2 Enkóder vs. Dekóder v pôvodnej architektúre

| Komponent | Enkóder (Encoder) | Dekóder (Decoder) |
| :--- | :--- | :--- |
| **Účel** | Porozumenie vstupnému textu a extrakcia kontextových príznakov | Autoregresívne generovanie výstupnej sekvencie token po tokene |
| **Typ Attention** | **Plný obojsmerný (Bidirectional) Self-Attention** — každý token vidí všetky tokeny vľavo aj vpravo | 1. **Maskovaný Self-Attention (Causal)** — bráni pohľadu do budúcnosti.<br>2. **Cross-Attention (Encoder-Decoder Attention)** — Query prichádza z dekódera, Key a Value z enkódera. |
| **Vstupy** | Celá zdrojová sekvencia (napr. celá veta v cudzom jazyku pri preklade) + pozičné kódovanie | Doteraz vygenerované cieľové tokeny (posunuté o jedna doprava: *Shifted Right*) |
| **Výstupy** | Matica skrytých stavov reprezentujúca celý kontext pre každú pozíciu | Pravdepodobnostné rozdelenie nad slovníkom pre nasledujúci token |

---

### 3.3 Prečo moderné modely využívajú iba enkóder alebo iba dekóder?

Pôvodný Transformer bol navrhnutý pre úlohu **Sequence-to-Sequence (strojový preklad)**, kde je potrebný aj enkóder (prečítať vetu v jazyku A) aj dekóder (vygenerovať vetu v jazyku B). Moderné NLP zistilo, že mnohé úlohy vyžadujú iba jednu polovicu:

1. **Iba Enkóder (Encoder-only, napr. BERT, RoBERTa, DeBERTa):**
   * **Prečo:** Pre úlohy porozumenia textu (klasifikácia textu, Named Entity Recognition, analýza sentimentu, extrakcia informácií) nepotrebujeme generovať nový text. Potrebujeme maximálne bohatú reprezentáciu, kde každý token môže neobmedzene interagovať s celým kontextom zľava aj sprava.
2. **Iba Dekóder (Decoder-only, napr. GPT-3/4, LLaMA, Mistral):**
   * **Prečo:** Ukázalo sa, že zjednotenie všetkých NLP úloh pod **autoregresívne generovanie textu** (prompting, in-context learning, chat) je mimoriadne škálovateľné. Dekóder dokáže riešiť ako generovanie, tak aj klasifikáciu (stačí vygenerovať odpoveď).
3. **Enkóder-Dekóder (Encoder-Decoder, napr. T5, BART):**
   * Používa sa stále pri úlohách, kde sa transformuje dlhý vstup na iný výstup (abstraktívna sumarizácia, formátovanie kódu, preklad).

---

## 4. Enkóderové a dekóderové jazykové modely

### 4.1 Obojsmerný vs. Maskovaný (Kauzálny) Attention
* **Obojsmerný (Bidirectional) Attention:** Matica pozornosti je plná. Pre pozíciu $i$ je váha $\alpha_{i, j} > 0$ pre všetky $j \in \{1, \dots, n\}$.
* **Maskovaný (Causal) Attention:** Na horný trojuholník matice pozornosti sa aplikuje maska $-\infty$ pred softmaxom:
  $$M_{i, j} = \begin{cases} 0 & \text{pre } j \le i \\ -\infty & \text{pre } j > i \end{cases}$$
  To zabezpečuje, že token na pozícii $i$ môže venovať pozornosť iba tokenom na pozíciách $\le i$. Tým sa zachováva princíp kauzality pri generovaní textu.

---

### 4.2 Princípy predtrénovaných úloh

1. **Masked Language Modeling (MLM - napr. BERT):**
   * Zo vstupnej sekvencie sa náhodne zamaskuje $15\%$ tokenov (pomocou špeciálneho tokenu `[MASK]`, náhodného slova alebo ponechania pôvodného slova).
   * Úlohou modelu je predikovať pôvodné slovo na základe obojsmerného kontextu.
   * Stratová funkcia: krížová entropia nad maskovanými pozíciami.
2. **Next Sentence Prediction (NSP - napr. BERT):**
   * Model dostane dvojicu viet $[A, B]$. V $50\%$ prípadov veta $B$ skutočne nasleduje za vetou $A$, v $50\%$ prípadoch ide o náhodnú vetu z korpusu.
   * Model sa učí binárnu klasifikáciu (`IsNext` vs. `NotNext`), čo pomáha pri úlohách vyžadujúcich pochopenie vzťahov medzi vetami (QA, NLI).
3. **Next Token Prediction (NTP / Autoregressive LM - napr. GPT):**
   * Klasický jazykový model, ktorý predikuje pravdepodobnosť nasledujúceho tokenu:
     $$\mathcal{L}_{\text{NTP}} = -\sum_{i=1}^{T} \log P(w_i \mid w_1, \dots, w_{i-1})$$
   * Trénuje sa na obrovských nestruktúrovaných korpusoch bez potreby ručného maskovania.

---

### 4.3 Porovnanie výhod a obmedzení

| Typ modelu | Výhody | Obmedzenia |
| :--- | :--- | :--- |
| **Enkóderové (BERT)** | • Plná obojsmernosť poskytuje najlepšie reprezentácie pre klasifikáciu a extrakciu.<br>• Každý token má prístup k budúcim informáciám. | • Nevhodné na plynulé generovanie dlhého textu (autoregresívne generovanie nie je prirodzené).<br>• Nesúlad medzi trénovaním (kde je `[MASK]`) a inferenciou (kde `[MASK]` nie je). |
| **Dekóderové (GPT)** | • Prirodzené generovanie koherentného textu ľubovoľnej dĺžky.<br>• Univerzálny prístup (few-shot prompting, instrukčné ladenie). | • Pri kódovaní tokenu na pozícii $i$ nemôže vidieť pravý kontext (menej optimálne pre úlohy ako NER alebo sentence similarity). |

---

## 5. Praktická časť: Analýza kosínusovej podobnosti kontextových embeddingov

V praktickej časti bol použitý skript [`task.py`](./task.py), ktorý využíva modely `bert-base-multilingual-cased` (pre slovenčinu) a `bert-base-uncased` (pre angličtinu).

### 5.1 Testované vety a polysémne slová

#### Príklad 1: Slovenské slovo **„zámok“**
* **Veta 1 (Architektúra / hrad):** *„Starobylý zámok sa týči na vysokom skalnatom kopci.“*
* **Veta 2 (Architektúra / hrad):** *„Kráľovský zámok a jeho nádvorie navštevujú tisíce turistov.“*
* **Veta 3 (Mechanizmus / dvere):** *„Zámočník rýchlo vymenil poškodený zámok na vchodových dverách.“*
* **Veta 4 (Mechanizmus / visiaci):** *„Na bráne visel ťažký oceľový zámok proti vlámaniu.“*

#### Príklad 2: Anglické slovo **„bank“**
* **Veta 1 (Financie):** *„I deposited all my savings into a commercial bank account.“*
* **Veta 2 (Financie):** *„The bank approved our mortgage loan with low interest rates.“*
* **Veta 3 (Príroda / rieka):** *„We sat on the muddy river bank fishing until the sunset.“*
* **Veta 4 (Príroda / rieka):** *„During the flood, the water overflowed the river bank.“*

### 5.2 Získané výsledky kosínusovej podobnosti

#### A. Výsledky pre slovenské slovo „zámok“ (`bert-base-multilingual-cased`):
| Veta | V1 (architektúra) | V2 (architektúra) | V3 (mechanizmus) | V4 (mechanizmus) |
| :--- | :---: | :---: | :---: | :---: |
| **V1 (architektúra)** | **1.0000** | **0.7385** | 0.7688 | 0.7238 |
| **V2 (architektúra)** | **0.7385** | **1.0000** | 0.7274 | 0.7589 |
| **V3 (mechanizmus)** | 0.7688 | 0.7274 | **1.0000** | **0.8207** |
| **V4 (mechanizmus)** | 0.7238 | 0.7589 | **0.8207** | **1.0000** |

*Najvyššia podobnosť v mechanickom zmysle bola dosiahnutá medzi vetami V3 a V4 ($0.8207$).*

#### B. Výsledky pre anglické slovo „bank“ (`bert-base-uncased`):
| Veta | V1 (finances) | V2 (finances) | V3 (river edge) | V4 (river edge) |
| :--- | :---: | :---: | :---: | :---: |
| **V1 (finances)** | **1.0000** | **0.7041** | 0.4651 | 0.5112 |
| **V2 (finances)** | **0.7041** | **1.0000** | 0.4161 | 0.4856 |
| **V3 (river edge)** | 0.4651 | 0.4161 | **1.0000** | **0.6629** |
| **V4 (river edge)** | 0.5112 | 0.4856 | **0.6629** | **1.0000** |

*Pri anglickom BERT modeli vidíme ešte dramatickejší sémantický rozdiel: vety s finančným významom majú medzi sebou podobnosť **0.7041**, vety s brehom rieky **0.6629**, no krížová podobnosť medzi financiami a riekou klesá až na **0.4161**.*

#### Porovnanie so statickými embeddingmi:
Pri **statických embeddingoch (Word2Vec / FastText)** má slovo vždy iba jeden vektor bez ohľadu na vetu. Kosínusová podobnosť slova so sebou samým by bola vo všetkých prípadoch striktne **1.0000**. Transformer dokázal na základe pozornosti (self-attention) prispôsobiť vektor konkrétnemu kontextu.

Grafické znázornenie matíc podobnosti je vygenerované v grafoch:
* [`similarity_matrix_sk.png`](./similarity_matrix_sk.png) (slovenský príklad)
* [`similarity_matrix_en.png`](./similarity_matrix_en.png) (anglický príklad)

# Human Agreement Study (`src/agreement.py`)

### 1. Overall Inter-Annotator Agreement — All Items (n = 402)

- **Pairwise Overall κ:** Tuğra–İbrahim `0.7052` · Tuğra–Mahmut `0.7740` · İbrahim–Mahmut `0.6481` → **Mean κ = `0.7091`**
- **Consensus:** 3/3 Unanimous `285` (70.9%) · 2/3 Majority `92` (22.9%) · 1/1/1 Disagreement `25` (6.2%)

| Label | Tuğra–İbrahim κ | Tuğra–Mahmut κ | İbrahim–Mahmut κ | **Mean κ** | Counts (Tuğra / İbrahim / Mahmut / Maj / Ref) |
|---|---|---|---|---|---|
| **`TR`** | 0.8097 | 0.8131 | 0.7553 | **0.7927** | 162 / 140 / 158 / **160** / `121` |
| **`EN`** | 0.6829 | 0.8656 | 0.6631 | **0.7372** | 93 / 138 / 90 / **99** / `89` |
| **`MIXED`** | 0.7973 | 0.8816 | 0.7896 | **0.8228** | 76 / 84 / 68 / **75** / `75` |
| **`NE`** | 0.4865 | 0.6481 | 0.3142 | **0.4829** | 41 / 17 / 77 / **43** / `80` |
| **`AMBIGUOUS`** | 0.2100 | -0.0128 | -0.0127 | **0.0615** | 17 / 16 / 3 / **13** / `30` |
| **`OTHER`** | 0.6931 | 0.5164 | 0.7655 | **0.6583** | 13 / 7 / 6 / **12** / `7` |

### 2. Inter-Annotator Agreement — Real TurEngMix Items Only (n = 202)

- **Pairwise Overall κ:** Tuğra–İbrahim `0.6909` · Tuğra–Mahmut `0.7263` · İbrahim–Mahmut `0.5770` → **Mean κ = `0.6647`**
- **Consensus:** 3/3 Unanimous `135` (66.8%) · 2/3 Majority `53` (26.2%) · 1/1/1 Disagreement `14` (6.9%)

| Label | Tuğra–İbrahim κ | Tuğra–Mahmut κ | İbrahim–Mahmut κ | **Mean κ** | Counts (Tuğra / İbrahim / Mahmut / Maj / Ref) |
|---|---|---|---|---|---|
| **`TR`** | 0.7703 | 0.7856 | 0.7172 | **0.7577** | 84 / 70 / 83 / **83** / `60` |
| **`EN`** | 0.7226 | 0.8604 | 0.6379 | **0.7403** | 57 / 76 / 50 / **57** / `50` |
| **`MIXED`** | 0.7493 | 0.8236 | 0.6949 | **0.7559** | 28 / 33 / 24 / **29** / `30` |
| **`NE`** | 0.4527 | 0.5313 | 0.2344 | **0.4061** | 18 / 7 / 43 / **19** / `40` |
| **`AMBIGUOUS`** | 0.3283 | -0.0090 | -0.0093 | **0.1033** | 9 / 13 / 1 / **8** / `20` |
| **`OTHER`** | 0.6599 | 0.2796 | 0.4963 | **0.4786** | 6 / 3 / 1 / **6** / `2` |

### 3. Inter-Annotator Agreement — Generated (Synthetic) Items Only (n = 200)

- **Pairwise Overall κ:** Tuğra–İbrahim `0.7160` · Tuğra–Mahmut `0.8186` · İbrahim–Mahmut `0.7168` → **Mean κ = `0.7505`**
- **Consensus:** 3/3 Unanimous `150` (75.0%) · 2/3 Majority `39` (19.5%) · 1/1/1 Disagreement `11` (5.5%)

| Label | Tuğra–İbrahim κ | Tuğra–Mahmut κ | İbrahim–Mahmut κ | **Mean κ** | Counts (Tuğra / İbrahim / Mahmut / Maj / Ref) |
|---|---|---|---|---|---|
| **`TR`** | 0.8501 | 0.8413 | 0.7946 | **0.8287** | 78 / 70 / 75 / **77** / `61` |
| **`EN`** | 0.6300 | 0.8701 | 0.6891 | **0.7297** | 36 / 62 / 40 / **42** / `39` |
| **`MIXED`** | 0.8256 | 0.9153 | 0.8484 | **0.8631** | 48 / 51 / 44 / **46** / `45` |
| **`NE`** | 0.5114 | 0.7763 | 0.4089 | **0.5655** | 23 / 10 / 34 / **24** / `40` |
| **`AMBIGUOUS`** | -0.0223 | -0.0163 | -0.0121 | **-0.0169** | 8 / 3 / 2 / **5** / `10` |
| **`OTHER`** | 0.7201 | 0.6567 | 0.8864 | **0.7544** | 7 / 4 / 5 / **6** / `5` |

## 4. Annotators vs. TurEngMix Gold on Real Items (Measured Human Ceiling)

| Annotator | Cohen's Kappa (κ) | Macro-F1 (Human Ceiling) | Accuracy | MIXED F1 | NE F1 |
|---|---|---|---|---|---|
| **Tugra** | `0.6629` | **`0.6558`** | `0.7426` | `0.7931` | `0.5172` |
| **Ibrahim** | `0.5672` | **`0.6136`** | `0.6683` | `0.6667` | `0.1702` |
| **Mahmut** | `0.6727` | **`0.6414`** | `0.7525` | `0.8519` | `0.6988` |
| **Majority** | `0.6630` | **`0.6497`** | `0.7426` | `0.8136` | `0.5085` |

## 5. Annotators & Majority Vote vs. Our Labels on Generated Items (Synthetic Data QA)

| Annotator | Cohen's Kappa (κ) | Macro-F1 | Accuracy | MIXED F1 | NE F1 |
|---|---|---|---|---|---|
| **Tugra** | `0.7389` | **`0.7038`** | `0.8000` | `0.9247` | `0.6349` |
| **Ibrahim** | `0.6990` | **`0.6859`** | `0.7700` | `0.8958` | `0.4000` |
| **Mahmut** | `0.8163` | **`0.7743`** | `0.8600` | `0.9663` | `0.7297` |
| **Majority** | `0.7840` | **`0.7474`** | `0.8350` | `0.9670` | `0.6562` |

## 6. All 1/1/1 Total Disagreements (Edge Cases)

| Row | Source | Target Word | Gold/Ref | Tuğra | İbrahim | Mahmut | Context Snippet |
|---|---|---|---|---|---|---|---|
| 7 | `real` | **`pardon`** | `AMBIGUOUS` | `OTHER` | `AMBIGUOUS` | `TR` | ...opluluğu hayir yani mantik ne [[pardon]] tanittigi ruju ojeyi alligi ojeyi vs gi... |
| 20 | `real` | **`"x`** | `AMBIGUOUS` | `AMBIGUOUS` | `EN` | `NE` | ...grad student" hadi onu gectim [["x]] university faculty" secmisler kendileri ici... |
| 26 | `real` | **`swiftuı`** | `NE` | `AMBIGUOUS` | `EN` | `NE` | ...sean allen'ın link bu robotla [[swiftuı]] ile adım adım uygulama geliştiriyor ek... |
| 48 | `generated` | **`final`** | `TR` | `OTHER` | `EN` | `TR` | [[final]] haftası geldi ve ben hala dersin slide'larını açmadım |
| 58 | `real` | **`amerika`** | `NE` | `TR` | `AMBIGUOUS` | `NE` | ...'i geçmesine rağmen) bunların [[amerika]] numarasını aradım, pozitif test sonucu... |
| 63 | `real` | **`mangasından`** | `TR` | `TR` | `MIXED` | `NE` | ...inen manga tabi bazen animesi [[mangasından]] iyi olan serilerde oluyor verebile... |
| 83 | `real` | **`anime`** | `AMBIGUOUS` | `TR` | `EN` | `NE` | ...m uzun soluklu serileri zaten [[anime]] olarak önermem asla direkt mangasını oku... |
| 108 | `generated` | **`protein`** | `AMBIGUOUS` | `OTHER` | `AMBIGUOUS` | `TR` | spor sonrası [[protein]] barı yemek benim için bir ödül, tadı kartondan farksız ... |
| 126 | `generated` | **`meksika'dan`** | `NE` | `TR` | `MIXED` | `NE` | [[meksika'dan]] calisirken sabah altida standup'a giriyorum, istanbul'daki ekip ... |
| 132 | `generated` | **`lego`** | `NE` | `AMBIGUOUS` | `EN` | `TR` | [[lego]] duplo parçaları evin her yerinde, gece yürürken ayağıma batınca çığlık ... |
| 154 | `generated` | **`sim`** | `EN` | `TR` | `EN` | `NE` | yeni [[sim]] karti aktive etmek icin magazaya gittim, sistem down oldugu icin ya... |
| 176 | `generated` | **`model`** | `AMBIGUOUS` | `AMBIGUOUS` | `EN` | `TR` | [[model]] uçak koleksiyonuma yeni bir boeing ekledim, kutusundan çıkarmaya kıyam... |
| 213 | `real` | **`yeaaa"`** | `AMBIGUOUS` | `OTHER` | `EN` | `TR` | ...der yazınca çok havalı oluyor [[yeaaa"]] mantığıyla başlatılmış ve haliyle babay... |
| 225 | `real` | **`filler`** | `EN` | `TR` | `MIXED` | `EN` | ...di bu seri artık bittiği için [[filler]] bölümler bizi ilgelendirmiyor artık bu ... |
| 234 | `generated` | **`manga`** | `AMBIGUOUS` | `TR` | `EN` | `NE` | [[manga]] kafesinde saatlik ücret ödeyip istediğin kadar okuyabiliyorsun, harika... |
| 239 | `generated` | **`anime`** | `AMBIGUOUS` | `TR` | `EN` | `NE` | [[anime]] festivalinde yediğim ramen gerçek japon ramenine en yakın şeydi |
| 242 | `real` | **`site`** | `AMBIGUOUS` | `AMBIGUOUS` | `EN` | `TR` | ...ak heyecanla izleyebildiğimiz [[site]] is went from being in a relationship with... |
| 245 | `real` | **`lit"e`** | `MIXED` | `AMBIGUOUS` | `EN` | `NE` | ...n "gidersin nature is fucking [[lit"e]] bir şey mi sorucaksın git "askreddit" ,d... |
| 257 | `generated` | **`indie`** | `EN` | `AMBIGUOUS` | `EN` | `NE` | [[indie]] grupların istanbul konserleri küçük barlarda oluyor, en samimi geceler... |
| 258 | `generated` | **`model`** | `AMBIGUOUS` | `AMBIGUOUS` | `EN` | `TR` | [[model]] gibi poz vermek için bir fotoğrafçıyla streetwear çekimi yaptık, sonuç... |
| 266 | `generated` | **`model`** | `AMBIGUOUS` | `AMBIGUOUS` | `EN` | `TR` | bu [[model]] telefonun kamerasi iyi ama batarya kotu, samsung kullanicilari hep ... |
| 288 | `real` | **`turntablea`** | `MIXED` | `MIXED` | `EN` | `NE` | [[turntablea]] ucundan kıyısından dokunup sikko mixler yapan hemen afilli takma ... |
| 344 | `real` | **`ibrahimoviç'le`** | `NE` | `TR` | `MIXED` | `NE` | ronaldo'yla messi'yle değil [[ibrahimoviç'le]] ronaldo'yla the original zidane'l... |
| 366 | `real` | **`"x`** | `AMBIGUOUS` | `OTHER` | `EN` | `NE` | ...an ote olamayacak genc tipler [["x]] university grad student" hadi onu gectim "x... |
| 371 | `real` | **`linke`** | `MIXED` | `MIXED` | `TR` | `AMBIGUOUS` | ...hex'e çevirin hex değerini bu [[linke]] yapıştırın elde ettiğiniz adrese para gö... |

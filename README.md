# cv-final — OSRS NPC Computer Vision

Computer vision project on NPC data scraped from the [OSRS Wiki](https://oldschool.runescape.wiki): 4,443 NPCs, each with a chathead portrait, a full-body image, and infobox metadata (class, gender, location, members status, release date).

Two problems, both framed as **champion vs. challenger** — a pretrained/transfer-learning model against one trained from scratch, head-to-head on identical data splits:

- **Problem 1 — Classification.** Predict NPC class (Human, Dwarf, Elf, Gnome, Vampyre, Ghost, Monkey, Dorgeshuun, Troll, Cat, or Other) from chathead + body images. Champion: two pretrained ResNet18s. Challenger: two from-scratch CNNs. Champion: 84.4% test accuracy. Challenger: 61.9%.
- **Problem 2 — Generation.** Generate new NPC chathead art conditioned on class. Champion: LoRA fine-tune of Stable Diffusion 1.5. Challenger: a conditional DCGAN trained from scratch. Champion: FID 234.95, CLIP-score 0.282. Challenger: FID 340.18, CLIP-score 0.242.

A third track uses CLIP directly: zero-shot attribute tagging (beard, headwear, eyewear, etc.) and a **visual RAG search** — embed all 3,468 NPC chatheads + bodies through CLIP's image encoder, average the pair into one 512-dim vector per NPC (`combined_embeddings.npy`), then at query time encode a text description (e.g. "old man with white beard") and return the top-k NPCs by cosine similarity, images and all.

---

## Project structure

```
cv-final/
├── data/
│   ├── npc.jsonl, npc.csv                  # scraped dataset (source of truth + built CSV)
│   ├── npc_pages.csv, npc_hrefs_flat.csv   # scrape checkpoints
│   ├── chatheads/{id}.png, bodies/{id}.png # raw scraped images
│   ├── chatheads_processed/{id}.png        # deduped, resized 128x128 set used for generation
│   │   └── metadata.jsonl                  # final caption manifest for LoRA training
│   ├── npc_attributes.csv                  # CLIP zero-shot attribute labels
│   ├── combined_embeddings.npy, embedding_ids.npy       # chathead+body CLIP embeddings (search/t-SNE)
│   ├── chathead_embeddings.npy, body_embeddings.npy     # per-image-type CLIP embeddings (+ id files)
│   ├── chathead_lora_manifest.csv          # image+caption manifest used to train the LoRA champion
│   ├── chathead_near_duplicates.csv        # near-duplicate pairs found via CLIP similarity
│   ├── chathead_url_mismatches.csv         # name-vs-filename audit from EDA
│   ├── flagged_candidates.csv, flagged_junk_images.csv, review_progress.csv  # junk-image QA pipeline
│   └── missing_processed_chatheads.csv     # validation check (empty — nothing missing)
├── img/                 # ~30 saved plots (EDA, clustering, training curves, confusion matrices,
│                         #   Grad-CAM, GAN/LoRA samples, classification & generation comparisons)
├── python/
│   ├── 01_web_scrape_osrs_wiki.ipynb       # scraping
│   ├── 02_eda.ipynb                        # EDA
│   ├── flag_junk_images.ipynb              # CLIP-based junk-image detection
│   ├── review_app.py                       # Streamlit human review UI
│   ├── delete_flagged_images.ipynb         # deletes confirmed junk
│   ├── 03_train.ipynb, 04_eval.ipynb       # early/draft classifier run — superseded, see below
│   ├── npc_data.py, eval_utils.py          # shared data-split / eval code (classification)
│   ├── 03a_train_champion_local.ipynb      # classification champion: pretrained ResNet18 x2
│   ├── 03b_train_custom_cnn.ipynb          # classification challenger: from-scratch CNN x2
│   ├── 04a_eval_champion_local.ipynb
│   ├── 04b_eval_custom_cnn.ipynb
│   ├── 04c_classification_comparison.ipynb # champion vs. challenger head-to-head
│   ├── 05_clip_attributes.ipynb            # zero-shot CLIP attribute tagging
│   ├── clip_test.ipynb                     # CLIP visual search + t-SNE data prep
│   ├── 06_lora_feasibility.ipynb           # dedup, HDBSCAN validation, Human-only LoRA MVP
│   ├── 07_train_lora_champion.ipynb        # generation champion: LoRA on Stable Diffusion 1.5
│   ├── 08_train_gan_challenger.ipynb       # generation challenger: conditional DCGAN
│   ├── 09_eval_image_generation.ipynb      # champion vs. challenger head-to-head
│   ├── train_text_to_image_lora.py         # vendored HuggingFace diffusers script (dependency)
│   └── demo_app.py                         # Streamlit demo: both problems, champion vs. challenger
├── docs/
│   └── model_operations.md   # deployment architecture + retraining/maintenance plan
├── tsne-viz/                 # interactive D3.js t-SNE viewer over CLIP embedding space
│   ├── index.html, style.css, script.js
│   └── tsne_data.json        # built by clip_test.ipynb
└── presentation/
    └── OSRS_NPC_CV_Final.pptx
```

Trained model checkpoints (`.pth` files, `lora_champion_output/`) are not checked into this repo.

---

## Data

### `data/npc.csv` — main dataset
One row per NPC (4,443 rows). Key columns:

| Column | Description |
|---|---|
| `id` | Sequential integer; matches image filenames in `chatheads/` and `bodies/` |
| `Name` | NPC name |
| `Class` | Renamed from the wiki's `Race` field; some values collapsed (e.g. Citizen of Arceuus → Human) |
| `Gender` | Collapsed to Male / Female / Other |
| `Location` | Location from infobox |
| `Members` | Whether NPC is members-only |
| `Release` | Release date from infobox |
| `chathead_url` / `body_url` | Source image URLs on the wiki |
| `has_chathead` | Boolean; whether a usable chathead image exists |

`data/npc.jsonl` is the source of truth (one JSON object per NPC, written incrementally during scraping so a crash mid-run doesn't lose progress); `npc.csv` is built from it. `npc_pages.csv` / `npc_hrefs_flat.csv` are scrape checkpoints, only needed to resume a crawl without re-hitting the category pages.

### Images
- `data/chatheads/{id}.png`, `data/bodies/{id}.png` — raw scraped images. Not every NPC has both; `has_chathead` flags availability.
- `data/chatheads_processed/` — cleaned, deduplicated, resized/padded (128×128) chathead set used for generation training. 1,712 images across 11 classes, built by `06_lora_feasibility.ipynb`.

### Data-quality pipeline
Raw scraping picked up ~130 wrong images (chathead/body pulled from a wiki navbox instead of the NPC's own page) and the game itself has many near-duplicate NPCs (e.g. dozens of identical "Banker" characters). Two passes handle this:

- **Junk detection** (`flag_junk_images.ipynb` → `review_app.py` → `delete_flagged_images.ipynb`): CLIP zero-shot classification flags likely-wrong images, a Streamlit app supports one-by-one human review, confirmed junk gets deleted. Outputs: `flagged_candidates.csv`, `review_progress.csv`, `flagged_junk_images.csv`.
- **Near-duplicate removal** (`06_lora_feasibility.ipynb`): CLIP cosine similarity (≥0.98) + connected-components clustering found 381 duplicate clusters (9,390 pairs) and kept one representative per cluster. Output: `chathead_near_duplicates.csv`.

### Other data files
- `npc_attributes.csv` — zero-shot CLIP attribute labels per NPC (`05_clip_attributes.ipynb`)
- `combined_embeddings.npy` / `embedding_ids.npy` — CLIP embeddings (chathead+body averaged) used for search and t-SNE, shape (3468, 512)
- `chathead_embeddings.npy`, `body_embeddings.npy` (+ id files) — CLIP embeddings per image type
- `chathead_lora_manifest.csv` — final image + caption manifest used to train the LoRA champion
- `chathead_url_mismatches.csv` — name-vs-filename audit from EDA
- `missing_processed_chatheads.csv` — validation check (currently empty; nothing missing)

---

## Notebooks & scripts (`python/`)

Numbered where there's a pipeline order; unnumbered files are shared utilities or standalone tools.

| File | Purpose |
|---|---|
| `01_web_scrape_osrs_wiki.ipynb` | Crawls the NPC category, scrapes infobox fields, downloads images. Writes `npc.jsonl` + `npc.csv`. |
| `02_eda.ipynb` | Field coverage, univariate/bivariate distributions, image dimension checks, chathead grid samples, name/filename consistency audit. |
| `flag_junk_images.ipynb`, `delete_flagged_images.ipynb`, `review_app.py` | Junk-image detection → human review → deletion (see Data-quality pipeline above). |
| `03_train.ipynb`, `04_eval.ipynb` | Early/draft classification run (Colab). Superseded by `03a`/`04a` below — kept for reference, don't cite its accuracy number (0.862; the saved, current champion scores 0.844). |
| `npc_data.py`, `eval_utils.py` | Shared data-loading/splitting and evaluation code, so both classification models train and get evaluated on byte-identical splits. |
| `03a_train_champion_local.ipynb` | Trains the classification champion: two pretrained ResNet18 backbones, fused, fine-tuned. Saves `npc_classifier.pth`. |
| `03b_train_custom_cnn.ipynb` | Trains the classification challenger: two from-scratch CNNs, same fusion head. Saves `custom_cnn_classifier.pth`. |
| `04a_eval_champion_local.ipynb`, `04b_eval_custom_cnn.ipynb` | Per-model eval: classification report, confusion matrix, Grad-CAM, t-SNE. |
| `04c_classification_comparison.ipynb` | Head-to-head: accuracy/F1/precision/recall, per-class F1 gap, ROC/AUC, parameter count/checkpoint size/inference latency. |
| `05_clip_attributes.ipynb` | Zero-shot CLIP attribute tagging (facial hair, headwear, eyewear, cape, wings, hair color — 6 of 11 tried attributes proved reliable enough to keep). Writes `npc_attributes.csv`. |
| `clip_test.ipynb` | Visual RAG search: embeds all chatheads+bodies via CLIP, averages into one vector/NPC (`combined_embeddings.npy`), then answers text or image queries via cosine similarity, returning top-k matching NPCs. Also builds the sampled `tsne_data.json` used by `tsne-viz/`. |
| `06_lora_feasibility.ipynb` | Chathead/body resolution EDA, CLIP-based near-duplicate removal, HDBSCAN cluster validation, final processed-image manifest, and a Human-only LoRA proof-of-concept. |
| `07_train_lora_champion.ipynb` | Trains the generation champion: LoRA fine-tune of Stable Diffusion 1.5 on all 11 classes, using the manifest from `06`. |
| `08_train_gan_challenger.ipynb` | Trains the generation challenger: a class-conditional DCGAN from scratch. |
| `09_eval_image_generation.ipynb` | Head-to-head generation eval: FID, CLIP-score, and a classifier-agreement check (documented as confounded — see notebook for why). |
| `train_text_to_image_lora.py` | Vendored HuggingFace `diffusers` LoRA training script (dependency, not team-authored). |
| `demo_app.py` | Streamlit demo: classify (champion vs. challenger side by side) and generate (LoRA vs. DCGAN side by side). |

---

## `docs/model_operations.md`

Deployment architecture (single API with `/classify` and `/generate` routes, champion+challenger loaded side by side) and a maintenance plan: re-scrape cadence, the same human-review pipeline for QA'ing new data, a promotion rule (a retrained model only replaces production if it beats the current champion on a fresh held-out split), versioned rollback, and monitoring signals (prediction drift, periodic FID/CLIP-score spot checks).

---

## `tsne-viz/`

Standalone D3.js visualization (no build step — just open `index.html`) of ~800 NPCs positioned by t-SNE over their CLIP embeddings. Each node is the NPC's actual chathead image; hover for name/class/id, pan and zoom, force simulation keeps nodes from overlapping. Data comes from `tsne_data.json`, built in `clip_test.ipynb`.

---

## Results summary

**Classification** (test set, n=519):

| | Champion (ResNet18 ×2) | Challenger (CNN ×2, scratch) |
|---|---|---|
| Accuracy | 0.844 | 0.619 |
| Macro F1 | 0.845 | 0.473 |
| Params | 22.6M | 3.4M |
| Checkpoint | 86.4 MB | 13.0 MB |
| Latency (GPU) | 1.26 ms/img | 0.95 ms/img |

The champion's edge is concentrated in the low-data classes (e.g. +0.70 F1 on Monkey, n=6) and nearly disappears on the best-represented class (+0.13 F1 on Human, n=314) — the expected shape of a transfer-learning benefit.

**Generation** (~300 generated images/model, 11 classes):

| | Champion (LoRA/SD1.5) | Challenger (DCGAN) |
|---|---|---|
| FID (lower better) | 234.95 | 340.18 |
| CLIP-score (higher better) | 0.282 | 0.242 |

Both metrics agree the champion produces more realistic, class-appropriate images. A third metric (classifier agreement) nominally favors the challenger but is documented in `09_eval_image_generation.ipynb` as a methodology artifact, not a genuine quality signal.

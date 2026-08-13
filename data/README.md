# data/

Files below are grouped by pipeline stage. `id` is the join key across all of them —
matches `chatheads/{id}.png`, `bodies/{id}.png`, and rows in `npc.csv`.

## 1. Scraping (01_web_scrape_osrs_wiki.ipynb)
- `npc.jsonl` — source of truth, one JSON object per NPC, written incrementally
- `npc.csv` — built from npc.jsonl, 4,443 rows
- `npc_pages.csv`, `npc_hrefs_flat.csv` — scrape checkpoints (resume without re-crawling)
- `chatheads/{id}.png`, `bodies/{id}.png` — raw scraped images

## 2. Data-quality pass (flag_junk_images.ipynb -> review_app.py -> delete_flagged_images.ipynb)
- `flagged_candidates.csv` — CLIP-flagged likely-wrong images
- `review_progress.csv` — full human keep/remove log
- `flagged_junk_images.csv` — confirmed junk, feeds the delete step
- `chathead_url_mismatches.csv` — name-vs-filename audit (from 02_eda.ipynb)

## 3. CLIP attributes + search (05_clip_attributes.ipynb, clip_test.ipynb)
- `npc_attributes.csv` — zero-shot attribute labels (beard, headwear, etc.)
- `combined_embeddings.npy` / `embedding_ids.npy` — chathead+body CLIP embeddings, shape (3468, 512), powers visual search + t-SNE
- `chathead_embeddings.npy`, `body_embeddings.npy` (+ id files) — per-image-type embeddings

## 4. Generation data prep (06_lora_feasibility.ipynb)
- `chathead_near_duplicates.csv` — CLIP-similarity duplicate pairs found
- `chatheads_processed/{id}.png` — deduped, 128x128 set, 1,712 images / 11 classes
- `chatheads_processed/metadata.jsonl` — final caption manifest
- `chathead_lora_manifest.csv` — manifest actually used to train the LoRA champion
- `missing_processed_chatheads.csv` — validation check, empty (nothing missing)

## Not in this folder
Trained checkpoints (`.pth`, `lora_champion_output/`) aren't committed — see root README.
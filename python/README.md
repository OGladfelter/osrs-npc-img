# python/

Read top to bottom — this is the actual pipeline order.

## Data
1. `01_web_scrape_osrs_wiki.ipynb` — scrape
2. `02_eda.ipynb` — EDA
3. `flag_junk_images.ipynb`, `review_app.py`, `delete_flagged_images.ipynb` — junk-image QA

## Problem 1: Classification
4. `npc_data.py`, `eval_utils.py` — shared split/eval code (both models see identical data)
5. `03a_train_champion_local.ipynb` — champion: pretrained ResNet18 x2
6. `03b_train_custom_cnn.ipynb` — challenger: from-scratch CNN x2
7. `04a_eval_champion_local.ipynb`, `04b_eval_custom_cnn.ipynb` — per-model eval
8. `04c_classification_comparison.ipynb` — **head-to-head, start here if short on time**

## CLIP tools (bridges both problems)
9. `05_clip_attributes.ipynb` — zero-shot attribute tagging
10. `clip_test.ipynb` — visual RAG search + t-SNE data prep

## Problem 2: Generation
11. `06_lora_feasibility.ipynb` — dedup, HDBSCAN validation, Human-only LoRA MVP
12. `07_train_lora_champion.ipynb` — champion: LoRA on Stable Diffusion 1.5
13. `08_train_gan_challenger.ipynb` — challenger: conditional DCGAN
14. `09_eval_image_generation.ipynb` — **head-to-head, start here if short on time**

## Demo
- `demo_app.py` — `streamlit run demo_app.py`, both problems side by side

## Archive (superseded, kept for reference only)
- `03_train.ipynb`, `04_eval.ipynb` — early Colab draft of classification. Current champion is 0.844.
- `train_text_to_image_lora.py` — vendored HuggingFace script, not our code
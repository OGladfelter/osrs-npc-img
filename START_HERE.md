# Start here

Hi Dr. Pujari and Hritik!

This is the shared repo for group 8: Oliver Gladfelter, Camille Javier, and Benjamin Swenson.

Our project is on NPC character models from a video game called OSRS. We scraped 4,443 NPCs from the OSRS Wiki (chathead + body images, class/gender/location metadata), cleaned the data (removed ~130 mis-scraped images, deduped near-identical NPCs), and used it to attack two problems. For each, we built a champion (pretrained / transfer learning) and a challenger
(trained from scratch) and measured the gap.

Two problems:

- 1. Classification.** Predict NPC class from its images. Champion (pretrained ResNet18 x2):
  84.4% test accuracy. Challenger (from-scratch CNN x2): 61.9%.
- 2. Generate new NPC art conditioned on class. Champion (LoRA / Stable Diffusion):
  FID 234.95. Challenger (DCGAN from scratch): FID 340.18.

# Understanding our repo

- Full pipeline order and file map: `python/README.md`, `data/README.md`.
- Interactive extras: `tsne-viz/index.html` (open directly, no server needed), `python/demo_app.py` (`streamlit run demo_app.py`).

## At a glance
1. `python/04c_classification_comparison.ipynb` — Problem 1 head-to-head
2. `python/09_eval_image_generation.ipynb` — Problem 2 head-to-head
3. `tsne-viz/index.html` — open directly in a browser, no server needed

## Links to sub-README markdown files and important python files
- `python/README.md` — every notebook, in pipeline order
- `data/README.md` — every data file, grouped by pipeline stage
- `docs/model_operations.md` — how we'd deploy and maintain this if it were a real product
- `python/demo_app.py` — `streamlit run demo_app.py`, both problems side by side, champion vs. challenger
# cv-final

Computer Vision project using OSRS (Old School RuneScape) NPC data scraped from the [OSRS Wiki](https://oldschool.runescape.wiki).

---

## Project Structure

```
cv-final/
├── data/
│   ├── npc_pages.csv          # 23 category page URLs crawled from the wiki
│   ├── npc_hrefs_flat.csv     # 4,443 NPC page href slugs (e.g. /w/Zygomite)
│   ├── npc.jsonl              # Raw scraped data, one JSON object per NPC per line
│   ├── npc.csv                # Final dataset (built from npc.jsonl); primary data file
│   ├── chatheads/             # Chathead images, named {id}.png
│   └── bodies/                # Full-body images, named {id}.png
├── img/
│   ├── osrs_npc_map.png
│   └── osrs_world_map.jpg
└── python/
    ├── web_scrape_osrs_wiki.ipynb   # Scraper: crawls wiki, downloads images, writes npc.jsonl + npc.csv
    └── 01_eda.ipynb                 # EDA: coverage, univariate/bivariate analysis, image dimension checks
```

---

## Data

### `npc.csv` — main dataset
One row per NPC. Columns include:

| Column | Description |
|---|---|
| `id` | Sequential integer (0–4442); matches image filenames in `chatheads/` and `bodies/` |
| `Name` | NPC name |
| `Race` | Race from infobox --> rename to `Class` in code |
| `Gender` | Gender from infobox |
| `Location` | Location from infobox |
| `Members` | Whether NPC is members-only |
| `Release` | Release date from infobox |
| `chathead_url` | Full URL to chathead image on the wiki |
| `body_url` | Full URL to full-body image on the wiki |
| `has_chathead` | Boolean; whether a chathead image was found |

### `npc.jsonl`
Source of truth. Each line is a JSON object for one NPC, written incrementally during scraping (durable — survives mid-run crashes). Rebuild `npc.csv` from this if needed:
```python
import json, pandas as pd
df = pd.DataFrame([json.loads(line) for line in open("data/npc.jsonl")])
df.to_csv("data/npc.csv", index=False)
```

### `npc_pages.csv` / `npc_hrefs_flat.csv`
Scraping checkpoints. Only needed to re-run or resume scraping from an intermediate step without re-crawling the category pages.

---

## Images

- `data/chatheads/{id}.png` — NPC portrait (chathead)
- `data/bodies/{id}.png` — Full-body NPC image (full resolution source file)

Not every NPC has both images — some wiki pages are missing one or both. `has_chathead` in `npc.csv` flags chathead availability; check `body_url` for body image availability.

---

## Notebooks

### `01_web_scrape_osrs_wiki.ipynb`
Crawls the [Non-player characters category](https://oldschool.runescape.wiki/w/Category:Non-player_characters), collects 4,443 NPC page hrefs, visits each page, scrapes infobox fields, and downloads chathead + body images. Writes `npc.jsonl` incrementally (one line per NPC as it goes) and builds `npc.csv` at the end.

### `02_eda.ipynb`
Exploratory analysis on `npc.csv` and the chathead images. Covers:
- Field coverage and missing value rates
- Univariate distributions (Race, Gender, Members)
- Bivariate heatmaps (Race × Location, Race × Members)
- Image dimension and file size histograms
- Chathead grid samples by race
- Chathead URL consistency check (name vs. filename stem mismatch audit)

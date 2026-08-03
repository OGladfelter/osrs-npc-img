"""Champion vs. challenger demo -- both cognitive problems, side by side.

Run (data paths are resolved relative to this file, so the working directory doesn't matter):
    streamlit run demo_app.py
    streamlit run python/demo_app.py   # also works from the repo root

Two tabs:
- Classify: pick an NPC, run its chathead+body through both classification models
  (champion DualCNN / pretrained ResNet18 x2, challenger ScratchDualCNN / from-scratch CNN x2),
  compare predictions + confidence.
- Generate: pick a class, run both generation models (champion LoRA/Stable Diffusion, challenger
  conditional DCGAN), compare the resulting chathead images.

This is the "deployment mechanism" referenced in docs/model_operations.md made concrete -- a
single process that loads all four model checkpoints and serves both problems interactively,
the same shape a real `/classify` + `/generate` API would take.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "..", "data")
CLASS_NAMES = ["Human", "Dwarf", "Elf", "Gnome", "Vampyre", "Ghost", "Monkey", "Dorgeshuun", "Troll", "Cat", "Other"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="OSRS NPC -- Champion vs. Challenger", page_icon="\U0001F3AE", layout="wide")


# --------------------------------------------------------------------------
# Model definitions (kept self-contained, matching the training/eval notebooks)
# --------------------------------------------------------------------------

class DualCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.chathead_cnn = models.resnet18(weights=None)
        self.body_cnn = models.resnet18(weights=None)
        self.chathead_cnn.fc = nn.Identity()
        self.body_cnn.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, chathead, body):
        return self.classifier(torch.cat([self.chathead_cnn(chathead), self.body_cnn(body)], dim=1))


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True), nn.MaxPool2d(2),
    )


class ScratchCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
            conv_block(128, 256), conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class ScratchDualCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.chathead_cnn = ScratchCNNBackbone()
        self.body_cnn = ScratchCNNBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, chathead, body):
        return self.classifier(torch.cat([self.chathead_cnn(chathead), self.body_cnn(body)], dim=1))


Z_DIM, EMBED_DIM, FEAT_MAPS, IMG_SIZE = 100, 50, 64, 128


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, num_classes=len(CLASS_NAMES), embed_dim=EMBED_DIM, feat_maps=FEAT_MAPS):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        in_dim = z_dim + embed_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, feat_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_maps * 16), nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 16, feat_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 8), nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 8, feat_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 4), nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 4, feat_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 2, feat_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps), nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        embed = self.label_embed(labels)
        x = torch.cat([z, embed], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


# --------------------------------------------------------------------------
# Cached model / data loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading classification models...")
def load_classifiers():
    champion = DualCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    champion.load_state_dict(torch.load(f"{DATA_DIR}/npc_classifier.pth", map_location=DEVICE))
    champion.eval()

    challenger = ScratchDualCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    challenger.load_state_dict(torch.load(f"{DATA_DIR}/custom_cnn_classifier.pth", map_location=DEVICE))
    challenger.eval()
    return champion, challenger


@st.cache_resource(show_spinner="Loading LoRA generation pipeline (this can take a minute)...")
def load_lora_pipeline():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", variant="fp16",
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        safety_checker=None,
    ).to(DEVICE)
    pipe.load_lora_weights(f"{DATA_DIR}/lora_champion_output")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@st.cache_resource(show_spinner="Loading GAN generator...")
def load_gan_generator():
    g = Generator().to(DEVICE)
    g.load_state_dict(torch.load(f"{DATA_DIR}/gan_generator.pth", map_location=DEVICE))
    g.eval()
    return g


@st.cache_data
def load_npc_table():
    df = pd.read_csv(f"{DATA_DIR}/npc.csv")
    df["has_chathead"] = df["id"].apply(lambda i: os.path.exists(f"{DATA_DIR}/chatheads/{i}.png"))
    df["has_body"] = df["id"].apply(lambda i: os.path.exists(f"{DATA_DIR}/bodies/{i}.png"))
    return df[df["has_chathead"] & df["has_body"]].reset_index(drop=True)


eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("OSRS NPC -- Champion vs. Challenger")
st.caption(
    "Problem 1 (classification): pretrained ResNet18 x2 champion vs. from-scratch CNN x2 challenger. "
    "Problem 2 (generation): LoRA/Stable Diffusion champion vs. from-scratch conditional DCGAN challenger."
)

tab_classify, tab_generate = st.tabs(["\U0001F50D Classify", "\U0001F3A8 Generate"])

with tab_classify:
    npc_df = load_npc_table()
    name_to_id = dict(zip(npc_df["Name"], npc_df["id"]))
    chosen_name = st.selectbox("Pick an NPC", sorted(name_to_id.keys()))
    npc_id = name_to_id[chosen_name]

    if st.button("Classify", key="classify_btn"):
        champion, challenger = load_classifiers()

        chathead_img = Image.open(f"{DATA_DIR}/chatheads/{npc_id}.png").convert("RGB")
        body_img = Image.open(f"{DATA_DIR}/bodies/{npc_id}.png").convert("RGB")
        chathead_t = eval_transform(chathead_img).unsqueeze(0).to(DEVICE)
        body_t = eval_transform(body_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            champ_probs = torch.softmax(champion(chathead_t, body_t), dim=1).squeeze().cpu().numpy()
            chal_probs = torch.softmax(challenger(chathead_t, body_t), dim=1).squeeze().cpu().numpy()

        img_col, champ_col, chal_col = st.columns(3)
        with img_col:
            st.image(chathead_img, caption="Chathead", width=160)
            st.image(body_img, caption="Body", width=160)
        with champ_col:
            st.subheader("Champion")
            st.write(f"**{CLASS_NAMES[champ_probs.argmax()]}** ({champ_probs.max():.1%} confidence)")
            st.bar_chart(pd.Series(champ_probs, index=CLASS_NAMES))
        with chal_col:
            st.subheader("Challenger")
            st.write(f"**{CLASS_NAMES[chal_probs.argmax()]}** ({chal_probs.max():.1%} confidence)")
            st.bar_chart(pd.Series(chal_probs, index=CLASS_NAMES))

with tab_generate:
    chosen_class = st.selectbox("Pick a class", CLASS_NAMES, key="gen_class")
    extra_prompt = st.text_input("Extra prompt detail (champion/LoRA only -- the challenger only sees the class)", "")
    seed = st.number_input("Seed", value=42, step=1)

    if st.button("Generate", key="generate_btn"):
        champ_col, chal_col = st.columns(2)

        with champ_col:
            st.subheader("Champion (LoRA)")
            with st.spinner("Generating..."):
                pipe = load_lora_pipeline()
                prompt = chosen_class if chosen_class != "Other" else "an OSRS character"
                if extra_prompt:
                    prompt = f"{prompt}, {extra_prompt}"
                generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
                image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
            st.image(image, caption=prompt, width=256)

        with chal_col:
            st.subheader("Challenger (DCGAN)")
            with st.spinner("Generating..."):
                g = load_gan_generator()
                label_idx = CLASS_NAMES.index(chosen_class)
                gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
                z = torch.randn(1, Z_DIM, device=DEVICE, generator=gen)
                label = torch.tensor([label_idx], device=DEVICE)
                with torch.no_grad():
                    img_t = g(z, label).cpu().squeeze(0)
                img_t = (img_t * 0.5 + 0.5).clamp(0, 1)
                image = transforms.ToPILImage()(img_t)
            st.image(image, caption=chosen_class, width=256)

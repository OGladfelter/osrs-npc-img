"""Shared data prep for the classification champion (03a/04a) and challenger (03b/04b/04c)
notebooks. Factored out so both models are trained/evaluated on an identical
70/15/15 split -- copy-pasting this into each notebook risked the splits silently
drifting apart, which would make the champion-challenger comparison unfair.

Mirrors the cleaning logic originally inlined in 03_train.ipynb / 04_eval.ipynb.
"""

import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 128
MINORITY_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])


def load_clean_npc_df(data_dir="../data"):
    """Load npc.csv and apply the same cleaning as 03_train.ipynb / 04_eval.ipynb,
    filtered to NPCs that have both a chathead and a body image on disk.
    """
    df = pd.read_csv(f"{data_dir}/npc.csv")

    df["Gender"] = df["Gender"].apply(lambda x: x if x in ("Male", "Female") else "Other")
    df["Members"] = df["Members"].replace("? (edit)", "Yes")
    df = df.rename(columns={"Race": "Class"})
    df["Class"] = df["Class"].replace({
        "Citizen of Arceuus": "Human",
        "Dwarf ( Imcando-descendant )": "Dwarf",
    })
    df["has_chathead"] = df["id"].apply(lambda i: os.path.exists(f"{data_dir}/chatheads/{i}.png"))
    df["has_body"] = df["id"].apply(lambda i: os.path.exists(f"{data_dir}/bodies/{i}.png"))

    df = df[df["has_chathead"] & df["has_body"]].reset_index(drop=True)
    return df


def add_top_class_labels(df, n=10):
    """Collapse to the top-n classes + 'Other', matching 03_train.ipynb."""
    top_classes = df["Class"].value_counts().head(n).index.tolist()
    df = df.copy()
    df["label"] = df["Class"].apply(lambda x: x if x in top_classes else "Other")

    class_names = top_classes + ["Other"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    df["label_idx"] = df["label"].map(class_to_idx)
    return df, class_names, class_to_idx


def stratified_split(df, seed=42):
    """70/15/15 train/val/test, stratified by label_idx -- identical call signature
    and random_state to 03_train.ipynb so champion and challenger see the same rows.
    """
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_idx"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_idx"], random_state=seed)
    return train_df, val_df, test_df


def get_transforms(img_size=IMG_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return train_transform, eval_transform


class NPCDataset(Dataset):
    """Same heavier-augmentation-for-minority-classes behavior as 03_train.ipynb,
    but driven by an explicit `is_train` flag instead of comparing transform identity.
    """

    def __init__(self, df, transform, data_dir="../data", is_train=False, minority_classes=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.data_dir = data_dir
        self.is_train = is_train
        self.minority_classes = minority_classes or set()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chathead = Image.open(f"{self.data_dir}/chatheads/{row['id']}.png").convert("RGB")
        body = Image.open(f"{self.data_dir}/bodies/{row['id']}.png").convert("RGB")
        if self.is_train and row["label"] in self.minority_classes:
            chathead = MINORITY_TRANSFORM(chathead)
            body = MINORITY_TRANSFORM(body)
        return self.transform(chathead), self.transform(body), row["label_idx"]


def class_weights(df, num_classes, device):
    counts = df["label_idx"].value_counts().sort_index().reindex(range(num_classes), fill_value=1).values
    return torch.tensor(1.0 / counts, dtype=torch.float).to(device)

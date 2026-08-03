"""Shared evaluation helpers for the classification champion (04a) and challenger (04b)
notebooks, plus the head-to-head comparison (04c). Both `DualCNN` (03a) and `ScratchDualCNN`
(03b) expose the same `.chathead_cnn` / `.body_cnn` / `.classifier` interface, so all of this
code works unmodified against either model -- only Grad-CAM needs an explicit list of
(layer_module, label) pairs from the caller, since ResNet18 and the from-scratch backbone
organize their layers differently.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for chathead, body, labels in loader:
        logits = model(chathead.to(device), body.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_preds.extend(probs.argmax(1).tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs)
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def print_classification_report(all_labels, all_preds, class_names):
    print(classification_report(all_labels, all_preds, target_names=class_names))


def plot_confusion_matrix(all_labels, all_preds, class_names, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return cm


def predictions_grid(model, test_df, class_names, eval_transform, data_dir, device, n=16, seed=42, save_path=None):
    model.eval()
    samples = test_df.sample(n, random_state=seed).reset_index(drop=True)
    rows = int(np.sqrt(n))
    fig, axes = plt.subplots(rows, n // rows, figsize=(4 * (n // rows), 4 * rows))
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        chathead = eval_transform(Image.open(f"{data_dir}/chatheads/{row['id']}.png").convert("RGB")).unsqueeze(0).to(device)
        body = eval_transform(Image.open(f"{data_dir}/bodies/{row['id']}.png").convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_idx = model(chathead, body).argmax(1).item()
        pred = class_names[pred_idx]
        true = row["label"]
        ax.imshow(Image.open(f"{data_dir}/bodies/{row['id']}.png").convert("RGB"))
        ax.set_title(f"True: {true}\nPred: {pred}", fontsize=9, color="green" if pred == true else "red")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def show_prediction(model, test_df, class_name, class_names, eval_transform, data_dir, device, seed=42):
    model.eval()
    subset = test_df[test_df["label"] == class_name]
    if len(subset) == 0:
        print(f"No test examples for class '{class_name}'")
        return
    row = subset.sample(1, random_state=seed).iloc[0]

    chathead = eval_transform(Image.open(f"{data_dir}/chatheads/{row['id']}.png").convert("RGB")).unsqueeze(0).to(device)
    body = eval_transform(Image.open(f"{data_dir}/bodies/{row['id']}.png").convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(chathead, body)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(Image.open(f"{data_dir}/chatheads/{row['id']}.png"))
    axes[0].set_title("Chathead"); axes[0].axis("off")
    axes[1].imshow(Image.open(f"{data_dir}/bodies/{row['id']}.png"))
    axes[1].set_title("Body"); axes[1].axis("off")
    axes[2].barh(class_names, probs)
    axes[2].set_xlim(0, 1)
    axes[2].set_title(f"True: {row['label']} | Pred: {class_names[probs.argmax()]}")
    plt.tight_layout()
    plt.show()


def grad_cam(model, chathead_tensor, body_tensor, class_idx, target_layer):
    gradients, activations = [], []

    def forward_hook(module, inp, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(chathead_tensor, body_tensor)
    output[0, class_idx].backward()

    fh.remove()
    bh.remove()

    grad = gradients[0].squeeze()
    act = activations[0].squeeze()
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * act).sum(0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def show_grad_cam(model, test_df, class_name, class_names, eval_transform, data_dir, device,
                   chathead_layers, body_layers, seed=None):
    """chathead_layers / body_layers: list of (nn.Module, label) pairs, shallow -> deep."""
    subset = test_df[test_df["label"] == class_name]
    if len(subset) == 0:
        print(f"No test examples for class '{class_name}'")
        return
    row = subset.sample(1, random_state=seed).iloc[0]

    chathead_img = Image.open(f"{data_dir}/chatheads/{row['id']}.png").convert("RGB")
    body_img = Image.open(f"{data_dir}/bodies/{row['id']}.png").convert("RGB")

    chathead_t = eval_transform(chathead_img).unsqueeze(0).to(device).requires_grad_(True)
    body_t = eval_transform(body_img).unsqueeze(0).to(device).requires_grad_(True)

    with torch.no_grad():
        pred_idx = model(chathead_t, body_t).argmax(1).item()

    n_layers = len(chathead_layers)
    fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))

    for row_idx, (img, label, layers) in enumerate([
        (chathead_img, "Chathead", chathead_layers),
        (body_img, "Body", body_layers),
    ]):
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"Original\n({label})")
        axes[row_idx, 0].axis("off")

        for col, (layer, name) in enumerate(layers):
            cam = grad_cam(model, chathead_t, body_t, pred_idx, layer)
            scale = max(1, img.width // cam.shape[1])
            axes[row_idx, col + 1].imshow(img)
            axes[row_idx, col + 1].imshow(
                plt.cm.jet(cam.repeat(scale, axis=0).repeat(scale, axis=1))[:, :, :3],
                alpha=0.5, extent=[0, img.width, img.height, 0],
            )
            axes[row_idx, col + 1].set_title(name, fontsize=9)
            axes[row_idx, col + 1].axis("off")

    plt.suptitle(f"True: {class_name} | Pred: {class_names[pred_idx]}", fontsize=12)
    plt.tight_layout()
    plt.show()


def tsne_of_body_features(model, loader, device, class_names, title, save_path=None):
    model.eval()
    features, labels_list = [], []
    with torch.no_grad():
        for chathead, body, labels in loader:
            body_feats = model.body_cnn(body.to(device)).cpu().numpy()
            features.append(body_feats)
            labels_list.extend(labels.tolist())

    features = np.vstack(features)
    labels_arr = np.array(labels_list)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, cls in enumerate(class_names):
        mask = labels_arr == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1], label=cls, alpha=0.7, s=20, color=colors[i])
    ax.legend()
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(checkpoint_path):
    import os
    return os.path.getsize(checkpoint_path) / (1024 ** 2)


@torch.no_grad()
def measure_inference_latency(model, loader, device, n_batches=10):
    import time
    model.eval()
    times = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            chathead, body, _ = next(it)
        except StopIteration:
            break
        chathead, body = chathead.to(device), body.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(chathead, body)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed / chathead.size(0))
    return np.mean(times) * 1000  # ms/image

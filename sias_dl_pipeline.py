"""
SIAS XIM Deep Learning Pipeline
================================
Commands:
  build  – Parse .xim files and build a labelled crop dataset
  train  – Fine-tune ResNet-18 on that dataset
  watch  – Real-time inference on newly archived .xim files

XIM binary format (little-endian):
  4 bytes  : magic / version
  4 bytes  : width  (uint32)
  4 bytes  : height (uint32)
  width*height * 4 bytes : float32 pixel values (raw image)
  width*height * 1 byte  : uint8  mask values  (SIAS defect mask)
"""

import argparse
import json
import logging
import os
import struct
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("sias")

# ---------------------------------------------------------------------------
# XIM reader
# ---------------------------------------------------------------------------

def read_xim(path: Path):
    """
    Parse a .xim binary file.

    Returns
    -------
    image : np.ndarray  shape (H, W), dtype float32  – raw pixel intensities
    mask  : np.ndarray  shape (H, W), dtype uint8    – SIAS defect mask
    """
    with open(path, "rb") as f:
        data = f.read()

    offset = 0

    # 4-byte header (magic / version) – skip
    offset += 4

    width  = struct.unpack_from("<I", data, offset)[0]; offset += 4
    height = struct.unpack_from("<I", data, offset)[0]; offset += 4

    npix = width * height

    image_bytes = npix * 4
    image = np.frombuffer(data, dtype="<f4", count=npix, offset=offset).reshape(height, width)
    offset += image_bytes

    mask = np.frombuffer(data, dtype=np.uint8, count=npix, offset=offset).reshape(height, width)

    return image.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Connected-component cropping
# ---------------------------------------------------------------------------

def _label_connected(mask: np.ndarray):
    """Simple flood-fill labelling (4-connectivity) without scipy dependency."""
    try:
        from scipy.ndimage import label as scipy_label
        labelled, n = scipy_label(mask > 0)
        return labelled, n
    except ImportError:
        pass

    # Pure-numpy fallback: row-scan union-find
    H, W = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    parent = [0]  # 0 = background

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    current = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c] == 0:
                continue
            above = labels[r - 1, c] if r > 0 else 0
            left  = labels[r, c - 1] if c > 0 else 0
            if above == 0 and left == 0:
                current += 1
                parent.append(current)
                labels[r, c] = current
            elif above != 0 and left == 0:
                labels[r, c] = above
            elif above == 0 and left != 0:
                labels[r, c] = left
            else:
                union(above, left)
                labels[r, c] = find(above)

    # Compress labels
    for r in range(H):
        for c in range(W):
            if labels[r, c]:
                labels[r, c] = find(labels[r, c])
    n = len(set(labels.flat)) - 1  # exclude background 0
    return labels, n


def extract_crops(image: np.ndarray, mask: np.ndarray, min_pixels: int = 4):
    """
    For each connected component in *mask* return the bounding-box crop of *image*.

    Returns list of np.ndarray (float32, variable size).
    """
    labelled, n_comp = _label_connected(mask)
    crops = []
    for lbl in range(1, n_comp + 1):
        ys, xs = np.where(labelled == lbl)
        if len(ys) < min_pixels:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crops.append(image[y0:y1, x0:x1])
    return crops


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def crop_to_tensor(crop: np.ndarray, size: int = 128):
    """
    Normalise a float32 crop to [0,1], resize to (size × size),
    replicate to 3 channels, return torch.Tensor (3, size, size).
    """
    import torch
    from PIL import Image

    # Normalise to [0, 255] uint8 for PIL
    mn, mx = crop.min(), crop.max()
    if mx > mn:
        arr = ((crop - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        arr = np.zeros_like(crop, dtype=np.uint8)

    pil = Image.fromarray(arr, mode="L").resize((size, size), Image.BILINEAR)
    rgb = Image.merge("RGB", [pil, pil, pil])

    tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)  # (3, H, W)

    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def cmd_build(args):
    import shutil

    xim_dir   = Path(args.xim_dir)
    ds_dir    = Path(args.dataset_dir)
    label_map = json.loads(args.label_map) if args.label_map else {}
    default_label = "defect"

    xim_files = sorted(xim_dir.rglob("*.xim"))
    if not xim_files:
        log.error("No .xim files found in %s", xim_dir)
        return

    log.info("Found %d .xim files", len(xim_files))

    total_crops = 0
    per_label: dict[str, int] = {}

    for xim_path in xim_files:
        stem = xim_path.stem.lower()

        # Determine label from prefix mapping
        label = default_label
        for prefix, mapped in label_map.items():
            if stem.startswith(prefix.lower()):
                label = mapped
                break

        try:
            image, mask = read_xim(xim_path)
        except Exception as exc:
            log.warning("Could not parse %s: %s", xim_path, exc)
            continue

        crops = extract_crops(image, mask)
        if not crops:
            log.debug("No crops in %s", xim_path.name)
            continue

        label_dir = ds_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for i, crop in enumerate(crops):
            from PIL import Image as PILImage
            mn, mx = crop.min(), crop.max()
            if mx > mn:
                arr = ((crop - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(crop, dtype=np.uint8)
            out_path = label_dir / f"{xim_path.stem}_crop{i:04d}.png"
            PILImage.fromarray(arr).save(out_path)
            total_crops += 1
            per_label[label] = per_label.get(label, 0) + 1
            log.debug("Saved %s", out_path)

        log.info("  %s → %d crops [%s]", xim_path.name, len(crops), label)

    log.info("Dataset built: %d crops total", total_crops)
    for lbl, cnt in sorted(per_label.items()):
        log.info("  %-20s : %d samples", lbl, cnt)

    # Write class index
    class_list = sorted(per_label.keys())
    idx_path = ds_dir / "classes.json"
    idx_path.write_text(json.dumps(class_list, indent=2))
    log.info("Class list written to %s", idx_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cmd_train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import models
    from PIL import Image

    ds_dir    = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load class list
    idx_path = ds_dir / "classes.json"
    if not idx_path.exists():
        log.error("Run 'build' first – %s not found", idx_path)
        return
    classes = json.loads(idx_path.read_text())
    num_classes = len(classes)
    log.info("Classes (%d): %s", num_classes, classes)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    class CropDataset(Dataset):
        def __init__(self, root: Path, classes: list[str], size: int = 128):
            self.size = size
            self.samples: list[tuple[Path, int]] = []
            for lbl_idx, lbl in enumerate(classes):
                lbl_dir = root / lbl
                if not lbl_dir.is_dir():
                    continue
                for img_path in sorted(lbl_dir.glob("*.png")):
                    self.samples.append((img_path, lbl_idx))
            if not self.samples:
                raise RuntimeError(f"No PNG images found under {root}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("L")
            img = img.resize((self.size, self.size), Image.BILINEAR)
            rgb = Image.merge("RGB", [img, img, img])
            arr = np.array(rgb, dtype=np.float32) / 255.0

            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor, label

    full_ds = CropDataset(ds_dir, classes, size=args.crop_size)
    n_total = len(full_ds)
    n_val   = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    log.info("Dataset split: %d train / %d val", n_train, n_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt    = model_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # --- validate ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
        val_acc = correct / n_val

        scheduler.step()

        log.info("Epoch %3d/%d  loss=%.4f  val_acc=%.3f", epoch, args.epochs, train_loss, val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, best_ckpt)
            log.info("  → checkpoint saved (val_acc=%.3f)", val_acc)

    # Save final
    final_ckpt = model_dir / "final.pt"
    torch.save({"model_state": model.state_dict(), "classes": classes}, final_ckpt)
    log.info("Training complete. Best val_acc=%.3f", best_val_acc)
    log.info("Checkpoints: %s  %s", best_ckpt, final_ckpt)


# ---------------------------------------------------------------------------
# Real-time inference (watchdog)
# ---------------------------------------------------------------------------

def _load_model(model_dir: Path, device):
    """Load best checkpoint and return (model, classes)."""
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt_path = model_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "final.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, classes


def _predict_xim(xim_path: Path, model, classes: list, device, crop_size: int = 128):
    """
    Parse a .xim, extract crops, run inference on each crop.

    Returns list of (crop_idx, predicted_class, confidence).
    """
    import torch

    try:
        image, mask = read_xim(xim_path)
    except Exception as exc:
        log.warning("Could not parse %s: %s", xim_path, exc)
        return []

    crops = extract_crops(image, mask)
    if not crops:
        return []

    tensors = torch.stack([crop_to_tensor(c, crop_size) for c in crops]).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensors)
        probs  = torch.softmax(logits, dim=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for i, (prob_row) in enumerate(probs):
        conf, cls_idx = prob_row.max(0)
        results.append((i, classes[cls_idx.item()], float(conf)))

    log.debug("Inference on %d crops in %.1f ms", len(crops), elapsed_ms)
    return results


def cmd_watch(args):
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        log.error("Install watchdog: pip install watchdog")
        raise SystemExit(1)

    import torch

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    xim_dir   = Path(args.xim_dir)
    crop_size = args.crop_size

    log.info("Loading model from %s …", model_dir)
    model, classes = _load_model(model_dir, device)
    log.info("Model ready. Classes: %s", classes)

    class XimHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix.lower() != ".xim":
                return
            log.info("New file detected: %s", p.name)
            # Brief wait to ensure write is complete
            time.sleep(0.1)
            results = _predict_xim(p, model, classes, device, crop_size)
            if not results:
                log.info("  No defect crops found in %s", p.name)
                return
            for crop_idx, cls, conf in results:
                log.info("  crop %04d → %-20s (conf=%.3f)", crop_idx, cls, conf)

        def on_moved(self, event):
            """Handle files moved/renamed into the watched directory."""
            self.on_created(type("E", (), {"is_directory": False, "src_path": event.dest_path})())

    observer = Observer()
    observer.schedule(XimHandler(), str(xim_dir), recursive=True)
    observer.start()
    log.info("Watching %s  (Ctrl-C to stop)", xim_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    log.info("Watcher stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="sias_dl_pipeline",
        description="SIAS XIM Deep Learning Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- build ----
    p_build = sub.add_parser("build", help="Parse .xim files and build crop dataset")
    p_build.add_argument("--xim_dir",    required=True, help="Root directory containing .xim files")
    p_build.add_argument("--dataset_dir",required=True, help="Output dataset directory")
    p_build.add_argument("--label_map",  default=None,
                         help='JSON dict mapping filename prefix → class label. '
                              'E.g. \'{"top434":"PSD871","top435":"PSD872"}\'')

    # ---- train ----
    p_train = sub.add_parser("train", help="Fine-tune ResNet-18 on crop dataset")
    p_train.add_argument("--dataset_dir", required=True)
    p_train.add_argument("--model_dir",   required=True)
    p_train.add_argument("--epochs",      type=int,   default=30)
    p_train.add_argument("--batch_size",  type=int,   default=32)
    p_train.add_argument("--lr",          type=float, default=1e-4)
    p_train.add_argument("--crop_size",   type=int,   default=128)

    # ---- watch ----
    p_watch = sub.add_parser("watch", help="Real-time inference on new .xim files")
    p_watch.add_argument("--xim_dir",   required=True)
    p_watch.add_argument("--model_dir", required=True)
    p_watch.add_argument("--crop_size", type=int, default=128)

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "watch":
        cmd_watch(args)


if __name__ == "__main__":
    main()

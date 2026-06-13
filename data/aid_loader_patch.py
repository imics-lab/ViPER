"""
aid_loader_patch.py — Add AID dataset to ~/ViPER/data/data-loader.py

AID (Aerial Image Dataset):
  - 10,000 aerial images from Google Earth
  - 30 scene categories (airport, beach, bridge, ...)
  - 600×600 native resolution (2× higher than RESISC45)
  - Standard 50/50 train/test split

Two integration options:
  A. HuggingFace cache (recommended) — auto-downloads via `jonathan-roberts1/AID`
  B. Manual download from captain-whu.github.io/AID

This script does Option A (HuggingFace).

To apply:
    cd ~/ViPER/data
    python aid_loader_patch.py    # patches data-loader.py in place
"""

import re
from pathlib import Path

PATCH_BLOCK = '''

# ─── 5. AID (Aerial Image Dataset — 10k aerial RGB images, 600x600 native) ───
def get_aid(data_root="./data", batch_size=32, image_size=224,
            seed=SEED, num_workers=2):
    """AID — 30-class aerial scene classification, native 600x600 from Google Earth.

    Uses HuggingFace `jonathan-roberts1/AID` dataset (auto-downloads ~600 MB).
    Standard 50/50 train/test split; we further reserve 15% of train for val.
    """
    from datasets import load_dataset

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    cache_dir = str(Path(data_root) / "aid_hf")
    ds_full = load_dataset("jonathan-roberts1/AID", cache_dir=cache_dir)

    # Handle different split conventions across HF mirrors
    if "train" in ds_full and "test" in ds_full:
        # Standard 50/50 split; carve val from train
        all_train = ds_full["train"]
        all_test = ds_full["test"]
        n = len(all_train)
        rng = list(range(n))
        # Deterministic 85/15 split on the existing train
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        n_val = int(n * 0.15)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        tr_hf = all_train.select(tr_idx)
        va_hf = all_train.select(val_idx)
        te_hf = all_test
    elif "train" in ds_full:
        # Only 'train' split — do 70/15/15 ourselves
        full = ds_full["train"]
        tr_i, va_i, te_i = make_split(len(full), 0.15, 0.15, seed)
        tr_hf = full.select(tr_i)
        va_hf = full.select(va_i)
        te_hf = full.select(te_i)
    else:
        raise RuntimeError(f"Unexpected AID splits: {list(ds_full.keys())}")

    tr_ds = HFImageDataset(tr_hf, train_tf)
    va_ds = HFImageDataset(va_hf, eval_tf)
    te_ds = HFImageDataset(te_hf, eval_tf)

    return (*make_loaders(tr_ds, va_ds, te_ds, batch_size, num_workers),
            30, image_size, image_size)
'''

REGISTRY_PATCH = '    "aid":         get_aid,\n'
DEFAULTS_PATCH = '    "aid":         dict(image_size=224, batch_size=32),\n'


def main():
    path = Path("data-loader.py")
    if not path.exists():
        raise FileNotFoundError("Run this from ~/ViPER/data/")

    text = path.read_text()

    # 1) Insert function block before DATASET_REGISTRY
    if "def get_aid" in text:
        print("AID function already present")
    else:
        anchor = "# ─── Unified registry ────"
        if anchor not in text:
            # Fall back to inserting just before DATASET_REGISTRY = {
            anchor = "DATASET_REGISTRY = {"
            text = text.replace(anchor, PATCH_BLOCK + "\n" + anchor)
        else:
            text = text.replace(anchor, PATCH_BLOCK + "\n" + anchor)
        print("Added get_aid function")

    # 2) Add to DATASET_REGISTRY
    if '"aid"' in text and "DATASET_REGISTRY" in text:
        # Verify aid is INSIDE the registry block
        reg_match = re.search(r'DATASET_REGISTRY\s*=\s*\{([^}]*)\}', text)
        if reg_match and '"aid"' in reg_match.group(1):
            print("AID already in DATASET_REGISTRY")
        else:
            text = text.replace(
                '"dtd":         get_dtd,\n}',
                '"dtd":         get_dtd,\n' + REGISTRY_PATCH + '}'
            )
            print("Added AID to DATASET_REGISTRY")
    else:
        text = text.replace(
            '"dtd":         get_dtd,\n}',
            '"dtd":         get_dtd,\n' + REGISTRY_PATCH + '}'
        )
        print("Added AID to DATASET_REGISTRY")

    # 3) Add to DATASET_DEFAULTS
    def_match = re.search(r'DATASET_DEFAULTS\s*=\s*\{([^}]*)\}', text)
    if def_match and '"aid"' in def_match.group(1):
        print("AID already in DATASET_DEFAULTS")
    else:
        text = text.replace(
            '"dtd":         dict(image_size=224, batch_size=32),\n}',
            '"dtd":         dict(image_size=224, batch_size=32),\n' + DEFAULTS_PATCH + '}'
        )
        print("Added AID to DATASET_DEFAULTS")

    path.write_text(text)
    print(f"\nPatched {path.absolute()}")


if __name__ == "__main__":
    main()

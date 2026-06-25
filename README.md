## ViPER — Vision Positional Encoding via Wavelet Representation

Anonymous code release for review. ViPER is a small, **drop-in positional encoding (PE) module** for Vision Transformers. It derives a content-aware positional signal from a *J*-level 2D wavelet transform of the input and plugs into ViT / DeiT / Swin / PVT-v2 with no backbone changes (adds **8,896 params**).

This README is only meant to let reviewers **set up, run, and verify** the code.

---

### 1. Setup

```bash
conda create -n viper python=3.10 -y
conda activate viper

pip install torch torchvision          # pick the build matching your CUDA
pip install timm pytorch-wavelets PyWavelets medmnist
pip install numpy scikit-learn pyyaml tqdm
```

Tested with Python 3.10, PyTorch ≥ 2.0, CUDA 11.8+ (single or multi-GPU).
CPU works for a quick smoke test but is slow.

---

### 2. Quick smoke test (≈ a few minutes on 1 GPU)

Trains ViPER on EuroSAT for 1 epoch to confirm everything runs end to end:

```bash
python viper_v5.py --dataset eurosat --backbone vit_b_16 --from_scratch \
    --seed 42 --epochs 1
```

You should see per-epoch train loss and validation accuracy/F1, and a run
history written to `results/`.

---

### 3. Run the main experiments

```bash
# ViPER on any dataset / backbone
python viper.py --dataset <DATASET> --backbone <BACKBONE> --from_scratch --seed 42

# A baseline PE for comparison
python train_extra_pes.py --dataset <DATASET> --pe <PE> --from_scratch --seed 42
```

- `<DATASET>` ∈ `eurosat, resisc45, pathmnist, bloodmnist, dermamnist, tissuemnist, dtd, flowers102, fgvc_aircraft`
- `<BACKBONE>` ∈ `vit_b_16, deit_s, swin_t, pvt_v2_b2`
- `<PE>` ∈ `nope, ape, sincos2d, rope2d, irpe, cpe, wpe`

Reported numbers are averaged over **5 seeds: 0, 1, 7, 42, 123**.

```bash
for s in 0 1 7 42 123; do
  python viper.py --dataset eurosat --backbone vit_b_16 --from_scratch --seed $s
done
```

Optional — multi-resolution generalization eval:

```bash
python eval_multires.py --dataset resisc45 --ckpt results/<checkpoint>.pt
```

---

### 4. Datasets

| Dataset | Source | Auto-download |
|---|---|:---:|
| EuroSAT, RESISC45 | remote sensing | yes |
| BloodMNIST, DermaMNIST, PathMNIST, TissueMNIST | MedMNIST | yes |
| DTD, Flowers-102, FGVC-Aircraft | torchvision | yes |

All loaders download on first use into a local data cache. No manual steps.

---

### 5. Files

```
viper.py            # ViPER module + training entry point  ← use this
train_extra_pes.py  # baseline PE training
eval_multires.py    # multi-resolution evaluation
viper_datasets.py   # dataset loaders / registry
results/            # JSON run histories (loss, acc, f1, auc per epoch)
```

---

### Notes for reviewers

- Set `--epochs` low for a fast sanity check; default trains for 30 epochs.
- ViPER adds 8,896 parameters; the per-component breakdown is printed at startup.
- Author / institution info is omitted for double-blind review.

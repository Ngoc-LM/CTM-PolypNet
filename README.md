# CTM-PolypNet

**CTM-PolypNet: A Unified Convolution-Transformer-Mamba Model for Polyp Segmentation**

> Minh-Ngoc Luong, Minh Le, Van-Truong Pham, Thi-Thao Tran  
> *2025 2nd International Conference on Health Science and Technology (ICHST)*  
> DOI: [10.1109/ICHST66555.2025.11428431
        
        ](https://doi.org/10.1109/ICHST66555.2025.11428431
        
        )

---

## Overview

CTM-PolypNet integrates three complementary paradigms in a U-Net-style architecture:

| Component | Role |
|-----------|------|
| **PVT-v2** backbone | Hierarchical global feature extraction |
| **Mamba-MLP Fusion** (proposed) | Long-range dependency modeling with efficient MLP mixing in the decoder |
| **RAFE** — Reverse Attention Feature Enhancement (proposed) | Boundary-aware feature refinement via reverse attention |
| **PASPP** | Multi-scale dilated context at the bottleneck |
| **CBAM / CoordAtt / SE** | Channel and spatial recalibration on skip connections |
| **Deep supervision** | Weighted Dice-Tversky loss at all 4 decoder stages |


---

## Repository Structure

```
CTM-PolypNet/
├── models/
│   ├── pvt_v2.py          # PVT-v2 backbone (B0–B5 variants)
│   ├── blocks.py          # CBAM, CoordAtt, SEBlock, PASPP, conv_layer, …
│   ├── mamba_blocks.py    # SS2D, VSSBlock, MambaMLPMixer, SpatialMLPAttention
│   ├── ctm_polypnet.py    # Main model + RAFEB + build_ctm_polypnet()
│   └── __init__.py
├── data/
│   ├── dataset.py         # PolypNpzDataset, PolypDirDataset, build_dataloaders()
│   └── __init__.py
├── utils/
│   ├── losses.py          # DiceLoss, TverskyLoss, DiceTverskyLoss, deep_supervision_loss
│   └── __init__.py
├── weights/               # Place pretrained weights here (not tracked by git)
├── train.py               # Training entry point (PyTorch Lightning)
├── evaluate.py            # Evaluation on all 5 test splits
├── profile_model.py       # FLOPs + parameter count
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/CTM-PolypNet.git
cd CTM-PolypNet
pip install -r requirements.txt
```

> **Note:** `mamba-ssm` and `triton` require a CUDA GPU and compatible drivers.
> For CPU-only experiments these packages can be omitted, but the model will not run.

---


Raw datasets are available from:
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
- [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
- [ETIS-LaribPolypDB](https://polyp.grand-challenge.org/EtisLarib/)

Download the PVT-v2-B2 backbone weights from the
[official PVT repository](https://github.com/whai362/PVT) and place them
in `weights/pvt_v2_b2.pth`.

---

## Training

```bash
python train.py \
    --data_path     /path/to/polypData.npz \
    --pretrained_backbone  weights/pvt_v2_b2.pth \
    --ckpt_dir      ./checkpoints \
    --max_epochs    100 \
    --batch_size    16 \
    --lr            2e-4 \
    --precision     16
```

Resume from a checkpoint:

```bash
python train.py ... --ckpt_path checkpoints/ckpt0.XXXX.ckpt
```

---

## Evaluation

```bash
python evaluate.py \
    --data_path  /path/to/polypData.npz \
    --ckpt_path  checkpoints/ckpt0.XXXX.ckpt \
    --pretrained_backbone  weights/pvt_v2_b2.pth
```

---

## Model Complexity

```bash
python profile_model.py --pretrained_backbone weights/pvt_v2_b2.pth
```


---

## Citation

```bibtex
@inproceedings{luong2025ctm,
  title={CTM-PolypNet: A Unified Convolution-Transformer-Mamba Model for Polyp Segmentation},
  author={Luong, Minh-Ngoc and Le, Minh and Pham, Van-Truong and Tran, Thi-Thao},
  booktitle={2025 2nd International Conference on Health Science and Technology (ICHST)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

---

## Acknowledgements

The backbone code is adapted from the official
[PVT-v2 repository](https://github.com/whai362/PVT).
The Mamba SSM implementation is based on
[mamba-ssm](https://github.com/state-spaces/mamba).
The RAFE block is developed from the BFEB concept in
[PCRNet](https://github.com/), building on the boundary-aware design of
[LiteMamba-Bound](https://github.com/).

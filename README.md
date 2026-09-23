# SGFRNet

Official code organization for **Enhancing Target and Rectifying Clutter: A Structure-Guided Frequency Rectification Network for Infrared Small Target Detection, 2026, IEEE JSTARS**.

This release contains the SGFRNet architecture, a self-contained training/evaluation entry point, dataset loaders, losses, evaluation metrics, the warmup scheduler, and source implementations of the comparison models used in the experiments. 

## Project structure

```text
SGFRNet/
├── train.py                      #
├── test.py                       # 
├── dataset.py                    # Dataset loading and augmentation
├── loss.py                       # SGFRNet training losses
├── metrics.py                    # mIoU, nIoU, Pd, and Fa
├── utils.py                      # Reproducibility and preprocessing helpers
├── warmup_scheduler.py           # Gradual learning-rate warmup
├── model/
│   ├── SGFRNet.py                # SGFRNet architecture
│   ├── Config.py                 # Configuration used by comparison models
│   └── comparative_experiment/  # Comparison-model source code
├── requirements.txt
└── requirements-comparison.txt
```

All default paths are relative to the repository root. The training entry point has no machine-specific absolute path.

## Environment

Python 3.10 or later is recommended.

Install the dependencies required by SGFRNet:

```bash
pip install -r requirements.txt
```

To import and inspect every comparison model, install the optional dependencies as well:

```bash
pip install -r requirements-comparison.txt
```

## Dataset layout

Place a dataset under `datasets/` using the following structure:

```text
datasets/
└── IRSTD-1K/
    ├── images/
    │   ├── 000001.png
    │   └── ...
    ├── masks/
    │   ├── 000001.png
    │   └── ...
    └── img_idx/
        ├── trainval.txt
        └── test.txt
```

Each index file contains one image identifier per line, without requiring a filename extension. PNG, BMP, JPG, and TIFF files are supported.

## Training

The defaults reproduce the main settings retained from the original experiment script: 256-pixel patches, AdamW, a learning rate of `5e-4`, 20 warmup epochs, and 1000 total epochs.

```bash
python train.py \
  --mode train \
  --dataset-name IRSTD-1K \
  --dataset-dir ./datasets \
  --save-dir ./runs
```



Resume a saved run with optimizer and scheduler states:

```bash
python train.py \
  --mode train \
  --checkpoint ./runs/IRSTD-1K/SGFRNet_last.pth.tar \
  --resume
```

## Evaluation

```bash
python test.py \
  --dataset-name IRSTD-1K \
  --checkpoint ./runs/IRSTD-1K/SGFRNet_best.pth.tar \
  --save-predictions
```

The script reports foreground mIoU, sample-wise nIoU, object-level probability of detection (`Pd`), and pixel-level false-alarm rate (`Fa`). It also prints `Fa × 10^6`, matching the reporting convention used in the paper. Metrics are saved to `./results/<dataset-name>/metrics.json`; predicted masks are written only when `--save-predictions` is enabled.

The same evaluation path remains available through `python train.py --mode test` for backward compatibility.

Checkpoints produced by the former `Net(model=...)` wrapper are accepted automatically when all parameter names use the `model.` prefix.

## Comparison models

The source files under `model/comparative_experiment/` are kept separate from the SGFRNet entry point. This prevents optional packages required by one comparison method from blocking SGFRNet training. 

## Citation

@article{RN1655,
   author = {Tu, Peng and Sun, Chunqiang and Liu, Jianjun},
   title = {Enhancing Target and Rectifying Clutter: A Structure-Guided Frequency Rectification Network for Infrared Small-Target Detection},
   journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remo te Sensing},
   volume = {19},
   pages = {28002-28020},
   DOI = {10.1109/JSTARS.2026.3725527},
   year = {2026},
   type = {Journal Article}
}


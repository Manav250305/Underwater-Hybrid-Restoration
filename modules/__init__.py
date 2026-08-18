"""
Underwater Image Restoration — Hybrid Adaptive Pipeline
VIT BCSE403L Digital Image Processing Project

Modules:
  1. white_balance.py         - Adaptive Gray-World white balance
  2. gamma_correction.py      - Histogram-driven adaptive gamma correction
  3. clahe_module.py          - CLAHE in CIELAB, adaptive clip limit
  4. dehazing.py              - Underwater Dark Channel Prior + guided filter
  5. water_type_classifier.py - NOVEL: routes stage parameters by water type
  6. fusion.py                - Laplacian pyramid multi-scale fusion
  7. fusion_optimizer.py      - NOVEL: quality-metric-guided fusion weight search
  8. metrics.py               - PSNR, SSIM, entropy, UCIQE, UIQM
  9. downstream_validation.py - NOVEL: ORB/SIFT + edge-density task validation
 10. dataset_loader.py         - Resolves UIEB raw/reference/challenging paths
                                 (local data/ or external SSD) and iterates pairs
"""

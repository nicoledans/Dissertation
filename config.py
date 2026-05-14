SAMPLE_SIZE = 600
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
ALPHA = 0.15
SEED = 42
IMG_SIZE = 224
HU_MIN = -1000
HU_MAX = 400
LUNG_HU_THRESHOLD = -500
MASK_DILATION = 3
RESULTS_DIR = "results"
# Split ratios: 70/15/15 train/val/test — enforced in dataset.patient_split()

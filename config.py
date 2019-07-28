import numpy as np
import math
from collections import OrderedDict


class Config(object):
    ##############################
    # Data And Dataset
    ##############################
    DATA_DIR = None
    MAX_NUMS = 100
    CLASSES = 80
    OUTPUT_SIZE = [128, 128]
    OUTPUT_STRIDE = 4
    GAUSSIAN_BUMP = True
    RADIUS = -1
    MINI_IOU = 0.7
    VIEW_SIZE = [512, 512]
    RANDOM_SCALES = np.arange(0.6, 1.4, 0.1)
    BORDER = 128
    MEANS = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    STD = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    EIG_VAL = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    EIG_VEC = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    ###################################
    # Network config
    ###################################
    NET_NAME = "center52"
    NUM_FEATS = 256
    INTER_CHANNELS = [256, 256, 384, 384, 384, 512]
    ###################################
    # Training Config
    ###################################
    COMPUTE_TIME = False
    CLIP_GRADIENT_NORM = 5.0
    EPOCH_BOUNDARY = [25]
    EPOCHS = 30
    WEIGHT_DECAY = 0.0001
    EPSILON = 1e-5
    MOMENTUM = 0.9
    BOUNDARYS = [15]
    LR_VALS = [2.5e-4, 2.5e-5]
    PER_GPU_IMAGE = 2
    NUM_GPUS = 1
    PRE_GPU_BATCH_SIZE = 4
    LOSS_WEIGHTS = {"ae_loss": 0.1}
    SAVE_EVERY_N_STEP = 10
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        self.BATCH_SIZE = self.NUM_GPUS*self.PRE_GPU_BATCH_SIZE
        self.META_SHAPE =1 + 3 + 3 + 4 + 1 + self.CLASSES


class COCOConfig(Config):
    CLASSES = 80
    NUM_SAMPLES = 118287
    DATA_DIR = "coco"
    TOP_K = 70
    AE_THRESHOLD = 0.5
    NM_THRESHOLD = 0.5

    # the summary and model will be saved in this location
    MODLE_DIR = "./logs"
    NET_NAME = "center52"

    def __init__(self):
        Config.__init__(self)
        self.STEPS_PER_EPOCH = int(self.NUM_SAMPLES/self.BATCH_SIZE)
        self.VAL_STEPS = int(5000/self.BATCH_SIZE)

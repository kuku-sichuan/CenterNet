import keras
import cv2
import numpy as np
from data.data import CocoDataset
from model.model import CenterNet52
from config import COCOConfig
import matplotlib.pyplot as plt
from data.data import load_image_gt
from data.transform import np_draw_gaussian
from libs.visualization import display_instances, display_heatmap

config = COCOConfig()
dataset_train = CocoDataset()
dataset_train.load_coco(config.DATA_DIR, "val", auto_download=False)
dataset_train.prepare()
image_id = np.random.choice(dataset_train.image_ids)
for image_id in dataset_train.image_ids:
    image, bboxs, class_ids, image_meta = load_image_gt(dataset_train, config, image_id, True)
    image *= 255
    out = np_draw_gaussian(bboxs, class_ids, config)
    #display_instances(image, bboxs, class_ids, dataset_train.class_names)
    #display_heatmap(image, out[0], out[1], out[2])
   # plt.show()
cv2.imwrite("temp.jpg", image)

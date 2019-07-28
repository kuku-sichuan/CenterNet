import os
import sys
import numpy as np
from pycocotools.coco import COCO

import  logging
import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from data import utils
from data.image import random_crop, color_jittering, lighting, normalize
from data.image import pad_same_size
from data.transform import np_draw_gaussian


class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        year = 2017 # or 2014
        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_bbox(self, image_id):
        """Load instance bounding box for the given image.
        Returns:
        bbox: A bool array of shape [num_instances, 4].(y1,x1,y2,x2)
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_bbox(image_id)

        instance_bboxs = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                x1, y1, w, h = annotation['bbox']
                bbox = np.array([y1, x1, y1+h, x1+w], dtype=np.float32)
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    bbox = np.zeros([4,], dtype=np.float32)
                instance_bboxs.append(bbox)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            bboxs = np.stack(instance_bboxs, axis=0).astype(np.float32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return bboxs, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_bbox(image_id)


def load_image_gt(dataset, config, image_id, augment=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).
    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    bboxs, class_ids = dataset.load_bbox(image_id)
    original_shape = image.shape
    if augment:
        rand_scales = np.random.choice(config.RANDOM_SCALES)
        image, bboxs = random_crop(image, bboxs, rand_scales, config.VIEW_SIZE, border=config.BORDER)
        data_rng = np.random.RandomState(123)
        color_jittering(data_rng, image)
        lighting(data_rng, image, 0.1, config.EIG_VAL, config.EIG_VEC)
    normalize(image, config.MEANS, config.STD)
    image, bboxs, scale, windows = pad_same_size(image, bboxs, config.VIEW_SIZE)
    
    valid_bbox_index = np.logical_and(bboxs[:,0] < bboxs[:, 2], bboxs[:,1] < bboxs[:, 3]) 
    bboxs = bboxs[valid_bbox_index]
    class_ids = class_ids[valid_bbox_index]
    assert bboxs.shape[0]==class_ids.shape[0]

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1
    image_meta = utils.compose_image_meta(image_id, original_shape, image.shape, windows, scale, active_class_ids)

    return image, bboxs, class_ids, image_meta


def data_generator(dataset, config, shuffle=True, augment=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment:  If true, apply random image augmentation.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,.
    """
    global batch_tl_heatmaps, batch_br_heatmaps, batch_ct_heatmaps, batch_tl_reg,\
        batch_br_reg, batch_ct_reg,batch_mask, batch_tl_tag, batch_br_tag, batch_tag_mask,\
        batch_images, batch_image_metas, batch_gt_boxes, batch_gt_class_ids

    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes
            image_id = image_ids[image_index]
            image, gt_boxes, gt_class_ids, image_meta = load_image_gt(dataset, config, image_id, augment=augment)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # generating the keypoint heatmap
            # tl_heatmaps, br_heatmaps, ct_heatmaps, tl_regrs, br_regrs, ct_regrs, mask, tag_mask, tl_tags, br_tags
            out = np_draw_gaussian(gt_boxes, gt_class_ids, config)


            # Init batch arrays
            if b == 0:
                batch_tl_heatmaps = np.zeros([config.BATCH_SIZE,]+ config.OUTPUT_SIZE+[config.CLASSES],
                                             dtype=out[0].dtype)
                batch_br_heatmaps = np.zeros([config.BATCH_SIZE,]+ config.OUTPUT_SIZE+[config.CLASSES],
                                             dtype=out[1].dtype)
                batch_ct_heatmaps = np.zeros([config.BATCH_SIZE,] + config.OUTPUT_SIZE+[config.CLASSES],
                                             dtype=out[2].dtype)
                batch_tl_reg = np.zeros([config.BATCH_SIZE,] + config.OUTPUT_SIZE+[2], dtype=out[3].dtype)
                batch_br_reg = np.zeros([config.BATCH_SIZE,] + config.OUTPUT_SIZE+[2], dtype=out[4].dtype)
                batch_ct_reg = np.zeros([config.BATCH_SIZE,] + config.OUTPUT_SIZE+[2], dtype=out[5].dtype)
                batch_mask = np.zeros([config.BATCH_SIZE, 3, ] + config.OUTPUT_SIZE, dtype=out[6].dtype)
                batch_tag_mask = np.zeros([config.BATCH_SIZE, config.MAX_NUMS], dtype=out[7].dtype)
                batch_tl_tag = np.zeros([config.BATCH_SIZE, config.MAX_NUMS], dtype=out[8].dtype)
                batch_br_tag = np.zeros([config.BATCH_SIZE, config.MAX_NUMS], dtype=out[9].dtype)
                batch_images = np.zeros([config.BATCH_SIZE,] + config.VIEW_SIZE+[3], dtype=np.float32)
                batch_image_metas = np.zeros((config.BATCH_SIZE, config.MAX_NUMS, config.META_SHAPE), dtype=image_meta.dtype)
                batch_gt_class_ids = np.zeros([config.BATCH_SIZE, config.MAX_NUMS], dtype=np.int64)
                batch_gt_boxes = np.zeros([config.BATCH_SIZE, config.MAX_NUMS, 4], dtype=np.float32)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_NUMS:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_NUMS, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]

            # Add to batch
            batch_tl_heatmaps[b] = out[0]
            batch_br_heatmaps[b] = out[1]
            batch_ct_heatmaps[b] = out[2]
            batch_tl_reg[b] = out[3]
            batch_br_reg[b] = out[4]
            batch_ct_reg[b] = out[5]
            batch_mask[b] = out[6]
            batch_tag_mask[b] = out[7]
            batch_tl_tag[b] = out[8]
            batch_br_tag[b] = out[9]
            batch_images[b] = image
            batch_image_metas[b] = image_meta
            batch_gt_boxes[b,:gt_boxes.shape[0]] = gt_boxes
            batch_gt_class_ids[b,:gt_class_ids.shape[0]]= gt_class_ids

            b += 1
            # Batch full?
            if b >= config.BATCH_SIZE:
                inputs = [batch_images, batch_image_metas, batch_tl_heatmaps, batch_br_heatmaps, batch_ct_heatmaps, batch_tl_reg,\
                          batch_br_reg, batch_ct_reg ,batch_mask, batch_tl_tag, batch_br_tag, batch_tag_mask, \
                           batch_gt_boxes, batch_gt_class_ids]
                outputs = []
                yield inputs, outputs
                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise
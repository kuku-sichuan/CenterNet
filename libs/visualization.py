import cv2
import random
import keras
import colorsys
import skimage
import numpy as np
import tensorflow as tf
from matplotlib import patches
import matplotlib.pyplot as plt


def class_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def display_instances(image, boxes, class_ids, class_names,
                      scores=None, nb_classes=80, title="",
                      figsize=(16, 16), ax=None,
                      show_bbox=True, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    nb_classes: (optional) number of classes
    title: (optional) Figure title
    show_bbox: To show bounding boxes or not
    figsize: (optional) the size of the image
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0]  == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = class_colors(nb_classes)
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[class_ids[i]]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="solid",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
    plt.savefig("temp1.jpg")


def display_heatmap(image, tl_heatmap, br_heatmap, ct_heatmap, nb_classes=80, title="",
                    figsize=(16, 16), ax=None):
    """
    heatmap: [height, width, nb_classes] heatmap for corner points
    reg_vector:[height, width, 2] offset to gt-corner-point
    nb_classes: (optional) number of classes
    title: (optional) Figure title
    show_bbox: To show bounding boxes or not
    figsize: (optional) the size of the image
    captions: (optional) A list of strings to use as captions for each object
    """

    # If no axis is passed, create one and automatically call show()
    assert tl_heatmap.shape == br_heatmap.shape
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = class_colors(nb_classes)
    mask_image = image.astype(np.float64).copy()
    height, width = mask_image.shape[:2]
    class_heatmap = np.zeros((tl_heatmap.shape[0], tl_heatmap.shape[1], 3))
    for i in range(nb_classes):
        color_map = np.expand_dims(np.expand_dims(np.array(colors[i]) * 255, 0), 0)
        class_heatmap += np.expand_dims(tl_heatmap[:, :, i], 2) * color_map
        class_heatmap += np.expand_dims(br_heatmap[:, :, i], 2) * color_map
        class_heatmap += np.expand_dims(ct_heatmap[:, :, i], 2) * color_map
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    class_heatmap = cv2.resize(class_heatmap, image.shape[:2])
    mask_image = cv2.addWeighted(src1=mask_image, alpha=0.8, src2=class_heatmap, beta=0.4, gamma=0)
    ax.imshow(mask_image.astype(np.uint8))
    if auto_show:
        plt.show()
    plt.savefig("temp2.jpg")

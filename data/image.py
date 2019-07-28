import cv2
import numpy as np
import random


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize(image, mean, std):
    image -= mean
    image /= std


def lighting(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend(alpha, image, gs[:, :, None])


def brightness(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend(alpha, image, gs_mean)


def color_jittering(data_rng, image):
    functions = [brightness, contrast, saturation]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)


def crop_image(image, center, size):
    cty, ctx = center
    height, width = size
    im_height, im_width = image.shape[0:2]
    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
        cropped_cty - top,
        cropped_cty + bottom,
        cropped_ctx - left,
        cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width // 2
    ])

    return cropped_image, border, offset


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def random_crop(image, detections, random_scales, view_size, border=64):
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]

    scale  = random_scales
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    cropped_detections[:, 1:4:2] += cropped_cty - top_h
    # deal with the outlier bbox
    cropped_detections[:, 0] = np.maximum(cropped_detections[:, 0], cropped_cty - top_h)
    cropped_detections[:, 1] = np.maximum(cropped_detections[:, 1], cropped_ctx - left_w)
    cropped_detections[:, 2] = np.minimum(cropped_detections[:, 2], cropped_cty + bottom_h-1)
    cropped_detections[:, 3] = np.minimum(cropped_detections[:, 3], cropped_ctx + right_w-1)

    return cropped_image, cropped_detections


def pad_same_size(image, bbox, size):
    new_bbox = bbox.copy()
    height, width = image.shape[0:2]
    ratio = min(size[0] / height, size[1] / width)
    new_height, new_width = int(height*ratio), int(width*ratio)
    windows = np.zeros((4,), dtype=np.float32)
    pad_h = size[0] - new_height
    pad_w = size[1] - new_width
    pre_pad_h = pad_h // 2
    pre_pad_w = pad_w // 2
    post_pad_h = pad_h - pre_pad_h
    post_pad_w = pad_w - pre_pad_w
    windows_h = pre_pad_h + new_height
    windows_w = pre_pad_w + new_width
    # rescale the images
    image = cv2.resize(image, (new_width, new_height))
    new_bbox[:] *= ratio

    # padding the images
    windows[:] = (pre_pad_h, pre_pad_w, windows_h, windows_w)
    image = np.pad(image, [[pre_pad_h, post_pad_h], [pre_pad_w, post_pad_w], [0, 0]],mode="constant")
    new_bbox[:, 0:4:2] += pre_pad_h
    new_bbox[:, 1:4:2] += pre_pad_w
    try:
        new_bbox[:, 2] = np.minimum(new_bbox[:,2], size[0]-1)
        new_bbox[:, 3] = np.minimum(new_bbox[:,3], size[1]-1)
    except:
        pass
    return image, new_bbox, ratio, windows


def resize_image(image, size):
    height, width = image.shape[0:2]
    ratio = min(size[0] / height, size[1] / width)
    new_height, new_width = int(height*ratio), int(width*ratio)
    windows = np.zeros((4,), dtype=np.float32)
    pad_h = size[0] - new_height
    pad_w = size[1] - new_width
    pre_pad_h = pad_h // 2
    pre_pad_w = pad_w // 2
    post_pad_h = pad_h - pre_pad_h
    post_pad_w = pad_w - pre_pad_w
    windows_h = pre_pad_h + new_height
    windows_w = pre_pad_w + new_width
    # rescale the images
    image = cv2.resize(image, (new_width, new_height))

    # padding the images
    windows[:] = (pre_pad_h, pre_pad_w, windows_h, windows_w)
    image = np.pad(image, [[pre_pad_h, post_pad_h], [pre_pad_w, post_pad_w], [0, 0]],mode="constant")
    return image, ratio, windows

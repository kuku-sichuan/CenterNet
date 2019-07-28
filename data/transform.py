import numpy as np


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    assert masked_heatmap.shape == masked_gaussian.shape, print(center)
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def np_draw_gaussian(bbox, label, config):
    tl_heatmaps = np.zeros((config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1], config.CLASSES), dtype=np.float32)
    br_heatmaps = np.zeros((config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1], config.CLASSES), dtype=np.float32)
    ct_heatmaps = np.zeros((config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1], config.CLASSES), dtype=np.float32)
    tl_regrs = np.zeros((config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1], 2), dtype=np.float32)
    br_regrs = np.zeros((config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1], 2), dtype=np.float32)
    ct_regrs = np.zeros((config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1], 2), dtype=np.float32)
    mask = np.zeros((3, config.OUTPUT_SIZE[0], config.OUTPUT_SIZE[1]), dtype=np.float32)
    tag_mask = np.zeros((config.MAX_NUMS,), dtype=np.float32)
    tl_tags = np.zeros((config.MAX_NUMS,), dtype=np.int64)
    br_tags = np.zeros((config.MAX_NUMS,), dtype=np.int64)
    tag_len = 0
    for ind, box in enumerate(bbox):
                category = int(label[ind]) - 1
                box = box/config.OUTPUT_STRIDE
                fytl, fxtl = box[0], box[1]
                fybr, fxbr = box[2], box[3]
                fyct, fxct = (box[2] + box[0])/2., (box[3]+box[1])/2.
                xtl, ytl, xbr, ybr, xct, yct = int(fxtl), int(fytl), int(fxbr), int(fybr), int(fxct), int(fyct)
                if config.GAUSSIAN_BUMP:
                    height  = box[2] - box[0]
                    width = box[3] - box[1]

                    if config.RADIUS == -1:
                        radius = gaussian_radius((height, width), config.MINI_IOU)
                        radius = max(0, int(radius))
                    else:
                        radius = config.RADIUS

                    draw_gaussian(tl_heatmaps[:,:,category], [xtl, ytl], radius)
                    draw_gaussian(br_heatmaps[:,:,category], [xbr, ybr], radius)
                    draw_gaussian(ct_heatmaps[:,:,category], [xct, yct], radius, delte = 5)

                else:
                    tl_heatmaps[ytl, xtl, category] = 1
                    br_heatmaps[ybr, xbr, category] = 1
                    ct_heatmaps[yct, xct, category] = 1
                mask[0, ytl, xtl] = 1
                mask[1, ybr, xbr] = 1
                mask[2, yct, xct] = 1
                tl_tags[tag_len] = ytl*config.OUTPUT_SIZE[1] + xtl
                br_tags[tag_len] = ybr*config.OUTPUT_SIZE[1] + xbr
                tag_mask[tag_len] = 1
                tag_len += 1

                tl_regrs[ytl, xtl, :]  = [fxtl - xtl, fytl - ytl]
                br_regrs[ybr, xbr, :]  = [fxbr - xbr, fybr - ybr]
                ct_regrs[yct, xct, :]  = [fxct - xct, fyct - yct]

    return tl_heatmaps, br_heatmaps, ct_heatmaps, tl_regrs, br_regrs, ct_regrs, mask, tag_mask, tl_tags, br_tags

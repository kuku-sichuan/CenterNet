import numpy as np
import torch

def topk(scores, K=20):
    scores = np.transpose(scores, (0,3,1,2))
    scores = torch.from_numpy(scores)
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    topk_scores = topk_scores.numpy()
    topk_inds = topk_inds.numpy()
    topk_clses = topk_clses.numpy()
    topk_ys = topk_ys.numpy()
    topk_xs = topk_xs.numpy()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def tranpose_and_gather_feat(x, ind):
    batch, height, width, chans = x.shape
    x = np.reshape(x, (batch, -1, chans))
    x = torch.from_numpy(x)
    ind = torch.from_numpy(ind)
    x = _gather_feat(x, ind)
    x = x.numpy()
    return x


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def heatmap_detections(out, config):
    tl_heatmap, br_heatmap, ct_heatmap, tl_tag, br_tag, tl_reg, br_reg, ct_reg = out
    batch, height, width, cat = tl_heatmap.shape
    scores_tl, inds_tl, clses_tl, ys_tl, xs_tl = topk(tl_heatmap, config.TOP_K)
    scores_br, inds_br, clses_br, ys_br, xs_br = topk(br_heatmap, config.TOP_K)
    scores_ct, inds_ct, clses_ct, ys_ct, xs_ct = topk(ct_heatmap, config.TOP_K)


    ys_tl = np.tile(np.reshape(ys_tl, (batch, config.TOP_K, 1)),(1, 1, config.TOP_K))
    xs_tl =  np.tile(np.reshape(xs_tl, (batch, config.TOP_K, 1)),(1, 1, config.TOP_K))
    ys_br = np.tile(np.reshape(ys_br, (batch, 1, config.TOP_K)), (1, config.TOP_K, 1))
    xs_br = np.tile(np.reshape(xs_br, (batch, 1, config.TOP_K)), (1, config.TOP_K, 1))
    ys_ct = np.tile(np.reshape(ys_ct, (batch, 1, config.TOP_K)), (1, config.TOP_K, 1))
    xs_ct = np.tile(np.reshape(xs_ct, (batch, 1, config.TOP_K)), (1, config.TOP_K, 1))

    tl_regr = tranpose_and_gather_feat(tl_reg, inds_tl)
    tl_regr = np.reshape(tl_regr, (batch, config.TOP_K, 1, 2))
    br_regr = tranpose_and_gather_feat(br_reg, inds_br)
    br_regr = np.reshape(br_regr, (batch, config.TOP_K, 1, 2))
    ct_regr = tranpose_and_gather_feat(ct_reg, inds_br)
    br_regr = np.reshape(br_regr, (batch, config.TOP_K, 1, 2))

    xs_tl = xs_tl + tl_regr[..., 0]
    ys_tl = ys_tl + tl_regr[..., 1]
    xs_br = xs_br + br_regr[..., 0]
    ys_br = ys_br + br_regr[..., 1]
    xs_ct = xs_ct + ct_regr[..., 0]
    ys_ct = ys_ct + ct_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((xs_tl, ys_tl, xs_br, ys_br), dim=3)

    tl_tag = tranpose_and_gather_feat(tl_tag, inds_tl)
    tl_tag = np.reshape(tl_tag, (batch, config.TOP_K, 1))
    br_tag = tranpose_and_gather_feat(br_tag, inds_br)
    br_tag = np.reshape(br_tag, (batch, 1, config.TOP_K))
    dists = np.abs(tl_tag - br_tag)

    scores_tl = np.tile(np.reshape(scores_tl, (batch, config.TOP_K, 1)),(1, 1, config.TOP_K))
    scores_br = np.tile(np.reshape(scores_br, (batch, 1, config.TOP_K)), (1, config.TOP_K, 1))
    scores = (scores_tl + scores_br) / 2

    # reject boxes based on classes
    clses_tl = np.tile(np.reshape(clses_tl, (batch, config.TOP_K, 1)),(1, 1, config.TOP_K))
    clses_br = np.tile(np.reshape(clses_br, (batch, 1, config.TOP_K)), (1, config.TOP_K, 1))
    cls_inds = (clses_tl != clses_br)

    # reject boxes based on distances
    dist_inds = (dists > config.AE_THRESHOLD)

    # reject boxes based on widths and heights
    width_inds = (xs_br < xs_tl)
    height_inds = (ys_br < ys_tl)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    # scores is (batch, K, K)
    scores = torch.from_numpy(scores)
    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, config.MAX_NUMS)
    scores = scores.unsqueeze(2)
    scores = scores.numpy()

    bboxes = torch.from_numpy(bboxes)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)
    bboxes = bboxes.numpy()

    clses_tl = torch.from_numpy(clses_tl)
    clses = clses_tl.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()
    clses = clses.numpy()

    xs_ct = xs_ct[:, 0, :]
    ys_ct = ys_ct[:, 0, :]

    center = np.concatenate([np.expand_dims(xs_ct, 2), np.expand_dims(ys_ct, 2), np.expand_dims(clses_ct, 2),
                             np.expand_dims(scores_ct, 2)], 2)

    detections = np.concatenate([bboxes, scores, clses], 2)
    return detections, center


def remove_by_center_iiem(detections, centers, config):
    result = []
    for i in range(config.MAX_NUMS):
        detection = detections[i]
        x_size = detection[2] -detection[0]
        y_size = detection[3] -detection[1]
        n = 3
        if np.sqrt(x_size*y_size)> 150:
            n = 5
        tlx = (detection[2] +detection[0])-x_size/(2*n)
        tly = (detection[3] +detection[1])-y_size/(2*n)
        brx = (detection[2] +detection[0])+x_size/(2*n)
        bry = (detection[3] +detection[1])+y_size/(2*n)
        proposal_center = centers[centers[:, 2] == detection[7]]
        if proposal_center.shape[0]==0:
            continue
        else:
            x_interval = np.logical_and(proposal_center[:, 0] > tlx, proposal_center[:, 0] < brx)
            y_interval = np.logical_and(proposal_center[:, 1] > tly, proposal_center[:, 1] < bry)
            interval = np.logical_and(x_interval, y_interval)
            proposal_center = proposal_center[interval]
        if proposal_center.shape[0] == 0:
            continue
        else:
            sort_index = np.argsort(proposal_center[:, 3])
            proposal_center = proposal_center[sort_index]
            detection[4] = (detection[4]*2 + proposal_center[0][3])/3
            result.append(detection)
    result = np.stack(result, 0)
    result = result[0:config.MAX_NUMS]
    result = np.pad(result, [[0, config.MAX_NUMS-result.shape[0], [0, 0]]], mode="constant")
    return result


def remove_by_center(detections, centers, config):
    result = []
    for i in range(detections.shape[0]):
        final_detect = remove_by_center_iiem(detections[i], centers[i], config)
        result.append(final_detect)
    result = np.stack(result, 0)
    return result


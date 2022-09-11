import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.ops.boxes import box_area

from scipy.optimize import linear_sum_assignment
import sys

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = box_cxcywh_to_xyxy(outputs['head_loc_pred'][idx])
        target_boxes = torch.cat([targets['head_loc_gt'][b_idx] for b_idx, (_, i) in enumerate(indices)], dim=0)
        # print(src_boxes)
        # print(target_boxes)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_is_head(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_is_head = outputs['is_head_pred'][idx]
        target_is_head = torch.cat([targets['is_head_gt'][b_idx] for b_idx, (_, i) in enumerate(indices)], dim=0)
        target_is_head = target_is_head.flatten().long()

        loss_is_head = F.cross_entropy(src_is_head, target_is_head, reduction='none')

        losses = {}
        losses['loss_is_head'] = loss_is_head.sum() / num_boxes

        return losses

    def loss_watch_outside(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_watch_outside = outputs['watch_outside_pred'][idx]
        target_watch_outside = torch.cat([targets['watch_outside_gt'][b_idx] for b_idx, (_, i) in enumerate(indices)], dim=0)
        target_watch_outside = target_watch_outside.flatten().long()

        loss_watch_outside = F.cross_entropy(src_watch_outside, target_watch_outside, reduction='none')

        losses = {}
        losses['loss_watch_outside'] = loss_watch_outside.sum() / num_boxes

        return losses

    def loss_gaze_map(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_gaze_heatmap = outputs['gaze_heatmap_pred'][idx]
        target_gaze_heatmap = torch.cat([targets['gaze_heatmap_gt'][b_idx] for b_idx, (_, i) in enumerate(indices)], dim=0)
        _, grid_num = target_gaze_heatmap.shape

        loss_gaze_map = F.mse_loss(src_gaze_heatmap, target_gaze_heatmap, reduction='sum')

        losses = {}
        losses['loss_gaze_map'] = loss_gaze_map.sum() / (num_boxes * grid_num)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'is_head': self.loss_is_head,
            'watch_outside': self.loss_watch_outside,
            'gaze_map': self.loss_gaze_map,

        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        bs, num_gt = targets["head_loc_gt"].shape[:2]
        num_boxes = bs*num_gt
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.alpha1 = 1.0
        self.alpha2 = 2.5
        self.beta1 = 2.0
        self.beta2 = 1.0
        self.beta3 = 1.0
        self.beta4 = 2.0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["head_loc_pred"].shape[:2]
        bs, num_gt = targets["head_loc_gt"].shape[:2]

        # get predictions
        head_loc_pred = outputs["head_loc_pred"]
        gaze_heatmap_pred = outputs["gaze_heatmap_pred"]
        is_head_pred = outputs["is_head_pred"]
        watch_outside_pred = outputs["watch_outside_pred"]

        # get ground-truth
        head_loc_gt = targets["head_loc_gt"]
        gaze_heatmap_gt = targets["gaze_heatmap_gt"]
        is_head_gt = targets["is_head_gt"]
        watch_outside_gt = targets["watch_outside_gt"]

        # Compute the cost between is head
        is_head_pred = is_head_pred.view(bs*num_queries, 2)
        is_head_gt = is_head_gt.view(bs*num_gt).long()
        cost_is_head = -is_head_pred[:, is_head_gt]

        # Compute the cost between is watch outside
        watch_outside_pred = watch_outside_pred.view(bs*num_queries, 2)
        watch_outside_gt = watch_outside_gt.view(bs*num_gt).long()
        cost_watch = -watch_outside_pred[:, watch_outside_gt]

        # Compute the L1 cost between boxes
        out_bbox = head_loc_pred.flatten(0, 1).float()
        tgt_bbox = head_loc_gt.flatten(0, 1).float()
        cost_bbox_l1 = torch.cdist(box_cxcywh_to_xyxy(out_bbox), tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), tgt_bbox)

        cost_bbox = self.alpha1 * cost_bbox_l1 + self.alpha2 * cost_giou

        # Compute the L2 cost between gaze maps
        gaze_heatmap_pred = gaze_heatmap_pred.flatten(0, 1)
        gaze_heatmap_gt = gaze_heatmap_gt.flatten(0, 1)
        cost_gaze_map = torch.cdist(gaze_heatmap_pred, gaze_heatmap_gt, p=2)

        # Final cost matrix
        C = self.beta1 * cost_bbox + self.beta2 * cost_is_head + self.beta3 * cost_watch + self.beta4 * cost_gaze_map
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [num_gt for _ in range(bs)]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class=1, set_cost_bbox=1, set_cost_giou=1):
    return HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union
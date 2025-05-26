import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range    # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range  # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes. 解码一个批次中单个样本的预测结果
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # 输入的是最后一层解码器对一张图像的预测类别得分、边界框预测
        max_num = self.max_num  # 配置文件中max_num=300,  最终输出框的数量上限

        cls_scores = cls_scores.sigmoid()
        # ls_scores.view(-1) 将 (num_query, num_classes) 展平为1D张量；
        # .topk(max_num) 获取前 'max_num' 个得分及其在展平张量中的原始索引
        # 原先900个query，10个类别，就会变成9000个元素的张量，选置信度最大的300个
        scores, indexs = cls_scores.view(-1).topk(max_num)  # scores 将包含最高的 max_num 个置信度值，而 indexs 将告诉我们它们来自哪里
        # 取余运算，得到300个高置信度的标签
        labels = indexs % self.num_classes
        # 取整除法，得到300个高置信度的对应的query
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        # 此时的 bbox_preds 是 (max_num, code_size)，例如 (300, 10)
        # 其格式为 (cx_pred, cy_pred, w_pred, l_pred, cz_pred, h_pred, sin_pred, cos_pred, vx_pred, vy_pred)

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        # denormalize_bbox 将网络的输出格式转换回标准的3D框参数。
        # 输入: (cx, cy, w, l, cz, h, sin, cos, vx, vy) (其中w,l,h是直接预测，不是log)
        # 输出: (cx, cy, cz, w_exp, l_exp, h_exp, atan2(sin,cos), vx, vy)
        #   - cx, cy, cz 在 Detr3DHead 的 forward 中已经按 pc_range 缩放，所以如果它们已经是世界尺度，denormalize_bbox 会按原样使用它们。
        #     或者，如果它们通过 inverse_sigmoid(reference_points) + offset 被归一化到 [0,1]，
        #     那么 denormalize_bbox 会使用 pc_range 来缩放它们。
        #     鉴于 Detr3DHead 的 forward 逻辑：
        #       tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        #       tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        #       tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        #     输入到 denormalize_bbox 的 cx, cy, cz 已经是世界尺度。
        #   - w, l, h 进行指数运算 (w.exp(), l.exp(), h.exp())，因为损失目标是 log(维度)。
        #   - 旋转角 (yaw) 使用 atan2(sin, cos) 恢复。
        #   - vx, vy 按原样使用。
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        '''输入的preds_dicts是
        preds_dicts = {
            'all_cls_scores': outputs_classes, # 每一层预测的分类概率 
            'all_bbox_preds': outputs_coords,  # 每一层预测的回归
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        '''
        all_cls_scores = preds_dicts['all_cls_scores'][-1]  # 仅使用最后一个解码器层的得分 
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]  # 仅使用最后一个解码器层的边界框预测
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            '''decode_single返回值：
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }
            '''
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list
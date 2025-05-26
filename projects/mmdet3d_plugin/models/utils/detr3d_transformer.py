
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    将已经经过 Sigmoid 处理的值还原回其原始的、未压缩的范围
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)  # mlvl_feats  [[B, 6, 256, 116, 200] [B, 6, 256, 58, 100] [B, 6, 256, 29, 50] [B, 6, 256, 15, 25]]
        query_pos, query = torch.split(query_embed, self.embed_dims , dim=1) #  (num_query, self.embed_dims)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)   # (bs, num_query, embed_dims)
        query = query.unsqueeze(0).expand(bs, -1, -1)   # (bs, num_query, embed_dims)
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        reference_points = self.reference_points(query_pos)     # (bs, num_query, 3) 初始化object quer的3D位置，900个object query都有了各自对应的一个初始(x,y,z)的坐标猜测
        reference_points = reference_points.sigmoid()  # 对初始(x,y,z)坐标的值压缩到(0,1)之间--为归一化后的坐标
        init_reference_out = reference_points

        # decoder
        # Transformer（特别是默认的 nn.Transformer）通常期望输入的格式是
        # (sequence_length, batch_size, embedding_dimension)
        # 在这里，num_query (Object Query的数量) 就是我们的 sequence_length (序列长度)。
        query = query.permute(1, 0, 2)  # (bs, num_query, embed_dims) -> (num_query, bs, embed_dims)
        query_pos = query_pos.permute(1, 0, 2)  
        '''
            配置文件中的 type='Detr3DTransformerDecoder', 对应projects/mmdet3d_plugin/models/utils/detr3d_transformer.py中的Detr3DTransformerDecoder类
            inter_states 包括了每层 decode r更新之后的object query（形状为 (num_query, bs, embed_dims)） 
            inter_references 是包含了每一层 decode 迭代优化后的3D参考点。每一层的输出参考点都会作为下一层的输入参考点（的起点）（形状为 (bs, num_query, 3)，归一化）
        '''
        inter_states, inter_references = self.decoder(
            query=query,    # (num_query, bs, embed_dims)
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,  # 初入的是初始3D参考点，形状为 (bs, num_query, 3)，且归一化了，后面会不断经过每一层的decoder得到细化结果
            reg_branches=reg_branches,
            **kwargs)
        
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    This decoder is a stack of `DetrTransformerDecoderLayer`.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate  # 配置文件中 return_intermediate=True

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        # self.layers 是一个 nn.ModuleList，包含了多个解码器层实例 (比如6个DetrTransformerDecoderLayer)
        for lid, layer in enumerate(self.layers):    # lid 层索引（0-5），layer是当前解码器的层，一层decoder包括：('self_attn', 'norm', 'cross_attn', 'norm','ffn', 'norm')
            reference_points_input = reference_points   # 第0层decoder的时候是MLP随机初始化query_pos的3D参考点；第1-4层就是上一层decoder更新参数之后学习到的结果了。
            
            # 得到的 output 包括了：多层特征图上加权之后的值 + 初始的query(残差) + 3D参考点的空间位置信息(未归一化的)
            # (bs, num_query, embed_dims)
            output = layer(
                output,     # 就是query，要经过6层decoder迭代，学到了多层特征图上的视觉特征值+3D参考点的绝对空间位置+初始query用于残差
                *args,
                reference_points=reference_points_input,    # 第0层decoder的时候是MLP随机初始化query_pos的3D参考点；第1-4层就是上一层decoder更新参数之后学习到的结果了。
                **kwargs)
            
            # 为了让接下来的 reg_branches 解码，调整out（query）的shape（要输入MLP了）
            output = output.permute(1, 0, 2)    # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)     # 获取每一层的边界框的偏移量， (bs, num_query, 10个维度)
                
                assert reference_points.shape[-1] == 3  # reference_points (bs, num_query, 3)

                new_reference_points = torch.zeros_like(reference_points)
                '''
                    更新3D参考点的坐标
                    目得：tmp = reg_branches[lid](output) 得到的是预测的偏移量，需要加上当前层的初始坐标位置
                '''
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2]) # 预测x,y的偏移量加上当前层初始参考点
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3]) # reference_points (bs, num_query, 3)，但是预测的偏移量是10维的
                
                new_reference_points = new_reference_points.sigmoid()   # 归一化

                reference_points = new_reference_points.detach()    

            output = output.permute(1, 0, 2)    # 还要经过下一层，因此调整回transformer期望的格式 (num_query, bs, embed_dims)
            if self.return_intermediate:    # 配置文件中 return_intermediate=True
                intermediate.append(output)         # 记录每层的query
                intermediate_reference_points.append(reference_points)  # 记录每层的预测的3D参考点

        if self.return_intermediate:
              # 把每层decoder更新参数之后的 query 和 
              # 根据上层更新参数之后的参考点（最初始的、未经任何解码器层处理的query和参考点，
              # 在 Detr3DTransformer 中被命名为 init_reference_out 并独立返回。）
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)    

        return output, reference_points # 配置文件中 return_intermediate=True,因此不返回这个  output就是经过6层解码器更新之后的object query； reference_points是迭代更新之后的，最后一层解码器输出的3D参考点（形状为 (bs, num_query, 3)，归一化）。


@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    让 3D 世界中的物体查询 (object queries) 能够有效地
    从多个摄像机视图、多个不同尺度的 2D 图像特征中聚合信息。
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,  # 每张图像的特征层个数，就是FPN采样了4个层级
                 num_points=5,  # 每个相机的每个特征层上采样的点的数量；配置文件中实际上是 1
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None, # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads  # 32
        self.norm_cfg = norm_cfg    # None，forward中没用到
        self.init_cfg = init_cfg    # None，forward中没用到
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range    #  # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step  # 64
        self.embed_dims = embed_dims    # 256
        self.num_levels = num_levels    # 4
        self.num_heads = num_heads      # 8
        self.num_points = num_points    # 配置文件中实际上是 1
        self.num_cams = num_cams        # 6

        '''
            # 关键层1: 学习采样点的注意力权重
            输入 是 query (embed_dims)，输出维度 是 num_cams * num_levels * num_points
            为每个query ，针对所有相机、每个相机的所有特征层级、每个特征层级的所有采样点都学习一个权重（也就是“重要性”得分attention_weights），这个得分后续会经过 Sigmoid函数，变成类似注意力权重的数值
        '''
        self.attention_weights = nn.Linear(embed_dims,num_cams*num_levels*num_points)   # (256,6*4*1)

        # 关键层2: 输出投影层
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        '''
            # 关键层3: 3D 参考点的位置编码器 (一个小型的 MLP)
            # 输入是 3D参考点坐标 (x,y,z)，输出是 embed_dims 维的特征 256
            用于将3D参考点的坐标本身编码成一个特征向量，这个特征向量最终会加到注意力模块的输出上，为模型提供关于 query 绝对空间位置信息。
        '''
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.batch_first = batch_first  # None，用不上

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # attention_weights 初始化为0，这意味着在训练初期，
        # 所有采样点的初始权重倾向于均等（ 经过Sigmoid后是0.5，乘以mask后需要看mask情况 ）
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,      # (num_query, bs, embed_dims)
                key,        # 在交叉注意力中通常是图像特征, 此处由 value (mlvl_feats) 代替
                value,      # 多层级多视图图像特征 ---FPN输出的多层特征图(mlvl_feats), 列表形式
                residual=None,      # 残差连接的输入，通常是 query
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,      # 初始的3D参考点，形状为 (bs, num_query, 3)，且归一化了
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos   # 这个就是完整的 3D Object Query了，将位置编码加到 query 上了

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)  #  (num_query, bs, embed_dims) ->  (bs, num_query, embed_dims)

        bs, num_query, _ = query.size()

        '''
        self.attention_weights = nn.Linear(embed_dims,num_cams*num_levels*num_points)  (256,6*4*1)
        query (bs, num_query, embed_dims) 
        self.attention_weights = nn.Linear(embed_dims,num_cams*num_levels*num_points)
        经过 attention_weights 线性变换后，变成了 (bs, num_query, num_cams * num_levels * num_points)
        又reshape成: (bs, 1, num_query, num_cams, num_points, num_levels)
            让模型为每一个 object query 动态地学习
            如何给来自不同相机、不同特征层级、不同采样点的视觉信息分配重要性（即权重）
        '''
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels) #  (bs, 1, num_query, num_cams, num_points, num_levels)
        
        '''
            reference_points_3d: 这里应该是经过多层解码器之后的归一化的3D参考点了，学习到位置信息了，(B, num_query, 3)
            output(sampled_feats): 特征层堆叠后的采样到的图像特征，(B, C, num_query, num_cams, num_points=1, num_levels)
            mask: 有效性掩码，(B, 1, num_query, num_cams, num_points=1, num_levels=1)
        '''
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)   #  (B, C, num_query, num_cam, 1采样点, num_lvl)
        mask = torch.nan_to_num(mask)


        '''
            原始权重值会经过 Sigmoid 函数，将它们的值域压缩到 (0, 1) 之间，形成标准的注意力权重。
            然后乘以 有效的mask，剔除无效采样点(投影到相机后方的、超出图像范围的)权重就为0
            attention_weights 原始值由 self.attention_weights(query) 生成，形状 (bs, 1, num_query, num_cams, num_points, num_levels)
            mask 形状 (bs, 1, num_query, num_cams, num_points, num_levels)
        '''
        attention_weights = attention_weights.sigmoid() * mask

        '''
            加权特征采样：
                output (采样特征) 形状: (bs, C, num_query, num_cams, num_points, num_levels)
                attention_weights (注意力得分) 形状: (bs, 1, num_query, num_cams, num_points, num_levels)
            每个output采样特征都乘以其对应的注意力得分
        '''
        output = output * attention_weights

        '''
            聚合所有加权后的采样特征：
            当前 output 形状: (bs, C, num_query, num_cams, num_points, num_levels)
            下一步需要为每个物体查询 (num_query 中的一个) 得到一个单一的、聚合了所有信息的 C 维特征向量
            因此，需要在 num_levels, num_points, num_cams 这三个维度上进行求和。
            .sum(-1).sum(-1).sum(-1)经过三步求和，对于每个 query，所有来自不同相机、不同层级、不同采样点的加权特征信息
            都被聚合成了一个维度为 C 的特征向量
           当前 output 形状: (bs, C, num_query, num_cams, num_points, num_levels)
           新的 output (B, C, num_query)
        '''
        output = output.sum(-1).sum(-1).sum(-1) # (B, C, num_query)
        # (bs, C, num_query) -> (num_query, bs, C)  (C 即 embed_dims)
        output = output.permute(2, 0, 1)

        # self.output_proj = nn.Linear(embed_dims, embed_dims)
        output = self.output_proj(output)   # (num_query, bs, embed_dims)
        
        # self.position_encoder MLP网络，输出维度embed_dims 把reference_points_3d调整成(bs, num_query, embed_dims)
        # reference_points_3d 这里应该是经过学习之后的归一化的3D参考点了，学习到位置信息了，(B, num_query, 3)
        # reference_points_3d会在每一层decoder中更新，
        # 它提供了与采样到的视觉特征互补的、关于查询点自身绝对3D位置的明确编码
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        # inp_residual 残差连接--query本身的初始状态
        return self.dropout(output) + inp_residual + pos_feat # 相当于是学习到的图像特征(加权了) + 位置信息


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    '''
        mlvl_feats: FPN 输出的多层特征图(mlvl_feats)
        reference_points: 这里应该是经过self-att之后的归一化的3D参考点了，学习到位置信息了，(B, num_query, 3)
        pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],将归一化的 reference_points 转换到真实的物理坐标
    '''
    # --- 步骤 1: 准备激光雷达到图像的投影矩阵 ---
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)       # 将列表转换为numpy数组
    lidar2img = reference_points.new_tensor(lidar2img)      # 转换为PyTorch张量，并确保与reference_points在同一设备上
   
    # --- 步骤 2: 准备3D参考点 ---
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()  # 这里应该是经过self-att之后的归一化的3D参考点了，学习到位置信息了，(B, num_query, 3)

    # --- 步骤 3: 将归一化的3D参考点反归一化到真实的激光雷达坐标系 ---
    # reference_points 的 x, y, z 分量原本在 [0,1] 区间
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # 现在 reference_points 中的坐标是真实的物理坐标了

    # --- 步骤 4: 将3D参考点转换为齐次坐标 (x, y, z, 1) ---
    # 为了进行矩阵乘法（投影），需要将3D点表示为4D齐次坐标
    # reference_points (B, num_queries, 4) 都是为每个3D参考点和每个相机的投影矩阵做好匹配
    # 齐次坐标 (x,y,z,1)，方便进行4x4矩阵的乘法运算
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)

    # --- 步骤 5: 将激光雷达坐标系下的3D点投影到各个相机视图的图像坐标系 ---
    # a.  为了让真实世界的3D参考点 与 lidar2img 相乘，变换（拓展）一下维度
    # reference_points 变为 (B, 1, num_query, 4) -> (B, num_cam, num_query, 4, 1)
    # 增加末尾维度 (B, num_cam, num_query, 4, 1) # 准备进行矩阵乘法 (4x4) @ (4x1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # 原始 (B, num_cam, 4, 4) -> (B, num_cam, 1, 4, 4) ->   (B, num_cam, num_query, 4, 4) 
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    # b. 投影！ reference_points(B, num_cam, num_query, 4, 4) @ lidar2img(B, num_cam, num_query, 4, 1)
    # 核心投影步骤：使用 lidar2img 矩阵将3D参考点从LiDAR坐标系转换到相机图像坐标系
     # 结果 reference_points_cam 形状 (B, num_cam, num_query, 4, 1)
     # 然后 squeeze(-1) 变为 (B, num_cam, num_query, 4)
     # 最后一维是投影后的齐次坐标 [x*depth, y*depth, depth, homogeneous_w=1]
     # 3D参考点在相机图像齐次坐标空间中的表示，它还没有完成到最终2D像素坐标的转换。并不是图像坐标系！
     # 这个结果中包含了深度信息！可以知道是否投影到相机前方！
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)    # (B, num_cam, num_query, 4)

    # --- 步骤 6: 计算初步的有效性掩码 (mask) ---
    eps = 1e-5
    # reference_points_cam (B, num_cam, num_query, 4)  最后一个维度的4代表：[x*depth, y*depth, depth, homogeneous_w=1]
    mask = (reference_points_cam[..., 2:3] > eps)   # 标记一下在相机前方的有效点
    
    # --- 步骤 7: 执行透视除法，得到2D像素坐标 (u, v) ---
     # 透视除法：将图像坐标的x,y分量除以深度z，得到归一化的图像平面坐标 (u,v)
     # 最后 reference_points_cam 的形状是 (B, num_cam, num_query, 2)，存储的是 (u,v) 像素坐标
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    
    # --- 步骤 8: 将像素坐标 (u,v) 归一化到图像的 [0,1] x [0,1] 范围 ---
    # 这里隐含了一个假设：对于一个batch内的所有样本，图像被处理到一个统一的标准
    # 最后 reference_points_cam(B, num_cam, num_query, 2) 中的 (u,v) 坐标在 [0,1] 区间 (如果点在图像内的话)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]  # u /= width
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]  # v /= height

    # --- 步骤 9: 将归一化到 [0,1] 的坐标转换为 F.grid_sample 所需的 [-1,1] x [-1,1] 范围 ---
    # torch.nn.functional.grid_sample 要求采样坐标在 [-1, 1] 范围内。
    # (-1,-1) 代表图像左上角，(1,1) 代表右下角
    #    变换公式: x_new = (x_old - 0.5) * 2
    reference_points_cam = (reference_points_cam - 0.5) * 2  
    
    # --- 步骤 10: 更新掩码 (mask)，确保投影点在图像的 [-1,1] x [-1,1] 有效范围内 ---
    # 结合之前 mask 标记了在相机前方的有效点  +  在图像范围内的点
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    
    # --- 步骤 11: 调整 mask 的维度以匹配后续 attention_weights 的广播需求 ---
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)   # 将mask中的NaN值替换为0，确保是有效的布尔值 

    # --- 步骤 12: 遍历多层级特征 (mlvl_feats)，并进行特征采样 ---
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        # F.grid_sample 要求输入特征的形状是 (Batch_total, C, H, W)
        # 所以将 B 和 N (num_cams) 合并为一个维度
        feat = feat.view(B*N, C, H, W)
        # reference_points_cam (B, num_cam, num_query, 2) 
        # F.grid_sample 要求采样点坐标的形状是 (Batch_total, H_out, W_out, 2)
        # 在这里，为每个 query 采样 num_points=1 个点
        # reference_points_cam 形状 (B, N, num_query, 2) -> (B*N, num_query, 1, 2)
        # 中间的 '1' 对应于 W_out=1 (因为只为每个参考点采一个点，而不是一个网格)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        # sampled_feat (B*N_cam, C, num_query, 1)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)    # 双线性插值算出来该2D位置点的视觉特征值
        # 调整采样后特征的形状和维度顺序 (B, C, num_query, N_cam, 1)
        # (B*N, C, num_query, 1) -> (B, N, C, num_query, 1) -> (B, C, num_query, N, 1)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)

    # --- 步骤 13: 将所有层级采样到的特征堆叠起来 ---
    # sampled_feats 是一个列表，其中每个元素是对应层级采样到的特征，形状 (B, C, num_query, num_cams, num_points)
    # torch.stack(sampled_feats, -1) 会在最后一个维度上堆叠，
    # 结果形状为 (B, C, num_query, num_cams, num_points, num_levels)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask

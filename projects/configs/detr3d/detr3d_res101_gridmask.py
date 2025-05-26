_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='Detr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),    # 归一化层的配置  requires_grad=False 表示这个归一化层的参数在训练时也不更新。
        norm_eval=True,
        style='caffe',
        # 后两个阶段使用DCN（可变性卷积）
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,   # 从第二层特征图开始，每次的输入通道数，输出256的通道数
        add_extra_convs='on_output',    # 因为最后要输出4个特征层，但是从第一层开始，第0层舍弃，因此要在最后一层特征层上再通过一层卷积下采样得到新的一层（原图的1/64）特征图
        num_outs=4, 
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='Detr3DHead',
        num_query=900,
        num_classes=10,
        in_channels=256,    # 输入到“Detr3DHead”的通道数，就是img_neck的输出通道数。
        sync_cls_avg_factor=True,   # 分布式训练，同步各GPU上的分类损失，确保梯度一致
        with_box_refine=True,   # 也就是decoder每一层（总共6层）都要预测边界框的结果（中心点坐标）。
        as_two_stage=False,     # DETR系列通常都是单阶段检测
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,       # 每一层都要得到预测结果，当作辅助损失
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[     # 包含多个注意力机制的配置
                        dict(
                            type='MultiheadAttention',      # 自注意力机制
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',        # 交叉注意力
                            pc_range=point_cloud_range,  # 定义了要关注3D空间的范围
                            num_points=1,   # 表示3D Object Query在投影到2D特征图上时，会采样多少个点获取特征
                            embed_dims=256)
                    ],
                    feedforward_channels=512, # FFN前馈神经网络的中间层维度
                    ffn_dropout=0.1,
                    # TODO 定义了DetrTransformerDecoderLayer层中的操作顺序！！！
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        # DetrTransformerDecoderLayer更新了3D object query，然后每层都要将query转换成实际的3D bbox和类别
        bbox_coder=dict(
            type='NMSFreeCoder',  # 无需NMS
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 预测出来的3D bbox允许的范围，超出此范围就会被过滤掉
            pc_range=point_cloud_range,
            max_num=300,    #  最终输出框的数量上限
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,      # 位置编码的特征维度，
            normalize=True,     # 归一化
            offset=-0.5         # 偏移量，调整编码的中心
            ),   
        # 分类损失    
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,   # 输出层使用 sigmoid 函数，多标签分类
            gamma=2.0,      #  Focal Loss的参数，调节难易样本的权重
            alpha=0.25,     #  正负样本的权重
            loss_weight=2.0 # 分类损失在总损失的权重
            ),
        # 边界框回归损失 
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    # train_cfg定义了在Detr3DHead训练时的配置
        train_cfg=dict(
        pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        # assigner 将预测的结果与GT进行一一匹配
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')  # 从硬盘里读取文件

# Database Sampler 数据增强方法，存储了训练集中的GT，然后在训练一个新场景时，随机挑出一些物体，把他们粘贴到当前正在处理的场景中
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl', # 每个物体单独的信息
    rate=1.0,   # 对于每个训练样本，有多大概率会启动数据增强？（0-1之间的值）
    # 从GT中挑选物体
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    # 每个类别会采样并粘贴多少个示例
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7, # 预测比较困难，多放一些
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',     # 加载的点云坐标是基于LiDAR的
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),    # 输入图片的尺寸需要是32的整数倍
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),      # 测试图像要被调整到的尺寸
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]


data = dict(
    samples_per_gpu=1,  # 每个GPU加载样本的个数  
    workers_per_gpu=4,  # 每个GPU使用多少个子进程来并行加载数据
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,        # TODO! 这是训练模式，不是测试模式！！！
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'     # 3d的bbox 坐标是基于雷达坐标系的！！！
        ),
    val=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality))

optimizer = dict(
    type='AdamW', 
    lr=2e-4,    # 学习率
    # paramwise_cfg 允许对不同的参数（backbone、neck、bbox_head）设置不同的学习率
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # img_backbone的学习率是 lr=2e-4 的0.1倍 【因为resnet都是预训练权重初始化的，无需太大的学习率去调整】
        }),
    weight_decay=0.01   # 权重衰减系数，防止过拟合
    )
# optimizer_config 定义了优化器的一些额外配置（梯度裁剪、混合精度训练），
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 训练时grad_norm时裁剪前的梯度(如果这个数大于max_norm，就裁剪)L2 范数

# 学习率调整策略
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24

# 训练过程中如何进行评估
evaluation = dict(interval=2, pipeline=test_pipeline)   # 每隔 2个epoch进行一次评估

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/fcos3d.pth'

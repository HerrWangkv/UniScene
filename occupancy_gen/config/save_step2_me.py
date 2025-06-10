grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 200
return_len_ = 10

multisteplr = False
multisteplr_config = dict(
    decay_t=[87 * 500], decay_rate=0.1, warmup_t=warmup_iters, warmup_lr_init=1e-6, t_in_epochs=False
)


optimizer = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-3,  # 1e-3,
        weight_decay=0.01,
    ),
)

data_path = "data/"


train_dataset_config = dict(
    type="nuScenesSceneDatasetLidar",
    data_path=data_path,
    return_len=return_len_,
    offset=0,
    nusc_dataroot="data/nuscenes",
    imageset="data/nuscenes_infos_train_temporal_v3_scene.pkl",
)

val_dataset_config = dict(
    type="nuScenesSceneDatasetLidar",
    data_path=data_path,
    return_len=return_len_,
    offset=0,
    nusc_dataroot="data/nuscenes",
    imageset="data/nuscenes_infos_val_temporal_v3_scene.pkl",
)

train_wrapper_config = dict(
    type="tpvformer_dataset_nuscenes_step2_wobev",
    phase="train",
)

val_wrapper_config = dict(
    type="tpvformer_dataset_nuscenes_step2_wobev",
    phase="val",
)

train_loader = dict(
    batch_size=1,
    shuffle=False,
    num_workers=2,
)

val_loader = dict(
    batch_size=1,
    shuffle=False,
    num_workers=2,
)


loss = dict(
    type="MultiLoss",
    loss_cfgs=[
        dict(
            type="ReconLoss",
            weight=1.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={"logits": "logits", "labels": "inputs"},
        ),
        dict(type="LovaszLoss", weight=1.0, input_dict={"logits": "logits", "labels": "inputs"}),
        dict(type="KL_Loss", weight=1.0),
        # dict(
        #     type='VQVAEEmbedLoss',
        #     weight=1.0),
    ],
)

loss_input_convertion = dict(
    logits="logits",
    kl_loss="kl_loss"
    # embed_loss='embed_loss'
)


load_from = ""

_dim_ = 16
expansion = 8
# base_channel = 64
base_channel = 4
n_e_ = 512
ch_multi_rate = 16
num_res = 2
model = dict(
    type="VAERes2D_DwT",
    encoder_cfg=dict(
        type="Encoder2D_new2",
        ch=base_channel * ch_multi_rate,
        out_ch=base_channel * 2,  # useless
        ch_mult=(1, 2, 4),
        num_res_blocks=num_res,
        attn_resolutions=(50,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=_dim_ * expansion,
        resolution=200,
        z_channels=base_channel,
        double_z=True,
    ),
    decoder_cfg=dict(
        type="Decoder3D_withT",
        z_channels=base_channel,
        ch_mult=(4, 2, 1),
        n_hiddens=base_channel * ch_multi_rate,
        n_res_layers=num_res,
        upsample=(1, 4, 4),
    ),
    num_classes=18,
    expansion=expansion,
    # vqvae_cfg=dict(
    #     type='VectorQuantizer',
    #     n_e = n_e_,
    #     e_dim = 256,
    #     beta = 1.,
    #     z_channels = base_channel,
    #     use_voxel=True)
)

shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"

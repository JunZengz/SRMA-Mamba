import torch
import argparse
from configs.config import *
from model.SRMAMamba import SRMAMamba

def build_SRMAMamba():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='SRMAMamba')
    parser.add_argument('--pretrained_weight_path', type=str,
                        default=None)
    parser.add_argument('--cfg', type=str,
                        default='configs/vssm1/vmambav2_tiny_224.yaml')
    parser.add_argument("--opts",
                        help="Modify config options by adding 'KEY VALUE' pairs. ",
                        default=None,
                        nargs='+')
    opt = parser.parse_args()
    config = get_config(opt)
    model = eval(opt.model_name)(pretrained=opt.pretrained_weight_path,
                       # volume_shape=opt.volume_shape,
                       patch_size=config.MODEL.VSSM.PATCH_SIZE,
                       in_chans=config.MODEL.VSSM.IN_CHANS,
                       num_classes=config.MODEL.NUM_CLASSES,
                       depths=config.MODEL.VSSM.DEPTHS,
                       dims=config.MODEL.VSSM.EMBED_DIM,
                       # ===================
                       ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                       ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                       ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                       ssm_dt_rank=(
                           "auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                       ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                       ssm_conv=config.MODEL.VSSM.SSM_CONV,
                       ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                       ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                       ssm_init=config.MODEL.VSSM.SSM_INIT,
                       forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                       # ===================
                       mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                       mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                       mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                       # ===================
                       drop_path_rate=config.MODEL.DROP_PATH_RATE,
                       patch_norm=config.MODEL.VSSM.PATCH_NORM,
                       norm_layer=config.MODEL.VSSM.NORM_LAYER,
                       downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                       patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                       gmlp=config.MODEL.VSSM.GMLP,
                       use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                       )

    return model


if __name__ == '__main__':
    x = torch.randn(1, 1, 224, 224, 128).cuda()
    SRMAMamba = build_SRMAMamba().cuda()
    SRMAMamba.train()
    out = SRMAMamba(x)
    print(out.shape)
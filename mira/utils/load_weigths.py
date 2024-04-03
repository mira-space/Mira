import copy
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
from utils.utils import instantiate_from_config


def expand_conv_kernel(pretrained_dict):
    """ expand 2d conv parameters from 4D -> 5D """
    for k, v in pretrained_dict.items():
        if v.dim() == 4 and not k.startswith("first_stage_model"):
            v = v.unsqueeze(2)
            pretrained_dict[k] = v
    return pretrained_dict


def load_from_pretrainedSD_checkpoint(model, pretained_ckpt, expand_to_3d=True, adapt_keyname=False):
    mainlogger.info(f'------------------- Load pretrained SD weights -------------------')
    try:
        sd_state_dict = torch.load(pretained_ckpt, map_location=f"cpu")
    except:
        mainlogger.info(f'------------------- Load pretrained SafeTensor SD weights -------------------')
        from safetensors import safe_open
        sd_state_dict = {}
        with safe_open(pretained_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_state_dict[key] = f.get_tensor(key)
    if "state_dict" in list(sd_state_dict.keys()):
        sd_state_dict = sd_state_dict["state_dict"]
    model_state_dict = model.state_dict()
    ## delete ema_weights just for <precise param counting>
    for k in list(sd_state_dict.keys()):
        if k.startswith('model_ema'):
            del sd_state_dict[k]
        if k.startswith('model.diffusion'):
            del sd_state_dict[k]
    mainlogger.info(f'Num of parameters of target model: {len(model_state_dict.keys())}')
    mainlogger.info(f'Num of parameters of source model: {len(sd_state_dict.keys())}')

    if adapt_keyname:
        ## adapting to standard 2d network: modify the key name because of the add of temporal-attention
        mapping_dict = {
            'middle_block.2': 'middle_block.3',
            'output_blocks.5.2': 'output_blocks.5.3',
            'output_blocks.8.2': 'output_blocks.8.3',
        }
        cnt = 0
        for k in list(sd_state_dict.keys()):
            for src_word, dst_word in mapping_dict.items():
                if src_word in k:
                    new_key = k.replace(src_word, dst_word)
                    sd_state_dict[new_key] = sd_state_dict[k]
                    del sd_state_dict[k]
                    cnt += 1
        mainlogger.info(f'[renamed {cnt} source keys to match target model]')

    pretrained_dict = {k: v for k, v in sd_state_dict.items() if k in model_state_dict} # drop extra keys
    empty_paras = [k for k, v in model_state_dict.items() if k not in pretrained_dict] # log no pretrained keys
    mainlogger.info(f'Pretrained parameters: {len(pretrained_dict.keys())}')
    mainlogger.info(f'Empty parameters: {len(empty_paras)} ')
    assert(len(empty_paras) + len(pretrained_dict.keys()) == len(model_state_dict.keys()))

    if expand_to_3d:
        ## adapting to 2d inflated network
        pretrained_dict = expand_conv_kernel(pretrained_dict)

    # overwrite entries in the existing state dict
    model_state_dict.update(pretrained_dict)

    # load the new state dict
    try:
        model.load_state_dict(model_state_dict)
    except:
        state_dict = torch.load(model_state_dict, map_location=f"cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model_state_dict = model.state_dict()
        ## for layer with channel changed (e.g. GEN 1's conditon-concatenating setting)
        for n, p in model_state_dict.items():
            if p.shape != state_dict[n].shape:
                mainlogger.info(f"Skipped parameter [{n}] from pretrained! ")
                state_dict.pop(n)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

    mainlogger.info(f'---------------------------- Finish! ----------------------------')
    return model, empty_paras



## Below: written by Yingqing --------------------------------------------------------

def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        mainlogger.info("missing keys:")
        mainlogger.info(m)
    if len(u) > 0 and verbose:
        mainlogger.info("unexpected keys:")
        mainlogger.info(u)

    model.eval()
    return model

def init_and_load_ldm_model(config_path, ckpt_path, device=None):
    assert(config_path.endswith(".yaml")), f"config_path = {config_path}"
    assert(ckpt_path.endswith(".ckpt")), f"ckpt_path = {ckpt_path}"
    config = OmegaConf.load(config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, ckpt_path)  # TODO: check path
    if device is not None:
        model = model.to(device)
    return model

def load_img_model_to_video_model(model, device=None, expand_to_3d=True, adapt_keyname=False,
                                  config_path="configs/latent-diffusion/txt2img-1p4B-eval.yaml",
                                  ckpt_path="models/ldm/text2img-large/model.ckpt"):
    pretrained_ldm = init_and_load_ldm_model(config_path, ckpt_path, device)    
    model, empty_paras = load_partial_weights(model, pretrained_ldm.state_dict(), expand_to_3d=expand_to_3d,
                                            adapt_keyname=adapt_keyname)
    return model, empty_paras

def load_partial_weights(model, pretrained_dict, expand_to_3d=True, adapt_keyname=False):
    model2 = copy.deepcopy(model)
    model_dict = model.state_dict()
    model_dict_ori = copy.deepcopy(model_dict)
    
    mainlogger.info(f'-------------- Load pretrained LDM weights --------------------------')
    mainlogger.info(f'Num of parameters of target model: {len(model_dict.keys())}')
    mainlogger.info(f'Num of parameters of source model: {len(pretrained_dict.keys())}')

    if adapt_keyname:
        ## adapting to menghan's standard 2d network: modify the key name because of the add of temporal-attention
        mapping_dict = {
            'middle_block.2': 'middle_block.3',
            'output_blocks.5.2': 'output_blocks.5.3',
            'output_blocks.8.2': 'output_blocks.8.3',
        }
        cnt = 0
        newpretrained_dict = copy.deepcopy(pretrained_dict)
        for k, v in newpretrained_dict.items():
            for src_word, dst_word in mapping_dict.items():
                if src_word in k:
                    new_key = k.replace(src_word, dst_word)
                    pretrained_dict[new_key] = v
                    pretrained_dict.pop(k)
                    cnt += 1
        mainlogger.info(f'--renamed {cnt} source keys to match target model.')
    '''
    print('==================not matched parames:')
    for k, v in pretrained_dict.items():
        if k not in model_dict:
            print(k, v.shape)
    '''
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # drop extra keys
    empty_paras = [k for k, v in model_dict.items() if k not in pretrained_dict] # log no pretrained keys
    mainlogger.info(f'Pretrained parameters: {len(pretrained_dict.keys())}')
    mainlogger.info(f'Empty parameters: {len(empty_paras)} ')
    # disable info
    # mainlogger.info(f'Empty parameters: {empty_paras} ')
    assert(len(empty_paras) + len(pretrained_dict.keys()) == len(model_dict.keys()))

    if expand_to_3d:
        ## adapting to yingqing's 2d inflation network
        pretrained_dict = expand_conv_kernel(pretrained_dict)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # load the new state dict
    try:
        model2.load_state_dict(model_dict)
    except:
        # if parameter size mismatch, skip them
        skipped = []
        for n, p in model_dict.items():
            if p.shape != model_dict_ori[n].shape:
                model_dict[n] = model_dict_ori[n] # skip by using original empty paras
                mainlogger.info(f'Skip para: {n}, size={pretrained_dict[n].shape} in pretrained, \
                {model_dict[n].shape} in current model')
                skipped.append(n)
        mainlogger.info(f"[INFO] Skip {len(skipped)} parameters becasuse of size mismatch!")
        # mainlogger.info(f"[INFO] Skipped parameters: {skipped}")
        model2.load_state_dict(model_dict)
        empty_paras += skipped
        mainlogger.info(f'Empty parameters: {len(empty_paras)} ')
        # import pdb;pdb.set_trace()

    mainlogger.info(f'-------------- Finish! --------------------------')
    return model2, empty_paras


def load_autoencoder(model, config_path=None, ckpt_path=None, device=None):
    if config_path is None:
        config_path="configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    if ckpt_path is None:
        ckpt_path = "models/ldm/text2img-large/model.ckpt"
    # if device is None:
    #     device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    
    pretrained_ldm = init_and_load_ldm_model(config_path, ckpt_path, device)
    autoencoder_dict = {}
    # import pdb;pdb.set_trace()
    # mainlogger.info([n for n in pretrained_ldm.state_dict().keys()])
    # mainlogger.info([n for n in model.state_dict().keys()])
    for n, p in pretrained_ldm.state_dict().items():
        if n.startswith('first_stage_model'):
            autoencoder_dict[n] = p
    model_dict = model.state_dict()
    model_dict.update(autoencoder_dict)
    mainlogger.info(f'Load [{len(autoencoder_dict)}] autoencoder parameters!')
    
    model.load_state_dict(model_dict)

    return model
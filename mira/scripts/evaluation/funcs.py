import argparse, os, sys, glob, yaml, math, random
from collections import OrderedDict
from decord import VideoReader, cpu

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from mira.models.samplers.ddim import DDIMSampler


def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if isinstance(cond, dict):
            c_cat, text_emb = cond["c_concat"][0], cond["c_crossattn"][0]
        else:
            text_emb = cond

        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            uc = torch.zeros_like(text_emb)
        else:
            raise NotImplementedError

        ## hybrid case
        if isinstance(cond, dict):
            uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
            if 'c_adm' in cond:
                uc_hybrid.update({'c_adm': cond['c_adm']})
            uc = uc_hybrid
    else:
        uc = None

    ## sampling
    batch_variants = []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=None,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


def batch_sliding_interpolation(model, cond, base_videos, base_stride, noise_shape, n_samples=1,\
                                ddim_steps=50, ddim_eta=1.0, cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    '''
    Current implementation has a flaw: the inter-episode keyframe is used as pre-last and cur-first, so keyframe repeated.
    For example, cond_frames=[0,4,7], model.temporal_length=8, base_stride=4, then
    base frame  : 0   4   8   12  16  20  24  28
    interplation: (0~7)   (8~15)  (16~23) (20~27)
    '''
    b,c,t,h,w = noise_shape
    base_z0 = model.encode_first_stage(base_videos)
    unit_length = model.temporal_length
    n_base_frames = base_videos.shape[2]
    n_refs = len(model.cond_frames)
    sliding_steps = (n_base_frames-1) // (n_refs-1)
    sliding_steps = sliding_steps+1 if (n_base_frames-1) % (n_refs-1) > 0 else sliding_steps

    cond_mask = model.cond_mask.to("cuda")
    proxy_z0 = torch.zeros((b,c,unit_length,h,w), dtype=torch.float32).to("cuda")
    batch_samples = None
    last_offset = None
    for idx in range(sliding_steps):
        base_idx = idx * (n_refs-1)
        ## check index overflow
        if base_idx+n_refs > n_base_frames:
            last_offset = base_idx - (n_base_frames - n_refs)
            base_idx = n_base_frames - n_refs
        cond_z0 = base_z0[:,:,base_idx:base_idx+n_refs,:,:]
        proxy_z0[:,:,model.cond_frames,:,:] = cond_z0

        if isinstance(cond, dict):
            c_cat, text_emb = cond["c_concat"][0], cond["c_crossattn"][0]
            episode_idx = idx * unit_length
            if last_offset is not None:
                episode_idx = episode_idx - last_offset * base_stride
            cond_idx = {"c_concat": [c_cat[:,:,episode_idx:episode_idx+unit_length,:,:]], "c_crossattn": [text_emb]}
        else:
            cond_idx = cond
        noise_shape_idx = [b,c,unit_length,h,w]
        ## batch, <samples>, c, t, h, w
        batch_idx = batch_ddim_sampling(model, cond_idx, noise_shape_idx, n_samples, ddim_steps, ddim_eta, cfg_scale, \
                                        temporal_cfg_scale, mask=cond_mask, x0=proxy_z0, **kwargs)

        if batch_samples is None:
            batch_samples = batch_idx
        else:
            ## b,s,c,t,h,w
            if last_offset is None:
                batch_samples = torch.cat([batch_samples[:,:,:,:-1,:,:], batch_idx], dim=3)
            else:
                batch_samples = torch.cat([batch_samples[:,:,:,:-1,:,:], batch_idx[:,:,:,last_offset * base_stride:,:,:]], dim=3)

    return batch_samples


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt, adapter_ckpt=None):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model

    if adapter_ckpt:
        ## main model
        load_checkpoint(model, ckpt, full_strict=False)
        print('>>> model checkpoint loaded.')
        ## adapter
        state_dict = torch.load(adapter_ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.adapter.load_state_dict(state_dict, strict=True)
        print('>>> adapter checkpoint loaded.')
    else:
        load_checkpoint(model, ckpt, full_strict=True)
        print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)

    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})


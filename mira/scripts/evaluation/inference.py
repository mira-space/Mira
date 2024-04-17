import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange, tqdm
from einops import repeat
from collections import OrderedDict

import torch
import torchvision
from torch import Tensor
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from mira.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config

def load_model_checkpoint(model, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd.keys():
            if 'scale_arr' in pl_sd['state_dict'].keys():
                del pl_sd['state_dict']['scale_arr']
            model.load_state_dict(pl_sd["state_dict"],strict=False)
    else:
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in pl_sd['module'].keys():
                new_pl_sd[key[16:]]=pl_sd['module'][key]
            model.load_state_dict(new_pl_sd, strict=False)

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

def save_results(prompt, samples, inputs, filename, realdir, fakedir, fps=10):
    ## save prompt
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    path = os.path.join(realdir, "%s.txt"%filename)
    with open(path, 'w') as f:
        f.write(f'{prompt}')
        f.close()

    ## save video
    videos = [inputs, samples]
    savedirs = [realdir, fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], "%s.mp4"%filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
        

def inference_prompt(model, prompts, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                unconditional_guidance_scale=1.0, unconditional_guidance_scale_temporal=None,negative_prompts=None,  **kwargs):
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    if isinstance(prompts, str):
        prompts = [prompts]
    batch_cond =  model.get_condition_validate(prompts)

    if unconditional_guidance_scale != 1.0:
        if negative_prompts is not None:
            assert len(negative_prompts) == batch_size
            uc = model.get_condition_validate(negative_prompts)
        else:
            prompts = batch_size * [""]
            uc = model.get_condition_validate(prompts)
    else:
        uc = None

    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=batch_cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal, **kwargs
                                            )
        

        batch_variants.append(samples)

    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def assign_tasks(num_samples, gpu_num):
    tasks_per_gpu = [num_samples // gpu_num] * gpu_num
    remaining_tasks = num_samples % gpu_num

    for i in range(remaining_tasks):
        tasks_per_gpu[i] += 1

    gpu_indices = []
    start = 0
    for tasks in tasks_per_gpu:
        end = start + tasks
        gpu_indices.append(list(range(start, end)))
        start = end

    return gpu_indices

@torch.no_grad()
def run_inference(args, rank, gpu_num):
    ## model config
    config = OmegaConf.load(args.base)
    data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['inference']=True               
    model = instantiate_from_config(model_config)
    gpu_per_node = torch.cuda.device_count()
    model = model.cuda(rank%gpu_per_node)
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    model.half()

    ## run over data
    assert (args.height % 8 == 0) and (args.width % 8 == 0), "Error: image size [h,w] should be multiples of 8!"

    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.in_channels
    frames = model.temporal_length
    noise_shape = [args.bs, channels, frames, h, w]

    realdir = os.path.join(args.savedir, "input")
    fakedir = os.path.join(args.savedir, "samples")
    os.makedirs(realdir, exist_ok=True)
    os.makedirs(fakedir, exist_ok=True)

    start = time.time()  
    if args.prompt_file:
        ## prompt file setting
        assert os.path.exists(args.prompt_file), "Error: prompt file Not Found!"
        prompt_list = load_prompts(args.prompt_file)
        if args.trigger_words is not None:
            prompt_list = [p + " " + args.trigger_words for p in prompt_list]
        negative_prompt_list = len(prompt_list) * [""]

        num_samples = len(prompt_list)
        gpu_indices = assign_tasks(num_samples, gpu_num)
        indices = gpu_indices[rank]
        samples_split = len(indices)
        print('Prompts testing [rank:%d] %d/%d samples loaded.'%(rank, samples_split, num_samples))
        
        while True:
            all_sample = []
            for i in tqdm(range(0, len(indices), args.bs), desc='Sample Batch'):
                indice = indices[i]
                prompts = prompt_list[indice:indice+args.bs]
                negative_prompts = negative_prompt_list[indice:indice+args.bs]
                batch_samples = inference_prompt(model, prompts, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                            args.unconditional_guidance_scale, args.unconditional_guidance_scale_temporal,
                                            cond_tau=args.cond_tau, target_size=args.target_size, fps=[args.fps] * args.bs, dilated_conv=args.dilated_conv, dilated_conv_tau=args.dilated_conv_tau, free_u=args.free_u, negative_prompts=negative_prompts)
                all_sample.append(batch_samples.detach() )
                del batch_samples

            ## To avoide OOM
            decoder = model.first_stage_model
            scale_factor = model.scale_factor
            if args.width > 384:
                model.cpu()
                decoder.cuda()
            from einops  import rearrange
            def decode(z):
                z = 1.0 / scale_factor * z
                b, c, t, h, w = z.shape
                z = rearrange(z, "b c t h w -> (b t) c h  w", t=t)
                kwargs = {"timesteps": t}
                torch.cuda.empty_cache()
                results = decoder.decode(z, **kwargs)
                results = rearrange(results, " (b t) c h  w  -> b c t h w ", t=t)
                return results
            nn = 0
            while all_sample:
                batch_samples = all_sample.pop(0)
                prompt = prompt_list[nn]
                if batch_samples.dim() == 5:
                    res = decode(batch_samples)
                    filename = "%04d" % (nn)
                    save_results(prompt, res, None, filename, realdir, fakedir, fps=6)
                elif batch_samples.dim() == 6:
                    batch_samples_cpu = [batch_samples[:, i].cpu() for i in range(batch_samples.shape[1])]
                    del batch_samples
                    i = 0
                    while(batch_samples_cpu):
                        res = decode(batch_samples_cpu.pop(0).cuda())
                        filename = "%04d_Version%04d" % (nn, i)
                        save_results(prompt, res, None, filename, realdir, fakedir, fps=6)
                        i+=1
                        del res
                nn += 1
            model.cuda()

            import time as t
            t.sleep(20)
    else:
        ## dataset settting
        try:
            dataset = instantiate_from_config(data_config.params.validation)
        except:
            print("Error: dataset configure failed!")

        num_samples = len(dataset)
        samples_split = num_samples // gpu_num
        print('Dataset testing [rank:%d] %d/%d samples loaded.'%(rank, samples_split, num_samples))
        #indices = random.choices(list(range(0, num_samples)), k=samples_split)
        indices = list(range(samples_split*rank, samples_split*(rank+1)))
        dataset_rank = torch.utils.data.Subset(dataset, indices)
        dataloader_rank = DataLoader(dataset_rank, batch_size=args.bs, num_workers=4, shuffle=False)

        for idx, batch in tqdm(enumerate(dataloader_rank), desc='Sample Batch'):
            prompts = batch[model.cond_stage_key]
            batch_samples = inference_prompt(model, prompts, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                        args.unconditional_guidance_scale, args.unconditional_guidance_scale_temporal,
                                        cond_tau=args.cond_tau, target_size=args.target_size)
            ## save each example individually
            inputs = batch[model.first_stage_key]
            inputs = inputs.unsqueeze(0).permute(1,0,2,3,4,5)
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = "%04d_randk%d"%(idx*args.bs+nn, rank)
                save_results(prompt, samples, inputs[nn], filename, realdir, fakedir, fps=8)
    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--base", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--cond_tau", type=float, default=1.0, help="",)
    parser.add_argument("--target_size", type=int, default=None, help="", nargs="+")
    parser.add_argument("--fps", type=int, default=8, help="", nargs="+")
    parser.add_argument("--dilated_conv", action="store_true")
    parser.add_argument("--dilated_conv_tau", type=int, default=None, help="", nargs="+")
    parser.add_argument("--free_u", action="store_true")
    parser.add_argument("--trigger_words", type=str, default=None, help="")
    return parser



if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@Mira Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    with torch.cuda.amp.autocast():
        run_inference(args, rank, gpu_num)

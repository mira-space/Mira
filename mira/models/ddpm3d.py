import os, random
from functools import partial
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import logging

mainlogger = logging.getLogger('mainlogger')


import torch
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_only

from utils.utils import instantiate_from_config, count_params, check_istarget
from mira.distributions import normal_kl, DiagonalGaussianDistribution
from mira.models.utils_diffusion import make_beta_schedule
from mira.models.samplers.ddim import DDIMSampler
from mira.basics import disabled_train
from mira.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)
import itertools
from pytorch_lightning import seed_everything
from utils.save_video import npz_to_video_grid
from mira.scripts.evaluation.inference import inference_prompt, load_prompts

from torchvision import transforms

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

from mira.models.base_ddpm import DDPM




class MiraDDPM(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 uncond_prob=0.2,
                 uncond_type="empty_seq",
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 frame_cond=None,
                 only_model=False,
                 logdir=None,
                 empty_params_only=False,
                 fix_layernorm=False,
                 train_attn=False,
                 inject_denoiser=False,
                 inject_clip=False,
                 inject_denoiser_key_word=None,
                 inject_clip_key_word=None,
                 inject_denoiser_child_name=None,
                 noise_normal=False,
                 use_scale=False,
                 scale_a=1,
                 scale_b=0.3,
                 mid_step=400,
                 train_block=None,
                 img2video=False,
                 fps_cond=False,
                 iv_ratio=0.3,
                 train_num_timesteps=0,
                 max_num_timesteps=1000,
                 train_spatial=False,
                 inference=False,
                 fix_scale_bug=False,
                 frame_loss=False,
                 img_copy_video=False,
                 single_frame_loss=False,
                 no_train_block=None,
                 bigbatch_encode=False,
                 spatial_mini_batch=1,
                 cond_stage_config2=None,
                 video_to_image=False,
                 video_diff_step=False,
                 noise_offset=0,  prompt_val=None, temporal_vae=False,
                 rescale_betas_zero_snr=False, rec_loss=0,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.rec_loss = rec_loss
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.empty_params_only = empty_params_only
        self.fix_layernorm = fix_layernorm
        self.train_attn = train_attn
        self.inject_denoiser = inject_denoiser
        self.inject_clip = inject_clip
        self.inject_denoiser_key_word = inject_denoiser_key_word
        self.inject_clip_key_word = inject_clip_key_word
        self.inject_denoiser_child_name = inject_denoiser_child_name
        self.noise_normal = noise_normal
        self.enable_deepspeed = False
        # load testing prompts
        self.prompt_val = load_prompts(prompt_val) if prompt_val else None
        self.temporal_vae = temporal_vae
        # scale factor
        self.use_scale = use_scale
        self.train_block = train_block
        self.img2video = img2video
        self.fps_cond = fps_cond
        self.iv_ratio = iv_ratio
        self.train_num_timesteps = train_num_timesteps
        self.max_num_timesteps = max_num_timesteps
        self.train_spatial = train_spatial
        self.frame_loss = frame_loss
        self.single_frame_loss = single_frame_loss
        self.img_copy_video = img_copy_video
        self.no_train_block = no_train_block
        self.bigbatch_encode = bigbatch_encode
        self.spatial_mini_batch = spatial_mini_batch
        self.video_to_image = video_to_image
        self.video_diff_step = video_diff_step
        self.noise_offset = noise_offset
        if self.img2video:
            self.img_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ])

        if self.use_scale:
            self.scale_a = scale_a
            self.scale_b = scale_b
            self.mid_step = mid_step
            if fix_scale_bug:
                scale_step = self.num_timesteps - mid_step
            else:  # bug
                scale_step = self.num_timesteps

            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            scale_arr_prev = np.append(scale_a, scale_arr[:-1])
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))
            # self.register_buffer('scale_arr_prev', to_torch(scale_arr_prev))
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        if cond_stage_config2 is not None:
            self.instantiate_cond_stage2(cond_stage_config2)

        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.cond_stage_config2 = cond_stage_config2
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert (encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert (uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        ## future frame prediction
        self.frame_cond = frame_cond
        if self.frame_cond:
            frame_len = self.temporal_length
            cond_mask = torch.zeros(frame_len, dtype=torch.float32)
            cond_mask[:self.frame_cond] = 1.0
            ## b,c,t,h,w
            self.cond_mask = cond_mask[None, None, :, None, None]
            mainlogger.info("---training for %d-frame conditoning T2V" % (self.frame_cond))
        else:
            self.cond_mask = None

        self.restarted_from_ckpt = False

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

        self.logdir = logdir



    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)


        if self.use_scale:
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start *
                    extract_into_tensor(self.scale_arr, t, x_start.shape) +
                    extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        else:
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                    extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def _freeze_model(self):
        for name, para in self.model.diffusion_model.named_parameters():
            para.requires_grad = False


    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and \
                not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            # x = super().get_input(batch, self.first_stage_key)
            x, c, fps = self.get_batch_input(batch, random_uncond=0, is_imgbatch=True)
            z = x.detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            from .utils_diffusion import rescale_zero_terminal_snr
            betas = rescale_zero_terminal_snr(betas)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == 'v':
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        if 'diffusers' in config.target:
            from utils.utils import get_obj_from_str
            logging.info("Using diffuser VAE")
            model = get_obj_from_str(config.target).from_pretrained(config.ckpt, subfolder="vae")
        else:
            model = instantiate_from_config(config)

        if self.training and self.rec_loss:
            self.first_stage_model = model.eval()
            self.first_stage_model.decoder = model.decoder.train()
            for n, param in self.first_stage_model.named_parameters():
                if n in self.first_stage_model.decoder.named_parameters():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            self.first_stage_model = model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False

        sd_ckpt = getattr(config, 'pretrain', None)
        if sd_ckpt:
            mainlogger.info('##### Using ST Pretrain VAE')
            from safetensors import safe_open
            sd_state_dict = {}
            with safe_open(sd_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    sd_state_dict[key] = f.get_tensor(key)
            new_ckpt = {}
            for k, v in sd_state_dict.items():
                if "first_stage_model" in k:
                    new_ckpt[k[len("first_stage_model."):]] = v
            self.first_stage_model.load_state_dict(new_ckpt, strict=True)

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def instantiate_cond_stage2(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model2 = model.eval()
            self.cond_stage_model2.train = disabled_train
            for param in self.cond_stage_model2.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model2 = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_learned_conditioning2(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model2, 'encode') and callable(self.cond_stage_model2.encode):
                c = self.cond_stage_model2.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model2(c)
        else:
            assert hasattr(self.cond_stage_model2, self.cond_stage_forward)
            c = getattr(self.cond_stage_model2, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False

        z = 1. / self.scale_factor * z

        results = self.first_stage_model.decode(z, **kwargs)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b, t=t)
        return results


    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def video_aug(self, video_tensor):
        B, C, T, H, W = video_tensor.shape
        rotate_bool = bool(random.getrandbits(1))
        affine_bool = bool(random.getrandbits(1))
        resize_bool = bool(random.getrandbits(1))
        if not (rotate_bool or affine_bool or resize_bool):
            rotate_bool = True
        for b in range(B):
            rotate_ang = random.random()
            rotate_neg = random.choice([-1, 1])
            # affine_rand = random.random()
            for t in range(T):
                if rotate_bool:
                    rotation_angle = t * rotate_ang * 3
                else:
                    rotation_angle = 0
                if affine_bool:
                    affine_num = 0.01
                else:
                    affine_num = 0
                if resize_bool:
                    scale_H = H - t * 10
                    scale_W = W - t * 10
                else:
                    scale_H = H
                    scale_W = W

                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(rotation_angle),
                    transforms.RandomAffine(0, translate=(t * affine_num, t * affine_num)),
                    transforms.Resize((scale_H, scale_W)),
                    transforms.Resize((H, W)),
                    transforms.ToTensor()
                ])

                frame_cpu = video_tensor[b, :, t].cpu()
                transformed_frame = transform(frame_cpu)
                video_tensor[b, :, t] = transformed_frame.cuda()
        return video_tensor

    @torch.no_grad()
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False,
                        is_imgbatch=False, log=False):
        ## image/video shape: b, c, t, h, w
        data_key = 'jpg' if is_imgbatch else self.first_stage_key
        x = super().get_input(batch, data_key)
        if is_imgbatch:
            ## pack image as video
            # x = x[:,:,None,:,:]
            if self.img2video:
                cond = self.img_transform(x)
            if self.img_copy_video:
                x = x.unsqueeze(2)  # B C 1 H W
                x = x.repeat(1, 1, self.temporal_length, 1, 1)
                # x = self.video_aug(x)
            else:
                b = x.shape[0] // self.temporal_length
                x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=self.temporal_length)
            if log:
                x = x[0:1, :, 0:1, :, :]
        elif self.video_to_image:
            x = x[:, :, 0:1, :, :]
            x = x.repeat(1, 1, self.temporal_length, 1, 1)
        x_ori = x

        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage_2DAE(x)

        if not self.img2video:
            if is_imgbatch:
                cond_key = 'txt'
                cond = batch[cond_key]

                if log:
                    cond = [cond[0]]
            else:
                cond_key = self.cond_stage_key
                cond = batch[cond_key]
            if random_uncond and self.uncond_type == 'empty_seq':
                for i, ci in enumerate(cond):
                    if random.random() < self.uncond_prob:
                        cond[i] = ""

        # for img2video
        if self.img2video:
            if is_imgbatch:
                pass
            else:
                middle_frame_index = x.size(2) // 2
                middle_frame = x.narrow(2, middle_frame_index, 1).squeeze(2)  # bcthw
                middle_frame = self.img_transform(middle_frame)
                cond = middle_frame
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)

        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))

        if self.cond_stage_config2:
            if isinstance(cond, dict) or isinstance(cond, list):
                cond_emb2 = self.get_learned_conditioning2(cond)
            else:
                cond_emb2 = self.get_learned_conditioning2(cond.to(self.device))
            cond_emb = torch.cat([cond_emb, cond_emb2[0]], dim=-1)

        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(ci)

        out = [z, cond_emb]

        # for fps cond
        if not is_imgbatch and self.fps_cond:
            if 'fps' not in batch.keys():
                print('data not support fps in ddpm3d!!')
                out.append(None)
            else:
                out.append(batch['fps'])
        else:
            # is_imgbatch is 100
            if log:
                out.append(8)
            else:
                out.append(100)

        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage_3DVAE_diff(z) if self.rec_loss else self.decode_first_stage_2DAE(z)
            out.extend([x_ori, xrec])
        if return_original_cond:
            out.append(cond)

        return out

    def forward(self, x, c, inter_num_timesteps=None, **kwargs):
        if inter_num_timesteps is not None:
            max_num_timesteps = inter_num_timesteps
        else:
            max_num_timesteps = self.max_num_timesteps
        t = torch.randint(self.train_num_timesteps, max_num_timesteps, (x.shape[0],), device=self.device).long()


        return self.p_losses(x, c, t, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        is_imgbatch = False
        if "loader_img" in batch.keys():
            # ratio = 10.0 / self.temporal_length
            ratio = self.iv_ratio
            if random.random() < ratio:
                is_imgbatch = True
                batch = batch["loader_img"]
            else:
                batch = batch["loader_video"]
        else:
            if 'json' in batch.keys():
                is_imgbatch = True
            else:
                pass

        if self.video_diff_step:
            inter_num_timesteps = 500
        else:
            inter_num_timesteps = None

        if self.rec_loss:
            x, c, fps, xori, xrec = self.get_batch_input(batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch,
                                                         return_first_stage_outputs=True)
        else:
            x, c, fps = self.get_batch_input(batch, random_uncond=random_uncond, is_imgbatch=is_imgbatch,
                                             return_first_stage_outputs=False)

        loss, loss_dict = self(x, c, is_imgbatch=is_imgbatch and not self.img_copy_video, fps=fps,
                               video_to_image=self.video_to_image, inter_num_timesteps=inter_num_timesteps, **kwargs)
        if self.rec_loss:
            rec_loss = self.get_loss(xori, xrec)
            loss = loss + rec_loss * self.rec_loss
            loss_dict.update({f'trian/rec_loss': rec_loss.mean()})
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if self.noise_normal:
            x_noisy = x_noisy / torch.std(x_noisy, dim=(2, 3, 4), keepdim=True)

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3, 4])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(
                f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=False)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, random_uncond=False)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        ## only log for each epoch
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        torch.cuda.empty_cache()

        return loss

    @torch.no_grad()
    def on_validation_epoch_start(self):
        ## saving for calculate FVD over the whole validation set
        subname = "val_epoch"  # "val_epoch%03d"%self.current_epoch
        self.val_dir = os.path.join(self.logdir, "images", subname)
        os.makedirs(self.val_dir, exist_ok=True)
        mainlogger.info(f"@validating at [epoch: {self.current_epoch} | global_step: {self.global_step}]...")

    def get_condition_validate(self, prompt):
        """ text embd or
            text embd & fps embd
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        c = self.get_learned_conditioning(prompt)
        bs = c.shape[0]
        if hasattr(self, 'cond_stage2_key') and self.cond_stage2_key == "temporal_context":
            batch = {'fps': torch.tensor([4] * bs, device=self.device).long()}
            c2 = self.cond_stage2_model(batch)
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            c = {key: [c], self.cond_stage2_key: c2}
        return c

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        val_dir = os.path.join(self.logdir, "val")
        os.makedirs(val_dir, exist_ok=True)
        step_dir = os.path.join(val_dir, f"global_step_{self.global_step:07d}")
        os.makedirs(step_dir, exist_ok=True)

        is_rank0 = (dist.get_rank() == 0)
        # val given prompt
        if self.val_prompt_file is not None:
            print(f'sample prompts: {self.val_prompt_file}')
            f = open(self.val_prompt_file, 'r')
            prompts = []
            for line in f.readlines():
                if len(line.strip()) != 0:
                    prompts.append(line.strip())
            f.close()

            ddim_steps = getattr(self.val_prompt_file_args, "ddim_steps", [25])
            save_mp4_single = getattr(self.val_prompt_file_args, "save_mp4_single", True)
            save_fps = getattr(self.val_prompt_file_args, "save_fps", 6)

            for ddim_step in ddim_steps:
                i = 0
                for prompt in prompts:
                    c = self.get_condition_validate(prompt)
                    uc = self.get_condition_validate("")
                    rank_id = dist.get_rank()
                    seed_everything(self.seed + rank_id)
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=1, ddim=True,
                                                             ddim_steps=ddim_step, eta=1.,
                                                             unconditional_guidance_scale=15,
                                                             unconditional_conditioning=uc,
                                                             mask=self.cond_mask)

                    i += 1
                    torch.cuda.empty_cache()
                    samples = self.decode_first_stage_2DAE(samples, mask_temporal=False, decode_bs=1)
                    samples = self.all_gather(samples).flatten(start_dim=0, end_dim=1)
                    if is_rank0:
                        samples = samples.to(torch.float32).clamp_(-1, 1).add_(1).mul(127.5).cpu().numpy().astype(
                            np.uint8).transpose(0, 2, 3, 4, 1)  # BTHWC
                        shape_str = "x".join([str(x) for x in samples.shape])
                        os.makedirs(os.path.join(step_dir, "sample_prompts"), exist_ok=True)
                        npz_to_video_grid(samples, os.path.join(step_dir, "sample_prompts",
                                                                f"ddim{ddim_step}_prompt_{i:03}_{shape_str}.mp4"),
                                          num_frames=samples.shape[1], fps=save_fps)
                        if save_mp4_single:
                            savedir = os.path.join(step_dir, "sample_prompts", "videos")
                            os.makedirs(savedir, exist_ok=True)
                            for j in range(samples.shape[0]):
                                vid = samples[j:j + 1, ...]
                                out_path = os.path.join(savedir, f"ddim{ddim_step}_prompt{i:03d}_{prompt}_{j:03d}.mp4")
                                npz_to_video_grid(vid, out_path, num_frames=samples.shape[1], fps=save_fps,
                                                  num_videos=None, nrow=None)

            seed_everything(self.seed)


        torch.cuda.empty_cache()
        if dist.get_rank() == 0:

            torch.cuda.empty_cache()
            fvd, kvd, n_samples = 0.000, 0.000, 2048
            mainlogger.info('metric calculating over %d samples' % n_samples)
            val_metrics = {
                'val/fvd': torch.tensor(fvd),
                'val/kvd': torch.tensor(kvd)
            }
            self.log_dict(val_metrics, prog_bar=False, logger=True, on_step=False, on_epoch=True, rank_zero_only=True)

    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W

        if denoise_row.dim() == 5:
            # img, num_imgs= n_log_timesteps * bs, grid_size=[bs,n_log_timesteps]
            # 先batch再n，grid时候一行是一个sample的不同steps，batch是列，行是n
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=200, ddim_eta=1., plot_denoise_rows=False, \
                   unconditional_guidance_scale=1.0, guidance_rescale=0, ddim_discretize="uniform_uniform_trailing",
                   **kwargs):
        """ log images for LatentDiffusion """
        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        is_imgbatch = False
        if "loader_img" in batch.keys():
            batch = batch["loader_video"]
        else:
            if 'json' in batch.keys():
                is_imgbatch = True
            else:
                pass
        fps = None
        torch.cuda.empty_cache()
        z, c, fps, x, xrec, xc = self.get_batch_input(batch, random_uncond=False,
                                                      return_first_stage_outputs=True,
                                                      return_original_cond=True, is_imgbatch=is_imgbatch,
                                                      log=is_imgbatch)
        N, _, T, H, W = x.shape
        log["inputs"] = x
        log["reconst"] = xrec
        log["condition"] = xc
        # if type(fps) == int:
        #     fps=None
        if fps is not None and self.fps_cond:
            if is_imgbatch:
                fps_list = [fps] * N
            else:
                fps_list = list(map(str, fps.cpu().tolist()))
            log["fps"] = fps_list

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_cat, c_emb = c["c_concat"][0], c["c_crossattn"][0]
                    # log["condition_cat"] = c_cat
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc = self.get_learned_conditioning(prompts)
                    if self.cond_stage_config2:
                        uc2 = self.get_learned_conditioning2(prompts)[0]
                        uc = torch.concat([uc, uc2], dim=-1)
                elif self.uncond_type == "zero_embed":
                    uc = torch.zeros_like(c_emb)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None
            is_imgbatch = False
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, mask=self.cond_mask, x0=z,
                                                         is_imgbatch=is_imgbatch, fps=fps,
                                                         guidance_rescale=guidance_rescale,
                                                         ddim_discretize=ddim_discretize, **kwargs)

            x_samples = self.decode_first_stage_2DAE(samples)
            log["samples"] = x_samples

            ## log testing samples
            if self.prompt_val:
                torch.cuda.empty_cache()
                iterator = tqdm(range(len(self.prompt_val)))
                noise_shape = (1, self.channels, self.temporal_length, *self.image_size)
                for i in iterator:
                    p = self.prompt_val[i]
                    batch_samples = inference_prompt(self, p, noise_shape, 1, ddim_steps,
                                                     ddim_eta,
                                                     unconditional_guidance_scale,
                                                     fps=fps, **kwargs)
                    save_p = p if len(p) < 100 else p[:100]
                    log["test/{}".format(save_p)] = batch_samples[0]

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    @torch.no_grad()
    def decode_first_stage_2DAE(self, z, **kwargs):
        if self.temporal_vae:
            return self.decode_first_stage_3DVAE(z, **kwargs)
        b, c, t, h, w = z.shape
        z = 1. / self.scale_factor * z
        if self.bigbatch_encode:
            print('decode 2DAE')
            mini_b = self.spatial_mini_batch
            print('DOCODE1:z.is_contiguous()', z.is_contiguous())
            z = z.reshape(mini_b, c, t * b // mini_b, h, w)
            print('DOCODE2:z.is_contiguous()', z.is_contiguous())
        _, _, n, _, _ = z.shape
        results = torch.cat([self.first_stage_model.decode(z[:, :, i], **kwargs).unsqueeze(2) for i in range(n)], dim=2)
        if self.bigbatch_encode:
            _, new_c, _, new_h, new_w = results.shape
            print('DOCODE1:results.is_contiguous()', results.is_contiguous())
            results = results.reshape(b, new_c, t, new_h, new_w)
            print('DOCODE2:results.is_contiguous()', results.is_contiguous())

        return results

    @torch.no_grad()
    def decode_first_stage_3DVAE(self, z):
        z = 1.0 / self.scale_factor * z
        b, c, t, h, w = z.shape
        z = rearrange(z, "b c t h w -> (b t) c h  w", t=t)
        kwargs = {"timesteps": t}
        torch.cuda.empty_cache()
        results = self.first_stage_model.decode(z, **kwargs)
        results = rearrange(results, " (b t) c h  w  -> b c t h w ", t=t)
        return results

    def decode_first_stage_3DVAE_diff(self, z):
        z = 1.0 / self.scale_factor * z
        b, c, t, h, w = z.shape
        z = rearrange(z, "b c t h w -> (b t) c h  w", t=t)
        kwargs = {"timesteps": t}
        torch.cuda.empty_cache()
        results = self.first_stage_model.decode(z, **kwargs)
        results = rearrange(results, " (b t) c h  w  -> b c t h w ", t=t)
        return results

    @torch.no_grad()
    def encode_first_stage_3DVAE(self, x):
        ## Currently still 2D form to suppotral SVD AEKL-Temporal-Decoder
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c   h w')
        if self.bigbatch_encode:
            mini_b = self.spatial_mini_batch
            x = x.reshape(mini_b, t * b // mini_b, c, h, w)
            results = torch.cat(
                [self.first_stage_model.encode(x[i]).detach().unsqueeze(0) for i in
                 range(mini_b)], dim=0)
            n, t_b_on_n, _, new_h, new_w = results.shape
            results = results.reshape(b * t, c, new_h, new_w)
        else:
            results = self.first_stage_model.encode(x)

        results = rearrange(results, '(b t) c h w -> b c t  h w', b=b)
        return results * self.scale_factor

    @torch.no_grad()
    def encode_first_stage_2DAE(self, x):
        if self.temporal_vae:
            return self.encode_first_stage_3DVAE(x)

        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c   h w')
        if self.bigbatch_encode:
            mini_b = self.spatial_mini_batch
            x = x.reshape(mini_b, t * b // mini_b, c, h, w)

            results = torch.cat(
                [self.get_first_stage_encoding(self.first_stage_model.encode(x[i])).detach().unsqueeze(0) for i in
                 range(mini_b)], dim=0)
            n, t_b_on_n, _, new_h, new_w = results.shape
            results = results.reshape(b * t, c, new_h, new_w)
        else:
            results = self.get_first_stage_encoding(self.first_stage_model.encode(x))

        results = rearrange(results, '(b t) c h w -> b c t  h w', b=b)
        return results

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None,
                        corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == 'v':
            x_recon = extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * x - (
                    extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * model_out)
            # mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, \
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise)

        noise = noise * temperature

        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, \
               verbose=True, timesteps=None, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps,
                                  mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            kwargs.update({"clean_cond": True})
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    def configure_optimizers(self):

        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        if self.empty_params_only and hasattr(self, "empty_paras"):
            if self.train_block != None and not self.train_spatial:
                params = [p for n, p in self.model.named_parameters() if
                          n in self.empty_paras and any(f in n for f in self.train_block)]
                print('self.empty_paras', len(self.empty_paras))
                mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
            elif self.train_spatial:
                params = [p for n, p in self.model.named_parameters() if n not in self.empty_paras]
                print('self.empty_paras', len(self.empty_paras))
                if self.train_block != None:
                    set_params = [p for n, p in self.model.named_parameters() if any(f in n for f in self.train_block)]
                    params = params + set_params
                mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
            else:
                params = [p for n, p in self.model.named_parameters() if n in self.empty_paras]
                print('self.empty_paras', len(self.empty_paras))
                mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
        elif self.train_block != None:  # for train_block
            params = [p for n, p in self.model.named_parameters() if any(f in n for f in self.train_block)]
            mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")
        elif self.no_train_block != None:  # for train_block
            params = [p for n, p in self.model.named_parameters() if not any(f in n for f in self.no_train_block)]
            mainlogger.info(f"@Training [{len(params)}] Empty Paramters ONLY.")

        elif hasattr(self, 'trainable_denoiser_modules') and len(self.trainable_denoiser_modules) > 0:
            params = []
            params_names = []
            for n, p in self.model.diffusion_model.named_parameters():
                if check_istarget(n, self.trainable_denoiser_modules):
                    params.append(p)
                    params_names.append(n)
            if self.cond_stage_trainable:
                for n, p in self.cond_stage_model.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    params_names.append(n)
            mainlogger.info(f"@Training [{len(params)}] Partial Paramters.")
        else:
            if self.fix_layernorm:
                params = [p for n, p in self.model.named_parameters() if
                          ((not (('norm' in n) and ('weight' in n))) and (not 'layers.0.weight' in n))]
                empty_params = []
                for n, p in self.model.named_parameters():
                    if (('norm' in n) and ('weight' in n)) or ('layers.0.weight' in n):
                        p.requires_grad = False
                        empty_params.append(p)

                mainlogger.info(f"@Training [{len(params)}] Paramters. with fix layernorm[{len(empty_params)}]")
            elif self.train_attn:
                params = [p for n, p in self.model.named_parameters() if (
                            'attn1.to_' in n or 'attn2.to_' in n or 'in_layers.2.weight' in n or 'emb_layers.1.weight' in n or 'out_layers.3.weight' in n or 'skip_connection.weight' in n)]
                empty_params = []
                for n, p in self.model.named_parameters():
                    if not (
                            'attn1.to_' in n or 'attn2.to_' in n or 'in_layers.2.weight' in n or 'emb_layers.1.weight' in n or 'out_layers.3.weight' in n or 'skip_connection.weight' in n):
                        empty_params.append(p)
                mainlogger.info(f"@Training attn [{len(params)}] Paramters. with fix [{len(empty_params)}]")
            else:
                params = list(self.model.parameters())
                mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        # if self.enable_deepspeed:
        #     if self.rec_loss:
        #         params += list(self.first_stage_model.decoder.parameters())
        #         mainlogger.info(f"@Training [{len(params)}] Full Paramters + VAE Decoder.")
        #     optimizer = torch.optim.AdamW(params, lr=lr)
        #     # from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
        #     # optimizer = DeepSpeedCPUAdam(params, lr=lr)
        #     # optimizer = FusedAdam(params, lr=lr)
        # else:
        #     optimizer = torch.optim.AdamW(params, lr=lr)

        optimizer = torch.optim.AdamW(params, lr=lr)
        if self.rec_loss:
            params += list(self.first_stage_model.decoder.parameters())
            mainlogger.info(f"@Training [{len(params)}] Full Paramters + VAE Decoder.")

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer

    def configure_schedulers(self, optimizer):
        assert 'target' in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split('.')[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                'interval': interval,
                'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step),
                'interval': interval,
                'frequency': frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler

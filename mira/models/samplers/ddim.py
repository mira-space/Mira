"""SAMPLING ONLY."""

import numpy as np
from tqdm import tqdm

import torch
from mira.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from mira.common import noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0
        # self.bd_noise = bd_noise
        # self.bd_ratio = bd_ratio
        # if self.bd_noise:
        #     self.bd = BD(G=self.temporal_length)
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)

        alphas_cumprod = self.model.alphas_cumprod.to(torch.float32)
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas.to(torch.float32)))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev.to(torch.float32)))
        self.use_scale = self.model.use_scale
        # try:
        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # except:
        #     pass

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        ddim_sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[self.ddim_timesteps]
        ddim_sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[self.ddim_timesteps]
        ddim_sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod[self.ddim_timesteps]
        ddim_sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod[self.ddim_timesteps]
        self.register_buffer('ddim_sqrt_recip_alphas_cumprod', ddim_sqrt_recip_alphas_cumprod)
        self.register_buffer('ddim_sqrt_recipm1_alphas_cumprod', ddim_sqrt_recipm1_alphas_cumprod)
        self.register_buffer('ddim_sqrt_alphas_cumprod', ddim_sqrt_alphas_cumprod)
        self.register_buffer('ddim_sqrt_one_minus_alphas_cumprod', ddim_sqrt_one_minus_alphas_cumprod)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.skip_step = self.ddpm_num_timesteps // S
        discr_method = "uniform_trailing" if self.model.rescale_betas_zero_snr else "uniform"
        self.make_schedule(ddim_num_steps=S, ddim_discretize=discr_method, ddim_eta=eta, verbose=schedule_verbose)

        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    **kwargs)
        return samples, intermediates



    # @torch.no_grad()
    # def sample(self,
    #            S,
    #            batch_size,
    #            shape,
    #            conditioning=None,
    #            callback=None,
    #            normals_sequence=None,
    #            img_callback=None,
    #            quantize_x0=False,
    #            eta=0.,
    #            mask=None,
    #            x0=None,
    #            temperature=1.,
    #            noise_dropout=0.,
    #            score_corrector=None,
    #            corrector_kwargs=None,
    #            verbose=True,
    #            schedule_verbose=False,
    #            x_T=None,
    #            log_every_t=100,
    #            unconditional_guidance_scale=1.,
    #            unconditional_conditioning=None,
    #            ddim_discretize="uniform",guidance_rescale=0.7,
    #            # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    #            **kwargs
    #            ):
    #
    #     # check condition bs
    #     if conditioning is not None:
    #         if isinstance(conditioning, dict):
    #             try:
    #                 cbs = conditioning[list(conditioning.keys())[0]].shape[0]
    #             except:
    #                 cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]
    #
    #             if cbs != batch_size:
    #                 print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
    #         else:
    #             if conditioning.shape[0] != batch_size:
    #                 print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
    #
    #     self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose, ddim_discretize=ddim_discretize)
    #
    #     # make shape
    #     if len(shape) == 3:
    #         C, H, W = shape
    #         size = (batch_size, C, H, W)
    #     elif len(shape) == 4:
    #         C, T, H, W = shape
    #         size = (batch_size, C, T, H, W)
    #     # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
    #
    #     samples, intermediates = self.ddim_sampling(conditioning, size,
    #                                                 callback=callback,
    #                                                 img_callback=img_callback,
    #                                                 quantize_denoised=quantize_x0,
    #                                                 mask=mask, x0=x0,
    #                                                 ddim_use_original_steps=False,
    #                                                 noise_dropout=noise_dropout,
    #                                                 temperature=temperature,
    #                                                 score_corrector=score_corrector,
    #                                                 corrector_kwargs=corrector_kwargs,
    #                                                 x_T=x_T,
    #                                                 log_every_t=log_every_t,
    #                                                 unconditional_guidance_scale=unconditional_guidance_scale,
    #                                                 unconditional_conditioning=unconditional_conditioning,
    #                                                 verbose=verbose,guidance_rescale =guidance_rescale,
    #                                                 **kwargs)
    #     return samples, intermediates
    #
    # @torch.no_grad()
    # def ddim_sampling(self, cond, shape,
    #                   x_T=None, ddim_use_original_steps=False,
    #                   callback=None, timesteps=None, quantize_denoised=False,
    #                   mask=None, x0=None, img_callback=None, log_every_t=100,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
    #                   cond_tau=1., target_size=None, start_timesteps=None, dilated_conv=False, dilated_conv_tau=None, replace_low=False,free_u=False,guidance_rescale=0,
    #                   **kwargs):
    #     device = self.model.betas.device
    #     b = shape[0]
    #     if x_T is None:
    #         img = torch.randn(shape, device=device)
    #         if self.model.bd_noise:
    #             noise_decor = self.model.bd(img)
    #             noise_decor = (noise_decor - noise_decor.mean()) / (noise_decor.std() + 1e-5)
    #             noise_f = noise_decor[:, :, 0:1, :, :]
    #             noise = np.sqrt(self.model.bd_ratio) * noise_decor[:, :, 1:] + np.sqrt(
    #                 1 - self.model.bd_ratio) * noise_f
    #             img = torch.cat([noise_f, noise], dim=2)
    #     else:
    #         img = x_T
    #
    #     if timesteps is None:
    #         timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
    #     elif timesteps is not None and not ddim_use_original_steps:
    #         subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
    #         timesteps = self.ddim_timesteps[:subset_end]
    #
    #     intermediates = {'x_inter': [img], 'pred_x0': [img]}
    #     time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
    #     total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
    #     if verbose:
    #         iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    #     else:
    #         iterator = time_range
    #
    #     init_x0 = False
    #     clean_cond = kwargs.pop("clean_cond", False)
    #     for i, step in enumerate(iterator):
    #         if step < 500:
    #             free_u=False
    #         # step_replace_low=False
    #         index = total_steps - i - 1
    #         ts = torch.full((b,), step, device=device, dtype=torch.long)
    #
    #         # video2video
    #         if start_timesteps is not None:
    #             assert x0 is not None
    #             if step > start_timesteps*time_range[0]:
    #                 continue
    #             elif not init_x0:
    #                 print('start x0', step)
    #                 ts = ts-100
    #                 img = self.model.q_sample(x0, ts)
    #                 init_x0 = True
    #
    #         if replace_low:
    #             assert x0 is not None
    #             if step > start_timesteps*time_range[0]*0.5:
    #                 print('replace_low step', step)
    #                 xt = self.model.q_sample(x0, ts)
    #                 img = img - updown(img) + updown(xt)
    #                 # pass
    #             else:
    #                 pass
    #
    #         # dilate_conv
    #         if dilated_conv and dilated_conv_tau is not None:
    #             if (self.ddim_timesteps.shape[0]-index) > dilated_conv_tau[0]: # index is bigger -> small
    #                 # no need in later denoising
    #                 dilated_conv = False
    #                 print(f'index={index}, close dilated_conv')
    #             else:
    #                 print(f'index={index}, open dilated_conv')
    #
    #         # use mask to blend noised original latent (img_orig) & new sampled latent (img)
    #         if mask is not None:
    #             assert x0 is not None
    #             if clean_cond:
    #                 img_orig = x0
    #             else:
    #                 img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
    #             img = img_orig * mask + (1. - mask) * img # keep original & modify use img
    #
    #         index_clip =  int((1 - cond_tau) * total_steps)
    #         # print(f'index={index},index_clip={index_clip}')
    #         if index <= index_clip and target_size is not None:
    #             target_size_ = [target_size[0], target_size[1]//8, target_size[2]//8]
    #             img = torch.nn.functional.interpolate(
    #             img,
    #             size=target_size_,
    #             mode="nearest",
    #             )
    #         outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
    #                                   quantize_denoised=quantize_denoised, temperature=temperature,
    #                                   noise_dropout=noise_dropout, score_corrector=score_corrector,
    #                                   corrector_kwargs=corrector_kwargs,
    #                                   unconditional_guidance_scale=unconditional_guidance_scale,
    #                                   unconditional_conditioning=unconditional_conditioning,
    #                                   x0=x0, dilated_conv=dilated_conv,free_u=free_u,guidance_rescale=guidance_rescale,
    #                                   **kwargs)
    #
    #         img, pred_x0 = outs
    #         if callback: callback(i)
    #         if img_callback: img_callback(pred_x0, i)
    #
    #         if index % log_every_t == 0 or index == total_steps - 1:
    #             intermediates['x_inter'].append(img)
    #             intermediates['pred_x0'].append(pred_x0)
    #
    #     return img, intermediates

    # @torch.no_grad()
    # def p_sample_ddim_old(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   uc_type=None, conditional_guidance_scale_temporal=None,guidance_rescale=0.7,  **kwargs):
    #     b, *_, device = *x.shape, x.device
    #     if x.dim() == 5:
    #         is_video = True
    #     else:
    #         is_video = False
    #
    #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #         e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
    #     else:
    #         # with unconditional condition
    #         if isinstance(c, torch.Tensor):
    #             e_t = self.model.apply_model(x, t, c, **kwargs)
    #             e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
    #         elif isinstance(c, dict):
    #             e_t = self.model.apply_model(x, t, c, **kwargs)
    #             e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
    #         else:
    #             raise NotImplementedError
    #         # text cfg
    #         if uc_type is None:
    #             e_t_ori = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
    #             if guidance_rescale > 0:
    #                 from mira.models.utils_diffusion import rescale_noise_cfg
    #                 e_t = rescale_noise_cfg(e_t_ori, e_t, guidance_rescale=guidance_rescale)
    #         else:
    #             if uc_type == 'cfg_original':
    #                 e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
    #             elif uc_type == 'cfg_ours':
    #                 e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
    #             else:
    #                 raise NotImplementedError
    #
    #         # temporal guidance
    #         if conditional_guidance_scale_temporal is not None:
    #             e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
    #             e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
    #             e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)
    #
    #     if score_corrector is not None:
    #         assert self.model.parameterization == "eps"
    #         e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
    #
    #     alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    #     # select parameters corresponding to the currently considered timestep
    #
    #     if is_video:
    #         size = (b, 1, 1, 1, 1)
    #     else:
    #         size = (b, 1, 1, 1)
    #     a_t = torch.full(size, alphas[index], device=device)
    #     a_prev = torch.full(size, alphas_prev[index], device=device)
    #     sigma_t = torch.full(size, sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)
    #
    #     # current prediction for x_0
    #     if self.model.parameterization == 'eps':
    #         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #         eps=e_t
    #     elif self.model.parameterization == 'v':
    #         eps = self.model.predict_eps_from_z_and_v(x, t, e_t)
    #         pred_x0 = self.model.predict_start_from_z_and_v(x, t, e_t)
    #         # sqrt_alphas_cumprod_t =  torch.full(size, self.ddim_sqrt_alphas_cumprod[index], device=device)
    #         # sqrt_one_minus_alphas_cumprod_t = torch.full(size, self.ddim_sqrt_one_minus_alphas_cumprod[index], device=device)
    #         # pred_x0 = (sqrt_alphas_cumprod_t*x)-(sqrt_one_minus_alphas_cumprod_t*e_t)
    #         # sqrt_recip_alphas_cumprod_t  = torch.full(size, self.ddim_sqrt_recip_alphas_cumprod[index], device=device)
    #         # sqrt_recipm1_alphas_cumprod_t = torch.full(size, self.ddim_sqrt_recipm1_alphas_cumprod[index], device=device)
    #         # eps = (sqrt_recip_alphas_cumprod_t * x - pred_x0) / sqrt_recipm1_alphas_cumprod_t
    #     else:
    #         raise NotImplementedError("mu not supported")
    #
    #     # if self.model.use_dynamic_rescale:
    #     #     scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
    #     #     prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
    #     #     rescale = (prev_scale_t / scale_t)
    #     #     pred_x0 *= rescale
    #
    #     # print(f't={t}, pred_x0, min={torch.min(pred_x0)}, max={torch.max(pred_x0)}',file=f)
    #     if quantize_denoised:
    #         pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
    #     # direction pointing to x_t
    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * eps
    #     # # norm pred_x0
    #     # p=2
    #     # s=()
    #     # pred_x0 = pred_x0 - torch.max(torch.abs(pred_x0))
    #
    #     noise = noise_like(x.shape, device, repeat_noise)
    #     if self.model.bd_noise:
    #         noise_decor = self.model.bd(noise)
    #         noise_decor = (noise_decor - noise_decor.mean()) / (noise_decor.std() + 1e-5)
    #         noise_f = noise_decor[:, :, 0:1, :, :]
    #         noise = np.sqrt(self.model.bd_ratio) * noise_decor[:,:,1:] + np.sqrt(1 - self.model.bd_ratio) * noise_f
    #         noise = torch.cat([noise_f, noise], dim=2)
    #     noise = sigma_t *  noise * temperature
    #     if noise_dropout > 0.:
    #         noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    #
    #     # alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #     if self.use_scale:
    #         scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
    #         scale_t = torch.full(size, scale_arr[index], device=device)
    #         scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
    #         scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
    #         rescale = (scale_t_prev / scale_t)
    #         pred_x0 *= rescale
    #         # pred_x0 /= scale_t
    #         # x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
    #         x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #     else:
    #         x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #
    #     return x_prev, pred_x0
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                img_orig = x0
                # img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img  # keep original & modify use img
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      **kwargs)

            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            # log_every_t = 1
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      conditional_guidance_scale_temporal=None, **kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t_cfg = self.model.apply_model(x, t, c, **kwargs)  # unet denoiser
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError

            # text cfg
            e_t_cfg = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            if self.model.rescale_betas_zero_snr:
                from ..utils_diffusion import rescale_noise_cfg
                e_t_cfg = rescale_noise_cfg(e_t_cfg, e_t, guidance_rescale=0.7)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, e_t_cfg)
        else:
            e_t = e_t_cfg

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, e_t_cfg)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        if self.model.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
            rescale = (scale_t_prev / scale_t)
            pred_x0 *= rescale
            # pred_x0 /= scale_t
            # x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

# import torch
import torch.nn.functional as F
def updown(x):
    downsampled_x = F.interpolate(x, scale_factor=1/8, mode='nearest')
    recovered_x = F.interpolate(downsampled_x, scale_factor=8, mode='nearest')
    return recovered_x

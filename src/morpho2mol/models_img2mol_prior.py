import json
import os
import random
from functools import partial

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

# vd prior
# for prior
from dalle2_pytorch.dalle2_pytorch import (
    MLP,
    Attention,
    CausalTransformer,
    FeedForward,
    LayerNorm,
    NoiseScheduler,
    Rearrange,
    RelPosBias,
    RotaryEmbedding,
    SinusoidalPosEmb,
    default,
    exists,
    l2norm,
    prob_mask_like,
    rearrange,
    repeat,
)
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
from torch import einsum, nn
from torchvision import transforms
from tqdm.auto import tqdm


class DiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        clip=None,
        mol_embed_dim=None,
        mol_size=None,
        mol_channels=3,
        timesteps=1000,
        sample_timesteps=None,
        cond_drop_prob=0.0,
        morpho_cond_drop_prob=None,
        mol_cond_drop_prob=None,
        loss_type="l2",
        predict_x_start=True,
        predict_v=False,
        beta_schedule="cosine",
        condition_on_morpho_encodings=True,  # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed morpho embed -> mol embed training
        sampling_clamp_l2norm=False,  # whether to l2norm clamp the mol embed at each denoising iteration (analogous to -1 to 1 clipping for usual DDPMs)
        sampling_final_clamp_l2norm=False,  # whether to l2norm the final mol embedding output (this is also done for mols in ddpm)
        training_clamp_l2norm=False,
        init_mol_embed_l2norm=False,
        mol_embed_scale=None,  # this is for scaling the l2-normed mol embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        clip_adapter_overrides=dict(),
    ):
        super().__init__()

        self.sample_timesteps = sample_timesteps

        self.noise_scheduler = NoiseScheduler(beta_schedule=beta_schedule, timesteps=timesteps, loss_type=loss_type)

        if exists(clip):
            assert (
                mol_channels == clip.mol_channels
            ), f"channels of mol ({mol_channels}) should be equal to the channels that CLIP accepts ({clip.mol_channels})"

        else:
            assert exists(mol_embed_dim), "latent dimension must be given, if training prior network without CLIP given"
            self.clip = None

        self.net = net
        self.mol_embed_dim = default(mol_embed_dim, lambda: clip.dim_latent)

        assert (
            net.dim == self.mol_embed_dim
        ), f"your diffusion prior network has a dimension of {net.dim}, but you set your mol embedding dimension (keyword mol_embed_dim) on DiffusionPrior to {self.mol_embed_dim}"
        assert (
            not exists(clip) or clip.dim_latent == self.mol_embed_dim
        ), f"you passed in a CLIP to the diffusion prior with latent dimensions of {clip.dim_latent}, but your mol embedding dimension (keyword mol_embed_dim) for the DiffusionPrior was set to {self.mol_embed_dim}"

        self.channels = default(mol_channels, lambda: clip.mol_channels)

        self.morpho_cond_drop_prob = default(morpho_cond_drop_prob, cond_drop_prob)
        self.mol_cond_drop_prob = default(mol_cond_drop_prob, cond_drop_prob)

        self.can_classifier_guidance = self.morpho_cond_drop_prob > 0.0 and self.mol_cond_drop_prob > 0.0
        self.condition_on_morpho_encodings = condition_on_morpho_encodings

        # in paper, they do not predict the noise, but predict x0 directly for mol embedding, claiming empirically better results. I'll just offer both.

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v  # takes precedence over predict_x_start

        self.mol_embed_scale = default(mol_embed_scale, self.mol_embed_dim**0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling

        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm

        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_mol_embed_l2norm = init_mol_embed_l2norm

        # device tracker

        self.register_buffer("_dummy", torch.tensor([True]), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, mol_embed):
        return l2norm(mol_embed) * self.mol_embed_scale

    def p_mean_variance(self, x, t, morpho_cond, self_cond=None, clip_denoised=False, cond_scale=1.0):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        pred = self.net.forward_with_cond_scale(x, t, cond_scale=cond_scale, self_cond=self_cond, **morpho_cond)

        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1.0, 1.0)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.mol_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, morpho_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.0):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, morpho_cond=morpho_cond, self_cond=self_cond, clip_denoised=clip_denoised, cond_scale=cond_scale
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, morpho_cond, cond_scale=1.0):
        batch, device = shape[0], self.device

        mol_embed = torch.randn(shape, device=device)
        x_start = None  # for self-conditioning

        if self.init_mol_embed_l2norm:
            mol_embed = l2norm(mol_embed) * self.mol_embed_scale

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            mol_embed, x_start = self.p_sample(
                mol_embed, times, morpho_cond=morpho_cond, self_cond=self_cond, cond_scale=cond_scale
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            mol_embed = self.l2norm_clamp_embed(mol_embed)

        return mol_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape, morpho_cond, *, timesteps, eta=1.0, cond_scale=1.0):
        batch, device, alphas, total_timesteps = (
            shape[0],
            self.device,
            self.noise_scheduler.alphas_cumprod_prev,
            self.noise_scheduler.num_timesteps,
        )

        times = torch.linspace(-1.0, total_timesteps, steps=timesteps + 1)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        mol_embed = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if self.init_mol_embed_l2norm:
            mol_embed = l2norm(mol_embed) * self.mol_embed_scale

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None

            pred = self.net.forward_with_cond_scale(
                mol_embed, time_cond, self_cond=self_cond, cond_scale=cond_scale, **morpho_cond
            )

            # derive x0

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(mol_embed, t=time_cond, v=pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(mol_embed, t=time_cond, noise=pred)

            # clip x0 before maybe predicting noise

            if not self.predict_x_start:
                x_start.clamp_(-1.0, 1.0)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise

            pred_noise = self.noise_scheduler.predict_noise_from_start(mol_embed, t=time_cond, x0=x_start)

            if time_next < 0:
                mol_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(mol_embed) if time_next > 0 else 0.0

            mol_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            mol_embed = self.l2norm_clamp_embed(mol_embed)

        return mol_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_mol_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_mol_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps=timesteps)

        mol_embed = normalized_mol_embed / self.mol_embed_scale
        return mol_embed

    def p_losses(self, mol_embed, times, morpho_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(mol_embed))

        mol_embed_noisy = self.noise_scheduler.q_sample(x_start=mol_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(mol_embed_noisy, times, **morpho_cond).detach()

        pred = self.net(
            mol_embed_noisy,
            times,
            self_cond=self_cond,
            morpho_cond_drop_prob=self.morpho_cond_drop_prob,
            mol_cond_drop_prob=self.mol_cond_drop_prob,
            **morpho_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(mol_embed, times, noise)
        elif self.predict_x_start:
            target = mol_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    def sample_batch_size(self, batch_size, morpho_cond, cond_scale=1.0):
        device = self.betas.device
        shape = (batch_size, self.mol_embed_dim)

        img = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                morpho_cond=morpho_cond,
                cond_scale=cond_scale,
            )
        return img

    @torch.no_grad()
    def sample(self, morpho, num_samples_per_batch=2, cond_scale=1.0, timesteps=None):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 mol embeddings, choose the top 1 similarity, as judged by CLIP
        morpho = repeat(morpho, "b ... -> (b r) ...", r=num_samples_per_batch)

        batch_size = morpho.shape[0]
        mol_embed_dim = self.mol_embed_dim

        morpho_embed, morpho_encodings = self.clip.embed_morpho(morpho)

        morpho_cond = dict(morpho_embed=morpho_embed)

        if self.condition_on_morpho_encodings:
            morpho_cond = {**morpho_cond, "morpho_encodings": morpho_encodings}

        mol_embeds = self.p_sample_loop(
            (batch_size, mol_embed_dim), morpho_cond=morpho_cond, cond_scale=cond_scale, timesteps=timesteps
        )

        # retrieve original unscaled mol embed

        morpho_embeds = morpho_cond["morpho_embed"]

        morpho_embeds = rearrange(morpho_embeds, "(b r) d -> b r d", r=num_samples_per_batch)
        mol_embeds = rearrange(mol_embeds, "(b r) d -> b r d", r=num_samples_per_batch)

        morpho_mol_sims = einsum("b r d, b r d -> b r", l2norm(morpho_embeds), l2norm(mol_embeds))
        top_sim_indices = morpho_mol_sims.topk(k=1).indices

        top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=mol_embed_dim)

        top_mol_embeds = mol_embeds.gather(1, top_sim_indices)
        return rearrange(top_mol_embeds, "b 1 d -> b d")

    def forward(
        self,
        morpho=None,
        mol=None,
        morpho_embed=None,  # allow for training on preprocessed CLIP morpho and mol embeddings
        mol_embed=None,
        morpho_encodings=None,  # as well as CLIP morpho encodings
        *args,
        **kwargs,
    ):
        assert exists(morpho) ^ exists(morpho_embed), "either morpho or morpho embedding must be supplied"
        assert exists(mol) ^ exists(mol_embed), "either mol or mol embedding must be supplied"
        assert not (
            self.condition_on_morpho_encodings and (not exists(morpho_encodings) and not exists(morpho))
        ), "morpho encodings must be present if you specified you wish to condition on it on initialization"

        if exists(mol):
            mol_embed, _ = self.clip.embed_mol(mol)

        # calculate morpho conditionings, based on what is passed in

        if exists(morpho):
            morpho_embed, morpho_encodings = self.clip.embed_morpho(morpho)

        morpho_cond = dict(morpho_embed=morpho_embed)

        if self.condition_on_morpho_encodings:
            assert exists(morpho_encodings), "morpho encodings must be present for diffusion prior if specified"
            morpho_cond = {**morpho_cond, "morpho_encodings": morpho_encodings}

        # timestep conditioning from ddpm

        batch, device = mol_embed.shape[0], mol_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # scale mol embed (Katherine)

        mol_embed *= self.mol_embed_scale

        # calculate forward loss

        return self.p_losses(mol_embed, times, morpho_cond=morpho_cond, *args, **kwargs)


class Img2MolDiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_mol_embeds=1,
        num_morpho_embeds=1,
        max_morpho_len=256,
        self_cond=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.num_time_embeds = num_time_embeds
        self.num_mol_embeds = num_mol_embeds
        self.num_morpho_embeds = num_morpho_embeds

        self.to_morpho_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_morpho_embeds) if num_morpho_embeds > 1 else nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_morpho_embeds),
        )

        self.continuous_embedded_time = not exists(num_timesteps)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)
            ),  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )

        self.to_mol_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_mol_embeds) if num_mol_embeds > 1 else nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_mol_embeds),
        )

        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)

        # dalle1 learned padding strategy

        self.max_morpho_len = max_morpho_len

        self.null_morpho_encodings = nn.Parameter(torch.randn(1, max_morpho_len, dim))
        self.null_morpho_embeds = nn.Parameter(torch.randn(1, num_morpho_embeds, dim))
        self.null_mol_embed = nn.Parameter(torch.randn(1, dim))

        # whether to use self conditioning, Hinton's group's new ddpm technique

        self.self_cond = self_cond

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, morpho_cond_drop_prob=1.0, mol_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        mol_embed,
        diffusion_timesteps,
        *,
        morpho_embed,
        morpho_encodings=None,
        self_cond=None,
        morpho_cond_drop_prob=0.0,
        mol_cond_drop_prob=0.0,
    ):
        batch, dim, device, dtype = *mol_embed.shape, mol_embed.device, mol_embed.dtype

        num_time_embeds, num_mol_embeds, num_morpho_embeds = (
            self.num_time_embeds,
            self.num_mol_embeds,
            self.num_morpho_embeds,
        )

        # setup self conditioning

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros(batch, self.dim, device=device, dtype=dtype))
            self_cond = rearrange(self_cond, "b d -> b 1 d")

        # in section 2.2, last paragraph
        # "... consisting of encoded morpho, CLIP morpho embedding, diffusion timestep embedding, noised CLIP mol embedding, final embedding for prediction"

        morpho_embed = self.to_morpho_embeds(morpho_embed)
        mol_embed = self.to_mol_embeds(mol_embed)

        # classifier free guidance masks

        morpho_keep_mask = prob_mask_like((batch,), 1 - morpho_cond_drop_prob, device=device)
        morpho_keep_mask = rearrange(morpho_keep_mask, "b -> b 1 1")

        mol_keep_mask = prob_mask_like((batch,), 1 - mol_cond_drop_prob, device=device)
        mol_keep_mask = rearrange(mol_keep_mask, "b -> b 1 1")

        # make morpho encodings optional
        # although the paper seems to suggest it is present <--

        if not exists(morpho_encodings):
            morpho_encodings = torch.empty((batch, 0, dim), device=device, dtype=dtype)

        mask = torch.any(morpho_encodings != 0.0, dim=-1)

        # replace any padding in the morpho encodings with learned padding tokens unique across position

        morpho_encodings = morpho_encodings[:, : self.max_morpho_len]
        mask = mask[:, : self.max_morpho_len]

        morpho_len = morpho_encodings.shape[-2]
        remainder = self.max_morpho_len - morpho_len

        if remainder > 0:
            morpho_encodings = F.pad(morpho_encodings, (0, 0, 0, remainder), value=0.0)
            mask = F.pad(mask, (0, remainder), value=False)

        # mask out morpho encodings with null encodings

        null_morpho_encodings = self.null_morpho_encodings.to(morpho_encodings.dtype)

        morpho_encodings = torch.where(
            rearrange(mask, "b n -> b n 1").clone() & morpho_keep_mask, morpho_encodings, null_morpho_encodings
        )

        # mask out morpho embeddings with null morpho embeddings

        null_morpho_embeds = self.null_morpho_embeds.to(morpho_embed.dtype)

        morpho_embed = torch.where(morpho_keep_mask, morpho_embed, null_morpho_embeds)

        # mask out mol embeddings with null mol embeddings

        null_mol_embed = self.null_mol_embed.to(mol_embed.dtype)

        mol_embed = torch.where(mol_keep_mask, mol_embed, null_mol_embed)

        # whether morpho embedding is used for conditioning depends on whether morpho encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)

        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch)

        if self.self_cond:
            learned_queries = torch.cat((self_cond, learned_queries), dim=-2)

        tokens = torch.cat((morpho_encodings, morpho_embed, time_embed, mol_embed, learned_queries), dim=-2)

        # attend

        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the mol embedding (per DDPM timestep)

        pred_mol_embed = tokens[..., -1, :]

        return pred_mol_embed


class Img2MolDiffusionPrior(DiffusionPrior):
    """Differences from original:

    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def p_sample(self, x, t, morpho_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.0, generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, morpho_cond=morpho_cond, self_cond=self_cond, clip_denoised=clip_denoised, cond_scale=cond_scale
        )
        if generator is None:
            noise = torch.randn_like(x)
        else:
            # noise = torch.randn_like(x)
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, morpho_cond, cond_scale=1.0, generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            mol_embed = torch.randn(shape, device=device)
        else:
            mol_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None  # for self-conditioning

        if self.init_mol_embed_l2norm:
            mol_embed = l2norm(mol_embed) * self.mol_embed_scale

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
            disable=True,
        ):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            mol_embed, x_start = self.p_sample(
                mol_embed,
                times,
                morpho_cond=morpho_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                generator=generator,
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            mol_embed = self.l2norm_clamp_embed(mol_embed)

        return mol_embed

    def p_losses(self, mol_embed, times, morpho_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(mol_embed))

        mol_embed_noisy = self.noise_scheduler.q_sample(x_start=mol_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(mol_embed_noisy, times, **morpho_cond).detach()

        pred = self.net(
            mol_embed_noisy,
            times,
            self_cond=self_cond,
            morpho_cond_drop_prob=self.morpho_cond_drop_prob,
            mol_cond_drop_prob=self.mol_cond_drop_prob,
            **morpho_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(mol_embed, times, noise)
        elif self.predict_x_start:
            target = mol_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss, pred

    def forward(
        self,
        morpho=None,
        mol=None,
        morpho_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
        mol_embed=None,
        morpho_encodings=None,  # as well as CLIP text encodings
        *args,
        **kwargs,
    ):
        assert exists(morpho) ^ exists(morpho_embed)
        assert exists(mol) ^ exists(mol_embed), "either image or image embedding must be supplied"
        assert not (
            self.condition_on_morpho_encodings and (not exists(morpho_encodings) and not exists(morpho))
        ), "text encodings must be present if you specified you wish to condition on it on initialization"

        # calculate text conditionings, based on what is passed in

        morpho_cond = dict(morpho_embed=morpho_embed)

        if self.condition_on_morpho_encodings:
            assert exists(morpho_encodings), "text encodings must be present for diffusion prior if specified"
            morpho_cond = {**morpho_cond, "text_encodings": morpho_encodings}

        # timestep conditioning from ddpm

        batch, device = mol_embed.shape[0], mol_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(mol_embed * self.mol_embed_scale, times, morpho_cond=morpho_cond, *args, **kwargs)

        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

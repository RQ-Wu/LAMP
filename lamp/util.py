import os
import imageio
import numpy as np
from typing import Union
from PIL import Image

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, 'GIF', duration=1/fps, loop=0)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def calc_img_clip_score(model, img, txt, preprocess, tokenizer):
    logit_scale = model.logit_scale.exp()

    if img.max() <= 1:
        img = (img * 255.0).cpu().numpy().astype(np.uint8)
    else:
        img = (img).cpu().numpy().astype(np.uint8)
    img = Image.fromarray(img)
    
    # import ipdb
    # ipdb.set_trace()
    img_features = model.encode_image(preprocess(img).unsqueeze(0).to('cuda'))
    text_features = model.encode_text(tokenizer(txt).to('cuda'))

    # normalize features
    img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
    text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)
    
    # calculate scores
    score = logit_scale * (text_features * img_features).sum()
    
    return score.cpu().item()

def calc_video_clip_score(model, video, txt, preprocess, tokenizer):
    score_acc = 0
    for image in video:
        score_acc += calc_img_clip_score(model, image, txt, preprocess, tokenizer)
    
    return score_acc / len(video)

def calc_video_consistency(model, video, preprocess):
    logit_scale = model.logit_scale.exp()
    feature_list = []
    score = 0.0
    count = 0
    for img in video:
        if img.max() <= 1:
            img = (img * 255.0).cpu().numpy().astype(np.uint8)
        else:
            img = (img).cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img_features = model.encode_image(preprocess(img).unsqueeze(0).to('cuda'))
        img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
        feature_list.append(img_features)
    
    for i in range(len(video)):
        for j in range(i):
            count += 1
            score += logit_scale * (feature_list[i] * feature_list[j]).sum()
    
    return score.cpu().item() / count

def calc_video_diversity(model, videos, preprocess):
    logit_scale = model.logit_scale.exp()
    feature_list = []
    for video in videos:
        video_features = None
        for img in video:
            if img.max() <= 1:
                img = (img * 255.0).cpu().numpy().astype(np.uint8)
            else:
                img = (img).cpu().numpy().astype(np.uint8)
            img = Image.fromarray(img)
            img_features = model.encode_image(preprocess(img).unsqueeze(0).to('cuda'))
            img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
            if video_features is None:
                video_features = img_features
            else:
                video_features = video_features + img_features
        video_features = video_features / 16.0
        feature_list.append(video_features)

    count = 0
    score = 0.0
    for i in range(len(videos)):
        for j in range(i):
            count += 1
            score += logit_scale * (feature_list[i] * feature_list[j]).sum()

    return score.cpu().item() / count


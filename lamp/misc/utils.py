from tying import Union
import os

from einops import rearrange
from PIL import Image
from tqdm import tqdm
import torchvision
import numpy as np
import imageio
import torch


## prompt 토큰의 임베딩 구해주는 함수.
get_embeddings        = lambda pipeline, input_: pipeline.text_encoder(input_.input_ids.to(pipeline.device))[0]


## 추론 이미지에 노이즈를 추가해주는 함수.
get_noise_pred_single = lambda latents, t, context, unet: unet(latents, t, encoder_hidden_states = context)['sample'] 


def save_videos_grid(videos: torch.Tensor, path: str, 
                     rescale: bool = False, n_rows: int = 4, fps: int = 8):

    ## video 텐서의 (batch, channel, time, height, width)를
    ## (time, batch, channel, height, width)로 변경
    videos  = rearrange(videos, 'b c t h w -> t b c h w')
    outputs = []

    for video in videos:

        ## 비디오를 4
        video = torchvision.utils.make_grid(video, nrows = n_rows)
        video = video.transpose(0, 2).squeeze(-1)

        ## [-1, 1] 구간에 있던 픽셀 값들을 [0, 1] 구간에 있도록 조정.
        if rescale: video = (video + 1.0) / 2.0

        ## [0, 1] 구간에 있는 픽셀 값들을 [0, 255] 구간에 있도록 조정하고,
        ## 정수형 픽셀로 변환.
        video = (video * 255).numpy().astype(np.uint8)
        outputs.append(video)

        ## gif 포맷의 이미지가 저장되는 폴더 생성.
        os.makedirs(os.path.dirname(path, exist_ok = True))

        ## gif 포맷의 이미지 저장.
        imageio.mimsave(path, outputs, 'GIF', dutation = 1/fps, loop = 0)


@torch.no_grad()
def init_prompt(prompt: str, pipeline) -> torch.Tensor:

    uncond_prompt     = pipeline.tokenizer(
                            [''], padding  = 'max_length', max_length=pipeline.tokenizer.model_max_length,
                            return_tensors = 'pt'
                        )
    text_prompt       = pipeline.tokenizer(
                            [prompt], padding = 'max_length', truncation = True,
                            return_tensors = 'pt'
                        )

    ## uncond prompt의 토큰과 text prompt의 토큰 임베딩을 구해줌.
    uncond_embeddings = get_embeddings(pipeline, uncond_prompt)
    text_embeddings   = get_embeddings(pipeline, text_prompt)

    ## uncond embedding과 text embedding을 하나로 합침.
    return torch.cat([uncond_embeddings, text_embeddings])


def next_step(model_output: Union[torch.FloatTensor, np.ndarray],
            sample: Union[torch.FloatTensor, np.ndarray],
            timestep: int, ddim_scheduler) -> Union[torch.FloatTensor, np.ndarray]:

    timestep, next_timestep = min(
            timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] \
                    if timestep >= 0 else ddim_scheduler.final_alpha_cumpod
    beta_prod_t  = 1 - alpha_prod_t

    alpha_prot_t_next     = ddim_scheduler.alphas_cumprod[next_timestep]
    next_original_sample  = (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output

    return alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction 
     

@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):

    contenxt                           = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)

    all_latent = [latent]
    latent     = latent.clone.detach()

    for idx in tqdm(range(num_inv_steps)):

        t          = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - idx - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent     = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)

    return all_latent


## 이미지 전처리 함수
def preprocessing_image(image: torch.Tensor) -> PIL.Image.Image:

    if image.max() <= 1: image = (image * 255.).cpu().numpy().astype(np.uint8)
    else: image  = image.cpu().numpy().astype(np.uint8) 

    return Image.fromarray(image)


## DDIM 역연산 함수
#! DDIM 이란!? : https://jang-inspiration.com/ddim
@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent,
                    num_inv_steps, prompt = '') -> list:

    return ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)


## 이미지 용 CLIP score 계산 함수
#! CLIP score란!? : https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html
def calc_img_clip_score(model, image, text, preprocess, tokenizer):

    logit_scale  = model.logit_scale.exp()
    image        = preprocessing_image(image)
    image_feats  = model.encode_image(preprocess(image).unsqueeze(0).to('cuda'))
    text_feats   = model.encode_text(tokenizer(text).to('cuda'))

    ## 이미지, 텍스트 feature 정규화
    image_feats /= image_feats.norm(dim = 1, keepdim = True).to(torch.float32)
    text_feats  /=  text_feats.norm(dim = 1, keepdim = True).to(torch.float32)  

    score = logit_score * (text_feats * image_feats).sum()
    return score.cpu().item()


## 비디오 용 CLIP score 계산 함수
def calc_video_clip_score(model, video, text, preprocess, tokenizer):

    ## 비디오 프레임 별 clip score를 구해 평균내는 방식으로 비디오 clip score 계산
    scores = [calc_img_clip_score(model, image, text, preprocess, tokenizer) for image in video]

    return sum(scores) / len(scores)


def calc_video_consistency(model, video, preprocess):

    logit_scale  = model.logit_scale.exp()
    feature_list = []
    score, count = 0.0, 0

    for image in video: 

        image        = preprocessing_image(image)
        image_feats  = model.encode_image(preprocess(image).unsqueeze(0).to('cuda'))
        image_feats /= image_feats.norm(dim = 1, keepdim = True).to(torch.float32)

        feature_list.append(image_feats)

        for _1dx in range(len(video)):
             
             for _2dx in range(_1dx):

                count += 1
                score += logit_scale * (feature_list[_1dx] * feature_list[_2dx]).sum()

    return score.cpu().item() / count


def calc_video_diversity(model, videos, preprocess):

    logit_scale  = model.logit_scale.exp()
    feature_list = []

    for video in videos:

        video_feats = None
        for frame in video:

            frame        = preprocessing_image(frame)
            frame_feats  = model.encode_image(preprocess(frame).unsqueeze(0).to('cuda'))
            frame_feats /= frame_feats.norm(dim = 1, keepdim = True).to(torch_float32)

            video_feats  = video_feats + frame_feats if video_feats is None else frame_feats

        video_feats = video_feats / 16.0
        feature_list.append(video_feats)

    count, score = 0, 0.0
    for _1dx in range(len(videos)):

        for _2dx in range(_1dx):
            count += 1
            score += logit_scale * (feature_list[_1dx] * feautre_list[_2dx]).sum()

    return score.cpu().item() / count
        
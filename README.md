# <p align=center> :movie_camera: `LAMP: Learn A Motion Pattern by Few-Shot Tuning a Text-to-Image Diffusion Model`</p>

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.1-blue.svg)

This repository is the official implementation of [LAMP]()

> **LAMP: Learn A Motion Pattern by Few-Shot Tuning a Text-to-Image Diffusion Model**<br>
> Ruiqi Wu, Linagyu Chen, Tong Yang, Chunle Guo, Chongyi Li, Xiangyu Zhang 
><br>( * indicates corresponding author)

[[Arxiv Paper (TBD)]  [中文版 (TBD)] [[Website Page](https://rq-wu.github.io/projects/LAMP/index.html)]

![method](assets/method.png)

:rocket: LAMP is a **few-shot-based** method for text-to-video generation. You only need **8~16 videos 1 GPU (> 15 GB VRAM)** for training!! Then you can generate videos with learned motion pattern.

## Visual Examples
### Few-Shot-Based Text-to-Video Generation
<table class="center">
    <tr>
        <td align="center" style="width: 7%"> Horse run</td>
        <td align="center">
            <img src="assets/inference-a_horse_runs_in_the_universe (1).gif">
        </td>
        <td align="center">
            <img src="assets/inference-a_horse_runs_on_the_Mars (3).gif">
        </td>
        <td align="center">
            <img src="assets/inference-a_horse_runs_on_the_road (1).gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"> </td>
        <td align="center">A horse runs in the universe.</td>
        <td align="center">A horse runs on the Mars.</td>
        <td align="center">A horse runs on the road.</td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Firework</td>
        <td align="center">
            <img src="assets/inference-fireworks_in_desert_night.gif">
        </td>
        <td align="center">
            <img src="assets/inference-fireworks_over_the_mountains (1).gif">
        </td>
        <td align="center">
            <img src="assets/inference-fireworks_in_the_night_city.gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"> </td>
        <td align="center">Fireworks in desert night.</td>
        <td align="center">Fireworks over the mountains.</td>
        <td align="center">Fireworks in the night city.</td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Play the guitar</td>
        <td align="center">
            <img src="assets/inference-GTA5_poster,_a_man_plays_the_guitar.gif">
        </td>
        <td align="center">
            <img src="assets/inference-a_woman_plays_the_guitar (1).gif">
        </td>
        <td align="center">
            <img src="assets/inference-an_astronaut_plays_the_guitar,_photorelastic.gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"> </td>
        <td align="center">GTA5 poster, a man plays the guitar.</td>
        <td align="center">A woman plays the guitar.</td>
        <td align="center">An astronaut plays the guitar, photorelastic.</td>
    </tr>
    <tr>
        <td align="center" style="width: 7%"> Birds fly</td>
        <td align="center">
            <img src="assets/inference-birds_fly_in_the_pink_sky.gif">
        </td>
        <td align="center">
            <img src="assets/inference-birds_fly_in_the_sky,_over_the_sea.gif">
        </td>
        <td align="center">
            <img src="assets/inference-many_birds_fly_over_a_plaza.gif">
        </td>
    </tr>
    <tr class="prompt-row">
        <td align="center" style="width: 7%"></td>
        <td align="center">Birds fly in the pink sky.</td>
        <td align="center">Birds fly in the sky, over the sea.</td>
        <td align="center">Many Birds fly over a plaza.</td>
    </tr>
<table>

### Video Editing
<table style="width: 100%;">
    <tbody>
        <tr class="prompt-row">
            <td align="center"> Origin Videos </td>
            <td align="center"> Editing Result-1</td>
            <td align="center"> Editing Result-2</td>
        </tr>
        <tr class="result-row">
            <td align="center">
                <img src="assets/run.gif">
            </td>
            <td align="center">
                <img src="assets/inference-a girl in black runs on the road.gif">
            </td>
            <td align="center">
                <img src="assets/inference-a man runs on the road.gif">
            </td>
        </tr>
        <tr class="prompt-row">
            <td align="center"></td>
            <td align="center">A girl in black runs on the road.</td>
            <td align="center">A man runs on the road.</td>
        </tr>
        <tr class="result-row">
            <td align="center">
                <img src="assets/dance.gif">
            </td>
            <td align="center">
                <img src="assets/inference-a man is dancing.gif">
            </td>
            <td align="center">
                <img src="assets/inference-a girl in white is dancing.gif">
            </td>
        </tr>
        <tr class="prompt-row">
            <td align="center"></td>
            <td align="center">A man is dancing.</td>
            <td align="center">A girl in white is dancing.</td>
        </tr>
    </tbody>
</table>
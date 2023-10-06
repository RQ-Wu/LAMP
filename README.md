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
<table style="width: 100%; text-align:center">
    <tbody>
        <tr class="result-row">
            <td style="width: 7%"> Horse run</td>
            <td>
                <img src="assets/inference-a_horse_runs_in_the_universe (1).gif">
            </td>
            <td>
                <img src="assets/inference-a_horse_runs_on_the_Mars (3).gif">
            </td>
            <td>
                <img src="assets/inference-a_horse_runs_on_the_road (1).gif">
            </td>
        </tr>
        <tr class="prompt-row">
            <td style="width: 7%"> </td>
            <td>A horse runs in the universe.</td>
            <td>A horse runs on the Mars.</td>
            <td>A horse runs on the road.</td>
        </tr>
        <tr class="result-row">
            <td style="width: 7%"> Firework</td>
            <td>
                <img src="assets/inference-fireworks_in_desert_night.gif">
            </td>
            <td>
                <img src="assets/inference-fireworks_over_the_mountains (1).gif">
            </td>
            <td>
                <img src="assets/inference-fireworks_in_the_night_city.gif">
            </td>
        </tr>
        <tr class="prompt-row">
            <td style="width: 7%"> </td>
            <td>Fireworks in desert night.</td>
            <td>Fireworks over the mountains.</td>
            <td>Fireworks in the night city.</td>
        </tr>
        <tr class="result-row">
            <td style="width: 7%"> Play the guitar</td>
            <td>
                <img src="assets/inference-GTA5_poster,_a_man_plays_the_guitar.gif">
            </td>
            <td>
                <img src="assets/inference-a_woman_plays_the_guitar (1).gif">
            </td>
            <td>
                <img src="assets/inference-an_astronaut_plays_the_guitar,_photorelastic.gif">
            </td>
        </tr>
        <tr class="prompt-row">
            <td style="width: 7%"> </td>
            <td>GTA5 poster, a man plays the guitar.</td>
            <td>A woman plays the guitar.</td>
            <td>An astronaut plays the guitar, photorelastic.</td>
        </tr>
        <tr class="result-row">
            <td style="width: 7%"> Birds fly</td>
            <td>
                <img src="assets/inference-birds_fly_in_the_pink_sky.gif">
            </td>
            <td>
                <img src="assets/inference-birds_fly_in_the_sky,_over_the_sea.gif">
            </td>
            <td>
                <img src="assets/inference-many_birds_fly_over_a_plaza.gif">
            </td>
        </tr>
        <tr class="prompt-row">
            <td style="width: 7%"></td>
            <td>Birds fly in the pink sky.</td>
            <td>Birds fly in the sky, over the sea.</td>
            <td>Many Birds fly over a plaza.</td>
        </tr>
    </tbody>
<table>

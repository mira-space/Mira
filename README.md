
<p align="center">
  <img src="assets/readme/miralogo.png" height=80>
</p>

#  Mira: A Mini-step Towards Sora-like Long Video Generation


> [Zhaoyang Zhang](https://zzyfd.github.io/)<sup>1*</sup>, [Ziyang Yuan](https://github.com/jiangyzy)<sup>1*</sup>, [Xuan Ju](https://github.com/juxuan27)<sup>1</sup>, [Yiming Gao](https://scholar.google.com/citations?user=uRCc-McAAAAJ&hl=zh-TW)<sup>1</sup>, [Xintao Wang](https://xinntao.github.io/)<sup>1#</sup>,  [Chun Yuan](https://scholar.google.com/citations?hl=en&user=fYdxi2sAAAAJ), [Ying Shan](https://www.linkedin.com/in/YingShanProfile/)<sup>1</sup>, <br>
> <sup>1</sup>ARC Lab, Tencent PCG <sup>*</sup>Equal contribution  <sup>#</sup>Project lead


    
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://mira-space.github.io/)
[![MiraData Page](https://img.shields.io/badge/MiraData-Page-blue)](https://github.com/mira-space/MiraData)


We introduce Mira (Mini-Sora), an initial foray into the realm of high-quality, long-duration video generation in the style of Sora. Sora stands out from existing text-to-video (T2V) generation frameworks in several key ways:

* Extended sequence length: While most frameworks are limited to generating short videos (2 seconds / 16 frames), Sora is designed to produce significantly longer sequences, potentially lasting 10 seconds, 20 seconds, or more.

* Enhanced dynamics: Sora has the capability to create videos with rich dynamics and intricate motions, setting it apart from the more static outputs of current video generation technologies.

* Interactive objects and environments: Sora supports the generation of videos where objects and surroundings engage in dynamic interactions, adding a layer of complexity and realism. 

* Strong 3D consistency: Despite the intricate dynamics and object interactions, Sora ensures the 3D integrity of objects is preserved throughout the video, avoiding noticeable distortions.

* Sustained object consistency: Sora maintains consistent object shapes, even when they temporarily exit and re-enter the frame, ensuring continuity and coherence.



The Mira project is our endeavor to investigate and refine the entire data-model-training pipeline for Sora-like, lightweight T2V frameworks, and to preliminarily demonstrate the aforementioned Sora characteristics. Our goal is to foster innovation and democratize the field of content creation, paving the way for more accessible and advanced video generation tools.





## Results

**10s 384×240**





https://github.com/mira-space/Mira/assets/13939478/4de6aade-4eca-4291-bcc6-950c7b44c981

Each individual video can be downloaded from [here](https://drive.google.com/drive/folders/1-GdDOQ3r0_FimMsH-uQaQgOYzrxXaEa8?usp=drive_link).


**20s 128×80**   




https://github.com/mira-space/Mira/assets/13939478/9f274503-9715-4d2a-a262-10113c4df78f








## 📰 Updates

**Stay tuned!**  We are actively working on this project. Expect a steady stream of updates as we expand our dataset, enhance our annotation processes, and refine our model checkpoints. Keep an eye out for these upcoming updates, as we continue to make strides in our project's development.

**[2024.04.01]** 🔥 We're delighted to announce the release of **Mira** and **MiraData-v0**. This  release offers a comprehensive open-source suite for data annotation and training pipelines, specifically tailored for the creation of long-duration videos with dynamic content and consistent quality. Our provided codes and checkpoints empower users to generate videos up to 20 seconds in 128x80 resolution and 10 seconds in 384x240 resolution. Dive into the future of video generation with Mira!






## Installation
```bash
## create a conda enviroment
conda update -n base -c defaults conda 
conda create -y -n mira python=3.8.5 
source activate mira 

## install dependencies
pip install torch==2.0 torchvision torchaudio decord==0.6.0  \
einops==0.3.0  imageio==2.9.0 \
numpy omegaconf==2.1.1 opencv_python pandas \
Pillow==9.5.0 pytorch_lightning==1.9.0 PyYAML==6.0 setuptools==65.6.3  \
torchvision tqdm==4.65.0 transformers==4.25.1 moviepy av  tensorboardx \
&& pip install  timm scikit-learn  open_clip_torch==2.22.0 kornia simplejson easydict pynvml rotary_embedding_torch==0.3.1 triton  cached_property  \
&& pip install xformers==0.0.18 \
&& pip install taming-transformers fairscale deepspeed  diffusers
```

## Training

### Checkpoints

| Name | Model Size | Data | Resolution |   
| ---- | ---- | ---- | ---- |
| [128-v0.pt](https://huggingface.co/TencentARC/Mira-v0) | 1.1B | Webvid(pretrain) + MiraData-v0 | 128x80, 120 frames |
| [384-v0.pt](https://huggingface.co/TencentARC/Mira-v0) | 1.1B | Webvid(pretrain) + MiraData-v0 | 384x240, 60 frames |

Please download the above checkponits in our [huggingface page](https://huggingface.co/TencentARC/Mira-v0). 

### Finetuning the Mira-v0 model on 128x80 resolution.

* Add path to your datasets and the pretrain models in [config_384_mira.yaml](configs/Mira/config_384_mira.yaml).
* Then conduct the following commands:

```bash
## activate envrionment
conda activate mira


## Run training
bash configs/Mira/run_128_mira.sh 0
```

### Finetuning the Mira-v0 model on 384x240 resolution.

* Add path to your datasets and the pretrain models in [config_128_mira.yaml](configs/Mira/config_128_mira.yaml).
* Then conduct the following commands:
  
```bash
## activate envrionment
conda activate mira

## Run training
bash configs/Mira/run_384_mira.sh 0
```

## Inference

###  Evaluate the Mira-v0 model on 128x80 resolution.

* Add path to your model checkponits in [run_text2video.sh](configs/inference/run_text2video.sh).
* Add your test prompts in [test_prompt.txt](prompts/test_prompt.txt).
* Then conduct the following commands:
  
```bash
## activate envrionment
conda activate mira

## Run inference
bash configs/inference/run_text2video.sh

```

### Evaluate the Mira-v0 model on 384x240 resolution.

* Add path to your model checkponits in [run_text2video_384.sh](configs/inference/run_text2video_384.sh).
* Add your test prompts in [test_prompt.txt](prompts/test_prompt.txt).
* Then conduct the following commands:
  
```bash
## activate envrionment
conda activate mira

## Run inference
bash configs/inference/run_text2video_384.sh

```

## Current Limitations
Mira-v0 represents our initial exploration into developing a Sora-like Text-to-Video (T2V) pipeline. Through this process, we have identified several areas for improvement in the current version:

* **Enhanced motion dynamics and scene intricacy at the expense of generic object generation.** The Mira-v0 model, being fine-tuned on the potentially limited MiraData-v0, shows a reduced capability in generating a diverse range of objects compared to the WebVid-pretrained MiraDiT. However, it's worth noting that the Mira-v0 model has shown notable advancements in motion dynamics, scene detail, and three-dimensional consistency.

| 10s 384x240 | 10s 384x240  |  10s 384x240| 
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  [<img src="https://github.com/mira-space/Mira/assets/13939478/b7e16946-04ec-438d-8df8-8bf0da6200e1" width="300">](https://github.com/mira-space/Mira/assets/13939478/b7e16946-04ec-438d-8df8-8bf0da6200e1) |    [<img src="https://github.com/mira-space/Mira/assets/13939478/a654666b-1d0e-429b-83ae-97ae9516985a" width="300">](https://github.com/mira-space/Mira/assets/13939478/a654666b-1d0e-429b-83ae-97ae9516985a)|[<img src="https://github.com/mira-space/Mira/assets/13939478/8c314482-81fc-4d95-ab1b-b45a28bd3dee" width="300">](https://github.com/mira-space/Mira/assets/13939478/8c314482-81fc-4d95-ab1b-b45a28bd3dee) | 
| A cute dog sniffing around the sandy coast. | A serene underwater scene featuring a sea turtle swimming through a coral reef. | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle is with its greenish-brown shell. | 


* **Architecture design**. The current ST-DiT-based model architecture lacks sophisticated spatial-temporal interactions
  
* **Reconstruction artifacts**. We are dedicated to further tune the video VAE to mitigate reconstruction artifacts.

* **Sustained object consistency**. Due to resource limitations, our present MiraDiT employs distinct modules for spatial and temporal processing, which may affect the stability of object representation in longer, dynamic video sequences.
  
* At this stage, aspects such as image quality (resolution, clarity) and text alignment have not been our focus, but they remain important considerations for future updates.



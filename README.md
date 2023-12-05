# MotionDirector

This is the official repository of [MotionDirector](https://showlab.github.io/MotionDirector).

**MotionDirector: Motion Customization of Text-to-Video Diffusion Models.**
<br/>
[Rui Zhao](https://ruizhaocv.github.io/),
[Yuchao Gu](https://ycgu.site/), 
[Jay Zhangjie Wu](https://zhangjiewu.github.io/), 
[David Junhao Zhang](https://junhaozhang98.github.io/),
[Jiawei Liu](https://jia-wei-liu.github.io/),
[Weijia Wu](https://weijiawu.github.io/),
[Jussi Keppo](https://www.jussikeppo.com/),
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>

[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://showlab.github.io/MotionDirector)
[![arXiv](https://img.shields.io/badge/arXiv-MotionDirector-b31b1b.svg)](https://arxiv.org/abs/2310.08465)

<p align="center">
<img src="https://github.com/showlab/MotionDirector/blob/page/assets/teaser.gif" width="1080px"/>  
<br>
<em>MotionDirector can customize text-to-video diffusion models to generate videos with desired motions.</em>
</p>

## News
- [2023.12.05] [Colab demo](https://github.com/camenduru/MotionDirector-colab) is available. Thanks to [Camenduru](https://twitter.com/camenduru).
- [2023.12.04] [MotionDirector for Cinematic Shots](#MotionDirector_for_Cinematic_Shots) released. Now, you can make AI films with professional cinematic shots!
- [2023.12.02] Code and model weights released!

## ToDo
- [ ] Gradio Demo
- [ ] More trained weights of MotionDirector

## Setup
### Requirements

```shell
# create virtual environment
conda create -n motiondirector python=3.8
conda activate motiondirector
# install packages
pip install -r requirements.txt
```

### Weights of Foundation Models
```shell
git lfs install
## You can choose the ModelScopeT2V or ZeroScope, etc., as the foundation model.
## ZeroScope
git clone https://huggingface.co/cerspense/zeroscope_v2_576w ./models/zeroscope_v2_576w/
## ModelScopeT2V
git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b ./models/model_scope/
```
### Weights of trained MotionDirector <a name="download_weights"></a>
```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ruizhaocv/MotionDirector_weights ./outputs
```

## Usage
### Training

#### Train MotionDirector on multiple videos:
```bash
python MotionDirector_train.py --config ./configs/config_multi_videos.yaml
```
#### Train MotionDirector on a single video:
```bash
python MotionDirector_train.py --config ./configs/config_single_video.yaml
```

Note:  
- Before running the above command, 
make sure you replace the path to foundational model weights and training data with your own in the config files `config_multi_videos.yaml` or `config_single_video.yaml`.
- Generally, training on multiple 16-frame videos usually takes `300~500` steps, about `9~16` minutes using one A5000 GPU. Training on a single video takes `50~150` steps, about `1.5~4.5` minutes using one A5000 GPU. The required VRAM for training is around `14GB`.
- Reduce `n_sample_frames` if your GPU memory is limited.
- Reduce the learning rate and increase the training steps for better performance.


### Inference
```bash
python MotionDirector_inference.py --model /path/to/the/foundation/model  --prompt "Your prompt" --checkpoint_folder /path/to/the/trained/MotionDirector --checkpoint_index 300 --noise_prior 0.
```
Note: 
- Replace `/path/to/the/foundation/model` with your own path to the foundation model, like ZeroScope.
- The value of `checkpoint_index` means the checkpoint saved at which the training step is selected.
- The value of `noise_prior` indicates how much the inversion noise of the reference video affects the generation. 
We recommend setting it to `0` for MotionDirector trained on multiple videos to achieve the highest diverse generation, while setting it to `0.1~0.5` for MotionDirector trained on a single video for faster convergence and better alignment with the reference video.


## Inference with pre-trained MotionDirector
All available weights are at official [Huggingface Repo](https://huggingface.co/ruizhaocv/MotionDirector_weights).
Run the [download command](#download_weights), the weights will be downloaded to the folder `outputs`, then run the following inference command to generate videos.

### MotionDirector trained on multiple videos:
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A person is riding a bicycle past the Eiffel Tower." --checkpoint_folder ./outputs/train/riding_bicycle/ --checkpoint_index 300 --noise_prior 0. --seed 7192280
```
Note:  
- Replace `/path/to/the/ZeroScope` with your own path to the foundation model, i.e. the ZeroScope.
- Change the `prompt` to generate different videos. 
- The `seed` is set to a random value by default. Set it to a specific value will obtain certain results, as provided in the table below.

Results:

<table class="center">
<tr>
  <td style="text-align:center;"><b>Reference Videos</b></td>
  <td style="text-align:center;" colspan="3"><b>Videos Generated by MotionDirector</b></td>
</tr>
<tr>
  <td><img src=assets/multi_videos_results/reference_videos.gif></td>
  <td><img src=assets/multi_videos_results/A_person_is_riding_a_bicycle_past_the_Eiffel_Tower_7192280.gif></td>
  <td><img src=assets/multi_videos_results/A_panda_is_riding_a_bicycle_in_a_garden_8040063.gif></td>              
  <td><img src=assets/multi_videos_results/An_alien_is_riding_a_bicycle_on_Mars_2390886.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A person is riding a bicycle."</td>
  <td width=25% style="text-align:center;">"A person is riding a bicycle past the Eiffel Tower.” </br> seed: 7192280</td>
  <td width=25% style="text-align:center;">"A panda is riding a bicycle in a garden."  </br> seed: 8040063</td>
  <td width=25% style="text-align:center;">"An alien is riding a bicycle on Mars."  </br> seed: 2390886</td>
</tr>
</table>

### MotionDirector trained on a single video:
16 frames:
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A tank is running on the moon." --checkpoint_folder ./outputs/train/car_16/ --checkpoint_index 150 --noise_prior 0.5 --seed 8551187
```
<table class="center">
<tr>
  <td style="text-align:center;"><b>Reference Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Videos Generated by MotionDirector</b></td>
</tr>
<tr>
  <td><img src=assets/single_video_results/reference_video.gif></td>
  <td><img src=assets/single_video_results/A_tank_is_running_on_the_moon_8551187.gif></td>
  <td><img src=assets/single_video_results/A_lion_is_running_past_the_pyramids_431554.gif></td>              
  <td><img src=assets/single_video_results/A_spaceship_is_flying_past_Mars_8808231.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is running on the road."</td>
  <td width=25% style="text-align:center;">"A tank is running on the moon.” </br> seed: 8551187</td>
  <td width=25% style="text-align:center;">"A lion is running past the pyramids." </br> seed: 431554</td>
  <td width=25% style="text-align:center;">"A spaceship is flying past Mars."  </br> seed: 8808231</td>
</tr>
</table>

24 frames:
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A truck is running past the Arc de Triomphe." --checkpoint_folder ./outputs/train/car_24/ --checkpoint_index 150 --noise_prior 0.5 --width 576 --height 320 --num-frames 24 --seed 34543
```
<table class="center">
<tr>
  <td style="text-align:center;"><b>Reference Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Videos Generated by MotionDirector</b></td>
</tr>
<tr>
  <td><img src=assets/single_video_results/24_frames/reference_video.gif></td>
  <td><img src=assets/single_video_results/24_frames/A_truck_is_running_past_the_Arc_de_Triomphe_34543.gif></td>
  <td><img src=assets/single_video_results/24_frames/An_elephant_is_running_in_a_forest_2171736.gif></td>              
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is running on the road."</td>
  <td width=25% style="text-align:center;">"A truck is running past the Arc de Triomphe.” </br> seed: 34543</td>
  <td width=25% style="text-align:center;">"An elephant is running in a forest." </br> seed: 2171736</td>
 </tr>
<tr>
  <td><img src=assets/single_video_results/24_frames/reference_video.gif></td>
  <td><img src=assets/single_video_results/24_frames/A_person_on_a_camel_is_running_past_the_pyramids_4904126.gif></td>              
  <td><img src=assets/single_video_results/24_frames/A_spacecraft_is_flying_past_the_Milky_Way_galaxy_3235677.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A car is running on the road."</td>
  <td width=25% style="text-align:center;">"A person on a camel is running past the pyramids." </br> seed: 4904126</td>
  <td width=25% style="text-align:center;">"A spacecraft is flying past the Milky Way galaxy."  </br> seed: 3235677</td>
</tr>
</table>


## MotionDirector for Cinematic Shots <a name="MotionDirector_for_Cinematic_Shots"></a>

### 1. Zoom
#### 1.1 Dolly Zoom (Hitchcockian Zoom)
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A firefighter standing in front of a burning forest captured with a dolly zoom." --checkpoint_folder ./outputs/train/dolly_zoom/ --checkpoint_index 150 --noise_prior 0.5 --seed 9365597
```
<table class="center">
<tr>
  <td style="text-align:center;"><b>Reference Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Videos Generated by MotionDirector</b></td>
</tr>
<tr>
  <td><img src=assets/cinematic_shots_results/dolly_zoom_16.gif></td>
  <td><img src=assets/cinematic_shots_results/A_firefighter_standing_in_front_of_a_burning_forest_captured_with_a_dolly_zoom_9365597.gif></td>
  <td><img src=assets/cinematic_shots_results/A_lion_sitting_on_top_of_a_cliff_captured_with_a_dolly_zoom_1675932.gif></td>              
  <td><img src=assets/cinematic_shots_results/A_Roman_soldier_standing_in_front_of_the_Colosseum_captured_with_a_dolly_zoom_2310805.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man standing in room captured with a dolly zoom."</td>
  <td width=25% style="text-align:center;">"A firefighter standing in front of a burning forest captured with a dolly zoom." </br> seed: 9365597 </br> noise_prior: 0.5</td>
  <td width=25% style="text-align:center;">"A lion sitting on top of a cliff captured with a dolly zoom." </br> seed: 1675932 </br> noise_prior: 0.5</td>
  <td width=25% style="text-align:center;">"A Roman soldier standing in front of the Colosseum captured with a dolly zoom."  </br> seed: 2310805 </br> noise_prior: 0.5 </td>
</tr>
<tr>
  <td><img src=assets/cinematic_shots_results/dolly_zoom_16.gif></td>
  <td><img src=assets/cinematic_shots_results/A_firefighter_standing_in_front_of_a_burning_forest_captured_with_a_dolly_zoom_4615820.gif></td>
  <td><img src=assets/cinematic_shots_results/A_lion_sitting_on_top_of_a_cliff_captured_with_a_dolly_zoom_4114896.gif></td>              
  <td><img src=assets/cinematic_shots_results/A_Roman_soldier_standing_in_front_of_the_Colosseum_captured_with_a_dolly_zoom_7492004.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A man standing in room captured with a dolly zoom."</td>
  <td width=25% style="text-align:center;">"A firefighter standing in front of a burning forest captured with a dolly zoom." </br> seed: 4615820 </br> noise_prior: 0.3</td>
  <td width=25% style="text-align:center;">"A lion sitting on top of a cliff captured with a dolly zoom." </br> seed: 4114896 </br> noise_prior: 0.3</td>
  <td width=25% style="text-align:center;">"A Roman soldier standing in front of the Colosseum captured with a dolly zoom."  </br> seed: 7492004</td>
</tr>
</table>

#### 1.2 Zoom In
The reference video is shot with my own water cup. You can also pick up your cup or any other object to practice camera movements and turn it into imaginative videos. Create your AI films with customized camera movements!

```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A firefighter standing in front of a burning forest captured with a zoom in." --checkpoint_folder ./outputs/train/zoom_in/ --checkpoint_index 150 --noise_prior 0.3 --seed 1429227
```
<table class="center">
<tr>
  <td style="text-align:center;"><b>Reference Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Videos Generated by MotionDirector</b></td>
</tr>
<tr>
  <td><img src=assets/cinematic_shots_results/zoom_in_16.gif></td>
  <td><img src=assets/cinematic_shots_results/A_firefighter_standing_in_front_of_a_burning_forest_captured_with_a_zoom_in_1429227.gif></td>
  <td><img src=assets/cinematic_shots_results/A_lion_sitting_on_top_of_a_cliff_captured_with_a_zoom_in_487239.gif></td>              
  <td><img src=assets/cinematic_shots_results/A_Roman_soldier_standing_in_front_of_the_Colosseum_captured_with_a_zoom_in_1393184.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A cup in a lab captured with a zoom in."</td>
  <td width=25% style="text-align:center;">"A firefighter standing in front of a burning forest captured with a zoom in." </br> seed: 1429227</td>
  <td width=25% style="text-align:center;">"A lion sitting on top of a cliff captured with a zoom in." </br> seed: 487239 </td>
  <td width=25% style="text-align:center;">"A Roman soldier standing in front of the Colosseum captured with a zoom in."  </br> seed: 1393184</td>
</tr>
</table>

#### 1.3 Zoom Out
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A firefighter standing in front of a burning forest captured with a zoom out." --checkpoint_folder ./outputs/train/zoom_out/ --checkpoint_index 150 --noise_prior 0.3 --seed 4971910
```
<table class="center">
<tr>
  <td style="text-align:center;"><b>Reference Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Videos Generated by MotionDirector</b></td>
</tr>
<tr>
  <td><img src=assets/cinematic_shots_results/zoom_out_16.gif></td>
  <td><img src=assets/cinematic_shots_results/A_firefighter_standing_in_front_of_a_burning_forest_captured_with_a_zoom_out_4971910.gif></td>
  <td><img src=assets/cinematic_shots_results/A_lion_sitting_on_top_of_a_cliff_captured_with_a_zoom_out_1767994.gif></td>              
  <td><img src=assets/cinematic_shots_results/A_Roman_soldier_standing_in_front_of_the_Colosseum_captured_with_a_zoom_out_8203639.gif></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A cup in a lab captured with a zoom out."</td>
  <td width=25% style="text-align:center;">"A firefighter standing in front of a burning forest captured with a zoom out." </br> seed: 4971910</td>
  <td width=25% style="text-align:center;">"A lion sitting on top of a cliff captured with a zoom out." </br> seed: 1767994 </td>
  <td width=25% style="text-align:center;">"A Roman soldier standing in front of the Colosseum captured with a zoom out."  </br> seed: 8203639</td>
</tr>
</table>

## More results

If you have a more impressive MotionDirector or generated videos, please feel free to open an issue and share them with us. We would greatly appreciate it.
Improvements to the code are also highly welcome.

Please refer to [Project Page](https://showlab.github.io/MotionDirector) for more results.


## Citation


```bibtex

@article{zhao2023motiondirector,
  title={MotionDirector: Motion Customization of Text-to-Video Diffusion Models},
  author={Zhao, Rui and Gu, Yuchao and Wu, Jay Zhangjie and Zhang, David Junhao and Liu, Jiawei and Wu, Weijia and Keppo, Jussi and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2310.08465},
  year={2023}
}

```

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers) and [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning). Thanks for open-sourcing!
- Thanks to [camenduru](https://twitter.com/camenduru) for the [colab demo](https://github.com/camenduru/MotionDirector-colab).
- Thanks to [yhyu13](https://github.com/yhyu13) for the [Huggingface Repo](https://huggingface.co/Yhyu13/MotionDirector_LoRA)
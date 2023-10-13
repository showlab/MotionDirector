# MotionDirector

This is the official repository of [MotionDirector](https://showlab.github.io/MotionDirector).

**[MotionDirector: Motion Customization of Text-to-Video Diffusion Models.](https://showlab.github.io/MotionDirector)**
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

## ToDo
- [ ] Release training and inference code.
- [ ] Release model weights.
- [ ] ...

## Results

### Decouple the appearances and motions!
<p align="center">
<img src="https://github.com/showlab/MotionDirector/blob/page/assets/tasks.gif" width="1080px"/>  
<br>
<em>(Row 1) Take two videos to train the proposed MotionDirector, respectively. 
(Row 2) MotionDirector can generalize the learned motions to diverse appearances. 
(Row 3) MotionDirector can mix the learned motion and appearance from different videos to generate new videos. 
(Row 4) MotionDirector can animate a single image with learned motions.</em>
</p>

### Motion customization on multiple videos
<p align="center">
<img src="https://github.com/showlab/MotionDirector/blob/page/assets/results/results_multi.gif" width="1080px"/>  
<br>
</p>

### Motion customization on a single video
<p align="center">
<img src="https://github.com/showlab/MotionDirector/blob/page/assets/results/results_single.gif" width="1080px"/>  
<br>
</p>

### More results
Please refer to [Project Page](https://showlab.github.io/MotionDirector).

## Citation


```bibtex

@misc{zhao2023motiondirector,
      title={MotionDirector: Motion Customization of Text-to-Video Diffusion Models}, 
      author={Rui Zhao and Yuchao Gu and Jay Zhangjie Wu and David Junhao Zhang and Jiawei Liu and Weijia Wu and Jussi Keppo and Mike Zheng Shou},
      year={2023},
      eprint={2310.08465},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
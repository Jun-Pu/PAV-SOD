# [ASOD60K:Audio-Induced Salient Object Detection in Panoramic Videos](https://arxiv.org/abs/2107.11629) 

Authors: [*Yi Zhang*](https://www.linkedin.com/in/bill-y-zhang/), [*Fang-Yi Chao*](https://scholar.google.com/citations?hl=en&user=C9vR9EwAAAAJ), [*Ge-Peng Ji*](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), [*Deng-Ping Fan*](https://dpfan.net/), [*Lu Zhang*](https://luzhang.perso.insa-rennes.fr/), [*Ling Shao*](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en).

# Introduction

<p align="center">
    <img src="./figures/fig_teaser.jpg"/> <br />
    <em> 
    Figure 1: Annotation examples from the proposed ASOD60K dataset. (a) Illustration of head movement (HM). The subjects wear Head-Mounted Displays (HMDs) and observe 360° scenes by moving their head to control a field-of-view (FoV) in the range of 360°×180°. (b) Each subject (i.e., Subject 1 to Subject N) watches the video without restriction. (c) The HMD-embedded eye tracker records their eye fixations. (d) According to the fixations, we provide coarse-to-fine annotations for each FoV including (e) super/sub-classes, instance-level masks and attributes (e.g., GD-Geometrical Distortion).
    </em>
</p>

Exploring to what humans pay attention in dynamic panoramic scenes is useful for many fundamental applications, including augmented reality (AR) in retail, AR-powered recruitment, and visual language navigation. With this goal in mind, we propose PV-SOD, a new task that aims to segment salient objects from panoramic videos. In contrast to existing fixation-level or object-level saliency detection tasks, we focus on multi-modal salient object detection (SOD), which mimics human attention mechanism by segmenting salient objects with the guidance of audio-visual cues. To support this task, we collect the first large-scale dataset, named ASOD60K, which contains 4K-resolution video frames annotated with a six-level hierarchy, thus distinguishing itself with richness, diversity and quality. Specifically, each sequence is marked with both its super-/sub-class, with objects of each sub-class being further annotated with human eye fixations, bounding boxes, object-/instance-level masks, and associated attributes (e.g., geometrical distortion). These coarse-to-fine annotations enable detailed analysis for PV-SOD modeling, e.g., determining the major challenges for existing SOD models, and predicting scanpaths to study the long-term eye fixation behaviors of humans. We systematically benchmark 11 representative approaches on ASOD60K and derive several interesting findings. We hope this study could serve as a good starting point for advancing SOD research towards panoramic videos. 


:running: :running: :running: ***KEEP UPDATING***.

------

# Related Dataset Works

<p align="center">
    <img src="./figures/fig_related_works.jpg"/> <br />
    <em> 
    Figure 2: Summary of widely used salient object detection (SOD) datasets and the proposed panoramic video SOD (PV-SOD) dataset. #Img: The number of images/frames. #GT: The number of ground-truth masks. Pub. = Publication. Obj.-Level = Object-Level. Ins.-Level = Instance-Level. Fix.GT = Fixation-guided ground truths. † denotes equirectangular (ER) images.
    </em>
</p>

------

# Dataset Annotations and Attributes

<p align="center">
    <img src="./figures/fig_attributes.jpg"/> <br />
    <em> 
    Figure 3: Examples of challenging attributes on equirectangular (ER) images from our ASOD60K, with instance-level GT and fixations as annotation guidance. f(k,l,m) denote random frames of a given video.
    </em>
</p>


<p align="center">
    <img src="./figures/fig_pass_reject.jpg"/> <br />
    <em> 
    Figure 4: More annotations. Passed and rejected examples of annotation quality control.
    </em>
</p>


<p align="left">
    <img src="./figures/fig_attr_statistics.jpg"/> <br />
    <em> 
    Figure 5: Attributes description and stastistics. (a)/(b) represent the correlation and frequency of ASOD60K’s attributes, respectively.
    </em>
</p>



# Dataset Statistics 

<p align="center">
    <img src="./figures/fig_categories.jpg"/> <br />
    <em> 
    Figure 6: Statistics of the proposed ASOD60K. (a) Super-/sub-category information. (b) Instance density of each sub-class. (c) Main components of ASOD60K scenes.
    </em>
</p>


------

# Benchmark

## Overall Quantitative Results

<p align="center">
    <img src="./figures/fig_quantitative_results.jpg"/> <br />
    <em> 
    Figure 7: Performance comparison of 7/3 state-of-the-art conventional I-SOD/V-SOD methods and one PI-SOD method over ASOD60K. ↑/↓ denotes a larger/smaller value is better. Best result of each column is bolded.
    </em>
</p>

## Attributes-Specific Quantitative Results

<p align="center">
    <img src="./figures/fig_attr_quantitative_results.jpg"/> <br />
    <em> 
    Figure 8: Performance comparison of 7/3/1 state-of-the-art I-SOD/V-SOD/PI-SOD methods based on each of the attributes.
    </em>
</p>

## Reference

**No.** | **Year** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-: 
01 | 2015 | IEEE TIP   |  Salient object detection: A benchmark | [Paper](https://arxiv.org/pdf/1501.02741.pdf)/Project
02 | 2018 | IEEE TCSVT |  Review of visual saliency detection with comprehensive information | [Paper](https://arxiv.org/pdf/1803.03391.pdf)/Project
03 | 2018 | ACM TIST   |  A review of co-saliency detection algorithms: Fundamentals, applications, and challenges | [Paper](https://arxiv.org/pdf/1604.07090.pdf)/Project
04 | 2018 | IEEE TSP   |  Advanced deep-learning techniques for salient and category-specific object detection: A survey| [Paper](https://ieeexplore.ieee.org/document/8253582)/Project
05 | 2018 | IJCV       |  Attentive systems: A survey | [Paper](https://link.springer.com/article/10.1007/s11263-017-1042-6)/Project
06 | 2018 | ECCV       |  Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground | [Paper](http://mftp.mmcheng.net/Papers/18ECCV-SOCBenchmark.pdf)/[Project](http://dpfan.net/socbenchmark/)
07 | 2019 | CVM        |  Salient object detection: A survey | [Paper](https://link.springer.com/content/pdf/10.1007/s41095-019-0149-9.pdf)/Project
08 | 2019 | IEEE TNNLS |  Object detection with deep learning: A review | [Paper](https://arxiv.org/pdf/1807.05511.pdf)/Project
09 | 2020 | arXiv      |  Light Field Salient Object Detection: A Review and Benchmark | [Paper](https://arxiv.org/pdf/2010.04968.pdf)/[Project](https://github.com/kerenfu/LFSOD-Survey) 
10 | 2021 | IEEE TPAMI      |  Salient Object Detection in the Deep Learning Era: An In-Depth Survey | [Paper](https://arxiv.org/pdf/1904.09146.pdf)/[Project](https://github.com/wenguanwang/SODsurvey) 

------

# Evaluation Toolbox

All the quantitative results were computed based on one-key Python toolbox: https://github.com/zzhanghub/eval-co-sod .

------

# Downloads

------

# Citation

    @article{zhang2021asod60k,
      title={ASOD60K: Audio-Induced Salient Object Detection in Panoramic Videos},
      author={Zhang, Yi and Chao, Fang-Yi and Ji, Ge-Peng and Fan, Deng-Ping and Zhang, Lu and Shao, Ling},
      journal={arXiv preprint arXiv:2107.11629},
      year={2021}
    }

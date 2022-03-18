# Panoramic Audiovisual Salient Object Detection (PAV-SOD)



# Introduction

<p align="center">
    <img src="./figures/fig_teaser.jpg"/> <br />
    <em> 
    Figure 1: An example of our PAVS10K where coarse-to-fine annotations are provided, based on a guidance of fixations acquired from subjective experiments conducted by multiple (N) subjects wearing Head-Mounted Displays (HMDs) and headphones. Each (e.g., fk, fl and fn, where random integral values {k, l, n} ∈ [1, T ]) of the total equirectangular (ER) video frames T of the sequence “Speaking”(Super-class)-“Brothers”(sub-class) are manually labeled with both object-level and instance-level pixel-wise masks. According to the features of defined salient objects within each of the sequences, multiple attributes, e.g., “multiple objects” (MO), “competing sounds” (CS), “geometrical distortion” (GD), “motion blur” (MB), “occlusions” (OC) and “low resolution” (LR) are further annotated to enable detailed analysis for PAV-SOD modeling.
    </em>
</p>

Object-level audiovisual saliency detection in 360° panoramic real-life dynamic scenes is important for exploring and modeling human perception in immersive environments, also for aiding the development of virtual, augmented and mixed reality applications in the fields of such as education, social network, entertainment and training. To this end, we propose a new task, panoramic audiovisual salient object detection (PAV-SOD), which aims to segment the objects grasping most of the human attention in 360° panoramic videos reflecting real-life daily scenes. To support the task, we collect PAVS10K, the first panoramic video dataset for audiovisual salient object detection, which consists of 67 4K-resolution equirectangular videos with per-video labels including hierarchical scene categories and associated attributes depicting specific challenges for conducting PAV-SOD, and 10,465 uniformly sampled video frames with manually annotated object-level and instance-level pixel-wise masks. The coarse-to-fine annotations enable multi-perspective analysis regarding PAV-SOD modeling. We further systematically benchmark 13 state-of-the-art salient object detection (SOD)/video object segmentation (VOS) methods based on our PAVS10K. Besides, we propose a new baseline model, i.e., CAV-Net, which takes advantage of both visual and audio cues of 360 video frames by using a new conditional variational auto-encoder. As a result, our CAV-Net outperforms all competing models and is able to represent the data bias within PAVS10K via uncertainty estimation. With extensive experimental results, we gain several findings about PAV-SOD challenges and insights towards PAV-SOD model interpretability. We hope that our work could serve as a starting point for advancing SOD towards immersive media.


:running: :running: :running: ***KEEP UPDATING***.

------

# PAVS10K

------

# CAV-Net (Baseline Model)

(release soon)

------

# Downloads

The whole object-/instance-level ground truth with default split can be downloaded from [Baidu Dirve](https://pan.baidu.com/s/1zDXE9iHGyWZFFUDIeaKIdQ)(k3h8) or [Google Drive](https://drive.google.com/file/d/1SjsYz57gArBVr_yzgcRnqYI4MpDiZ_Fh/view?usp=sharing).

The videos with default split can be downloaded from [Google Drive](https://drive.google.com/file/d/1qYnXwKLZUtn4Gb8R9U5P4qsibUCNGoUN/view?usp=sharing) or [OneDrive](https://1drv.ms/u/s!Ais1kZo7RR7Lg1Vt1cA_M05apzL7?e=PzZ4Va). 

The head movement and eye fixation data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1tZDIESRiy3W2g--8lnNWag3KhpEGqTHc?usp=sharing)

To generate video frames, please refer to [video_to_frames.py](https://github.com/PanoAsh/ASOD60K/blob/main/video_to_frames.py).

To get access to raw videos on YouTube, please refer to [video_seq_link](https://github.com/PanoAsh/ASOD60K/blob/main/video_seq_link). 

To check basic information regarding the raw videos, please refer to [video_information.txt](https://github.com/PanoAsh/ASOD60K/blob/main/video_information.txt) (keep updating).

------

# Contact

For any problems, please open an [issue](https://github.com/PanoAsh/ASOD60K/issues/new).

Specifically,

If you have any question on head movement and eye fixation data, please contact fang-yi.chao@tcd.ie


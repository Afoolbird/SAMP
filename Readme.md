# Semantic Ambiguity Modeling and Propagation for Fine-Grained Visual Cross View Geo-Localization

This document contains the method of obtaining the dataset as well as the environment configuration and training methods for the  codes of Semantic Ambiguity Modeling and Propagation for Fine-Grained Visual Cross View Geo-Localization.

# Overview

<img src=".\Fig_overview.png">  



# Abstract

Visual cross view geo-localization aims to estimate the GPS location of a query ground view image by matching it to images from a reference database of geo-tagged aerial images. Recent works introduce the joint retrieval and calibration frameworks, overlooking the semantic ambiguity arising from query and reference images characterized by low overlap rates, dynamic foreground, viewpoint changes and perceptual aliasing encountered in practical scenarios. Training without awareness of these issues may lead to a lack of automatic control over relative task importance, potentially compromising the retrieval task in favor of the offset regression task. Consequently, the model may experience conflicting and dominating gradients during the joint training process. Motivated by this, we argue that a good cross view geo-localization model should consider the semantic ambiguity to better deal with challenging samples and provide more control over the joint optimization results for robust training. To achieve this, we propose to model the semantic ambiguity during the offset regression process by integrating associated uncertainty scores, represented as 2D Gaussian distributions, aimed at mitigating negative transfer effects within the joint tasks.

# Dataset

Please follow the [guideline](./data/DATASET.md) to download and prepare the VIGOR dataset. For the resolution [issue](https://github.com/Jeff-Zilence/VIGOR/issues/2), authors of [SliceMatch](https://github.com/tudelft-iv/SliceMatch) has measured the resolution of all cities and revised the label. If you want to use this label, you may find it at their github [repo](https://github.com/tudelft-iv/SliceMatch). The CVACT dataset can be obtained  by github [repo](https://github.com/Liumouliu/OriCNN). Use the Samples_on_CVACT.py to re-sample the CVACT dataset.

# Requirement

> Python >= 3.5, Opencv, Numpy, Matplotlib
>
> Tensorflow == 1.13.1 

# Training

Download the initialization [weights](https://drive.google.com/file/d/1nAHPTq1lbbrseK4uFVgbvM4iL2BazrZ3/view?usp=sharing) from ImageNet, put it in "./data/".

Use the following script to train to get the baseline of VIGOR：

`python train_fist_E_1.py`

Next, semantic ambiguity was modeled as quantified uncertainty using the following script training:

`python train_second_E_2_1.py`

`python train_second_E_2_2.py`

Finally, the final model is obtained by training with the following script:

`python train_third_E_3.p`

# Reference

> https://github.com/shiyujiao/cross_view_localization_SAFA
>
> https://github.com/Jeff-Zilence/Explain_Metric_Learning
>
> https://github.com/david-husx/crossview_localisation.git
>
> https://github.com/Jeff-Zilence/VIGOR
>
> https://github.com/bityj/CVML


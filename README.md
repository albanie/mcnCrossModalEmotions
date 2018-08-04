## mcnCrossModalEmotions

This module contains code in support of the paper [Emotion Recognition in Speech using Cross-Modal Transfer in the Wild](http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions).

### Installation

The easiest way to use this module is to install it with the `vl_contrib` package manager. `mcnCrossModalEmotions` can be installed with the following commands from the root directory of your MatConvNet installation:

```
vl_contrib('install', 'mcnCrossModalEmotions') ;
vl_contrib('setup', 'mcnCrossModalEmotions') ;
```  

### Dependencies

mcnFerPlus requires the following module (which will be downloaded automatically):

* [mcnDatasets](https://github.com/albanie/mcnDatasets) - dataset helpers for MatConvNet
* [autonn](https://github.com/vlfeat/autonn) - autodiff for MatConvNet

### Overview

The high level idea of this work is to see if there is some common signal between the emotional content of someone's facial expression (or at least, how human annotators would label their expression) and the emotional content of their speech.  

Emotion is a notoriously noisy visual (or audio) signal for machine learning tasks. This is due to a number of reasons, but perhaps the most important one is that there is no "ground truth" (we rarely know the true emotional state of the subject, given only a picture of their face or a segment of their speech).  Despite this, it is possible to get a reasonably high level of agreement among human annotators when labelling facial expressions with emotion and we can use this as a flawed, but potentially still useful proxy for emotional state (predicting these human annotator labels is what we refer to as "emotion recognition").  

In this work we first train a CNN in a fully supervised manner to perform emotion recognition from faces.  We do this by taking a state-of-the-art image classification model (a [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507) network) that has been pretrained on a large face verification task ([VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/vggface2.pdf)) and then fine tune it to predict emotions on the much smaller [FERPlus](https://github.com/Microsoft/FERPlus) dataset. We then use the technique of cross-modal distillation popularised by [1] which aims to "distill" the knowledge of the facial expression model (the "teacher") across modalities to a "student" model that only gets to hear the speech of the speaker, but does not see their face. We apply this distillation process across a large collection of unlabelled videos (the [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset), by making emotion predictions for faces in the video clips with the teacher model and training a student model to match the distribution of teacher predictions.


The bigger aim is to try to see if there is enough common signal between the facial expression and voice to train the student  to match the outputs of the teacher.   If successful, this would allow us to train a model for speech recognition with only access to labelled facial expressions in images. 


**References**:

[1] Gupta, Saurabh, Judy Hoffman, and Jitendra Malik. "Cross modal distillation for supervision transfer." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016 ([link](https://arxiv.org/abs/1507.00448))


### Teacher Training


**Usage**: To use the teacher code, running [exps/bencmark\_ferplus\_models.m](exps/benchmark_ferplus_models.m) will download the pretrained teacher models and data and launch an evaluation function.  New teacher models can be trained with the [exps/ferplus\_baselines.m](exps/ferplus_baselines.m) function.


The following pretrained teacher CNNs are available:


| model | pretraining | training | Fer2013+ Val | Fer2013+ Test |
|-------|-------------|----------|--------------|---------------|
| resnet50-ferplus | [VGGFace2](https://arxiv.org/abs/1710.08092) | [Fer2013+](https://github.com/Microsoft/FERPlus) | 89.0 | 87.6 |
| senet50-ferplus | [VGGFace2](https://arxiv.org/abs/1710.08092) | [Fer2013+](https://github.com/Microsoft/FERPlus) | 89.8 | 88.8 |

More details relating to the models can be found [here](http://www.robots.ox.ac.uk/~albanie/mcn-models.html#cross-modal-emotion).




### Distillation

In the paper, we show that there is sufficient signal to learn emotion-predictive embeddings in the student, but that it is a very noisy task.  We validate that the student has learned something useful by testing it on external datasets for speech emotion recognition and showing that it can do quite a lot better than random, but as one would expect, not as well as a model trained with speech labels (Table 5 in the paper).

By treating the dominant prediction of the teacher as kind of ground-truth one-hot label, we can also assess whether the student is to match some portion of the teacher's predictive signal. It's worth noting that since we are using interview data to perform the distillation (because this is what VoxCeleb consists of), emotions such as neutral and happiness are better represented in the videos.   For each emotion below, we first show the ROC curve for the student on the training set, followed by predictions on the test set of "heard" identities in the second column, then "unheard" identities in the third column. 

The idea here is that by looking at the performance on previously heard/unheard identities we can get some idea as to whether it is trying to solve the task by "cheating".  In this case, cheating would correspond to exploiting some bias in the dataset by memorising the identity of the speaker, rather than listening to the emotional content of their speech.   

We find that the student is able to learn a weak classifier for the dominant emotion predicted by the teacher, hinting that there may be a small redundant signal between the (human annotated) facial emotion of a speaker and their (human annotated) speech emotion.

**Anger**

<img src="emoVoxCeleb/figs/anger-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/anger-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/anger-unheardTest.jpg" width="200" />

**Happiness**

<img src="emoVoxCeleb/figs/happiness-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/happiness-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/happiness-unheardTest.jpg" width="200" />

**Neutral**

<img src="emoVoxCeleb/figs/neutral-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/neutral-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/neutral-unheardTest.jpg" width="200" />

**Surprise**

<img src="emoVoxCeleb/figs/surprise-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/surprise-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/surprise-unheardTest.jpg" width="200" />

**Sadness**

<img src="emoVoxCeleb/figs/sadness-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/sadness-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/sadness-unheardTest.jpg" width="200" />

**Fear**

<img src="emoVoxCeleb/figs/fear-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/fear-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/fear-unheardTest.jpg" width="200" />


Note that since the dataset is highly unbalanced, some of the emotions have very few samples - the remaining emotions (disgust and contempt) are not predicted as dominant by the teacher on any frames of the test set. 

### References

If you find the models or code useful, please consider citing:

```
Albanie, Samuel, and Nagrani, Arsha and Vedaldi, Andrea, and Zisserman, Andrew,
"Emotion Recognition in Speech using Cross-Modal Transfer in the Wild." 
ACM Multimedia, 2018. 
```

References for the related datasets are the FER2013+ dataset:

```
@inproceedings{BarsoumICMI2016,
    title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
    author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
    booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
    year={2016}
}
```

and the VoxCeleb dataset:

```
@article{nagrani2017voxceleb,
  title={Voxceleb: a large-scale speaker identification dataset},
  author={Nagrani, Arsha and Chung, Joon Son and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1706.08612},
  year={2017}
}
```
## mcnCrossModalEmotions

This module contains code in support of the paper [Emotion Recognition in Speech using Cross-Modal Transfer in the Wild](http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions).

**Note**: This repo focuses on the code for distillation and student analysis. For the code used to trainer the facial expression CNN teacher and to download pretrained models, see [mcnFerPlus](https://github.com/albanie/mcnFerPlus) instead.


### Analysis

By treating the dominant prediction of the teacher as kind of ground-truth label, we can see how well the student is able to match it. For each emotion below, we first show the ROC curve for the student on the training set, followed by predictions on the test set of "heard" identities in the second column, then "unheard" identities in the third column.

**Anger**

<img src="emoVoxCeleb/figs/anger-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/anger-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/anger-unheardTest.jpg" width="200" />

**Happiness**

<img src="emoVoxCeleb/figs/happiness-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/happiness-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/happiness-unheardTest.jpg" width="200" />

**Neutral**

<img src="emoVoxCeleb/figs/neutral-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/neutral-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/neutral-unheardTest.jpg" width="200" />


**Sadness**

<img src="emoVoxCeleb/figs/sadness-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/sadness-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/sadness-unheardTest.jpg" width="200" />

**Fear**

<img src="emoVoxCeleb/figs/fear-train.jpg" width="200" /> <img src="emoVoxCeleb/figs/fear-heardTest.jpg" width="200" /> <img src="emoVoxCeleb/figs/fear-unheardTest.jpg" width="200" />

Interestingly, the speech model seems to struggle most with sadness.   Note that since the dataset is highly unbalanced, some of the emotions have very few samples (you can see this in ROC curves that make large steps). The remaining emotions (disgust and contempt) are rarely predicted by the teacher, so the curves don't provide much isnight. 
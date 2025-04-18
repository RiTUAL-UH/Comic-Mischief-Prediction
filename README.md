# Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model
<p align="right"><i>Authors: Elaheh Baharlouei, Mahsa Shafaei, Yigeng Zhang, Hugo Jair Escalante, Thamar Solorio </i></p> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the dataset and implementations of the model proposed in the paper ["Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model"]() on [The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation](https://lrec-coling-2024.org/) at the [LREC-COLING 2024](https://lrec-coling-2024.org/) conference.

# Comic-Mischief-Prediction
Figure 1 shows the overall structure of the proposed HIerarchical Cross Attention with CAPtion (HICCAP) model. 

![HICCAP](https://github.com/user-attachments/assets/c1b725aa-7ca7-4ab7-b579-7d46d550c3ec)

## Repository Structure
```
Comic-Mischief-Prediction
├── Binary
│   └── source
│       ├── config.py
│       ├── nlp_comic_binary.py
│       └── models
│           ├── attention.py
│           └── unified_model_binary.py
├── Data
│   ├── train_features_lrec_camera.json
│   ├── val_features_lrec_camera.json
│   └── test_features_lrec_camera.json
├── Hybrid-Pretraining
│   ├── nlp_comic_contrastive_loss_LREC.py
│   ├── nlp_comic_pretraining_Hybrid_LREC.py
│   └── unified_model_hybrid_LREC.py
├── Multi-Task
│   └── source
│       ├── config.py
│       ├── nlp_comic_multi_task.py
│       └── models
│           ├── attention.py
│           └── multi_task_model.py
└── HICCAP.pdf
```

## Data
In this directory, we provide three JSON files containing Metadata of train/val/test sets. These files also include the name of the videos on YouTube, original subtitles, and their extracted tokens using the BERT model, labels, and some additional information related to each video.
Due to the policy, we are not allowed to release the video data. If you need, please email Elaheh Baharlouei ([elaheh.bahar1@gmail.com](elaheh.bahar1@gmail.com)) and we will provide the following data:

```
1. Video features extracted using I3D model
2. Audio features extracted using VGGish model
```

## Binary
This directory contains the binary implementation of our approach. This directory includes source directory which has 1) the proposed HICCAP model implementation, 2) nlp_comic_binary.py script for training purpose, and 3) config.py contains the hyperparameters and configurations variables.  

## Multi-Task
Similar to the "Binary" directory, This directory contains the multi-task implementation of HICCAP approach. It includes source directory which has 1) the proposed HICCAP model implementation, 2) nlp_comic_multi_task.py script for training purpose, and 3) config.py contains the hyperparameters and configurations variables.  


## Data
This directory contains 1) Metadata of train/val/test sets, 2) VGGish audio feature vetors, and 3)I3D video feature vectors. 


## Citation
```
@inproceedings{chen-etal-2021-images,
    title = "Can images help recognize entities? A study of the role of images for Multimodal {NER}",
    author = "Chen, Shuguang  and
      Aguilar, Gustavo  and
      Neves, Leonardo  and
      Solorio, Thamar",
    booktitle = "Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wnut-1.11",
    pages = "87--96",
    abstract = "Multimodal named entity recognition (MNER) requires to bridge the gap between language understanding and visual context. While many multimodal neural techniques have been proposed to incorporate images into the MNER task, the model{'}s ability to leverage multimodal interactions remains poorly understood. In this work, we conduct in-depth analyses of existing multimodal fusion techniques from different perspectives and describe the scenarios where adding information from the image does not always boost performance. We also study the use of captions as a way to enrich the context for MNER. Experiments on three datasets from popular social platforms expose the bottleneck of existing multimodal models and the situations where using captions is beneficial.",
}
```

## Contact
Feel free to get in touch via email to elaheh.bahar1@gmail.com.

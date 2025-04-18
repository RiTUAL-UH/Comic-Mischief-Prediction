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


## Hybrid-Pretraining
This directory contains the implementation of the hybrid-pretraining approch including 1) nlp_comic_contrastive_loss_LREC.py for pretraining with contarstive learning, 2) nlp_comic_pretraining_Hybrid_LREC.py for loading the checkpoint of the pretrained model during contrastive learning and pretraining with various matching pretraining approch and 3) unified_model_hybrid_LREC.py a sample implementation of HICCAP architecture with required layers for hybrid pretraining approch. 


## Citation
```
@inproceedings{baharlouei-etal-2024-labeling,
    title = "Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model",
    author = "Baharlouei, Elaheh  and
      Shafaei, Mahsa  and
      Zhang, Yigeng  and
      Escalante, Hugo Jair  and
      Solorio, Thamar",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.874/",
    pages = "9999--10013",
    abstract = "We address the challenge of detecting questionable content in online media, specifically the subcategory of comic mischief. This type of content combines elements such as violence, adult content, or sarcasm with humor, making it difficult to detect. Employing a multimodal approach is vital to capture the subtle details inherent in comic mischief content. To tackle this problem, we propose a novel end-to-end multimodal system for the task of comic mischief detection. As part of this contribution, we release a novel dataset for the targeted task consisting of three modalities: video, text (video captions and subtitles), and audio. We also design a HIerarchical Cross-attention model with CAPtions (HICCAP) to capture the intricate relationships among these modalities. The results show that the proposed approach makes a significant improvement over robust baselines and state-of-the-art models for comic mischief detection and its type classification. This emphasizes the potential of our system to empower users, to make informed decisions about the online content they choose to see."
}

## Contact
Feel free to get in touch via email to elaheh.bahar1@gmail.com.

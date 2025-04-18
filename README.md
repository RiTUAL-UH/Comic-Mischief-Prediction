# Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model
<p align="right"><i>Authors: Elaheh Baharlouei, Mahsa Shafaei, Yigeng Zhang, Hugo Jair Escalante, Thamar Solorio </i></p> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the dataset and implementations of the model proposed in the paper ["Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model"]() on [The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation](https://lrec-coling-2024.org/) at the [LREC-COLING 2024](https://lrec-coling-2024.org/) conference.


## Repository Structure
```
multimodal_NER
└── src
    ├── commons
    │   ├── globals.py
    │   └── utils.py
    ├── data # implementation of dataset class
    ├── modeling 
    │   ├── layers.py # implementation of neural layers
    │   ├── model.py # implementation of neural networks
    │   └── train.py # functions to build, train, and predict with a neural network
    ├── experiment.py # entire pipeline of experiments
    └── main.py # entire pipeline of our system

```

## Data
In this repository, we provide three json files containing Metadata of train/val/test sets. These files also include the name of the videos in the Youtube, original subtitles and their extracted tokens using BERT model, labels, and some additional information related to each video. Due to the policy, we are not allowed to release the video data. If you need, please email Elaheh Baharlouei ([elaheh.bahar1@gmail.com](elaheh.bahar1@gmail.com)) and we will provide the following data:

```
1. Video features extracted using I3D model
2. Audio features extracted using VGGish model
```



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



# Comic-Mischief-Prediction
Repository for the work Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model (LREC-COLING 2024). Figure 1 shows the overall structure of the proposed HIerarchical Cross Attention with CAPtion (HICCAP) model. Totally we have 3 directories 1) Binary, 2) Multi-Task, and 3) Features. We will discuss the content of directories in the next sections.


![HICCAP](https://github.com/user-attachments/assets/c1b725aa-7ca7-4ab7-b579-7d46d550c3ec)

## Binary
This directory contains the binary implementation of our approach. Inside of this directory, we can find two directories 1) source, and 2) processed_data. The source directory contains all the Python files needed to run the model for binary prediction. In the Python file "nlp_comic_fine_tuning_binary.py", we have the training loop for binary classification. The cinfig.py contains all the required configurations for training the model. The "models" directory contains the Python file of the proposed model and also the attention file implemented in this project. The "experiments" directory also contains a couple of helper functions that we utilized in our training process. The processed_data directory contains the json files needed for training, validation, and testing the model. These files contain a dictionary that has the name of all videos with assigned binary labels and also the related description. 

## Multi-Task
Similar to the "Binary" directory, this directory contains the multi-task implementation of our approach. Inside of this directory, we can find two directories 1) source, and 2) processed_data. The source directory contains all the Python files needed to run the model for multi-task prediction. In the Python file "nlp_comic_multi_task.py", we have the training loop for multi-task classification. The cinfig.py contains all the required configurations for training the model. The "models" directory contains the Python file of the proposed model and also the attention file implemented in this project. The "experiments" directory also contains a couple of helper functions that we utilized in our training process. The processed_data directory contains the json files needed for training, validation, and testing the model. These files contain a dictionary that has the name of all videos with assigned labels for different categories and also the related description. 

## Features
During the training of the model, we load the i3d and vggish features of input videos and audio respectively. These files have been generated offline and stored in separate directories which will be read during the training and testing process. The "Features" directory has a vgg_vecs.zip file which contains the vggish audio features of input videos. For using these features you should extract this file to the "vgg_vecs" directory and set the address inside of the training loop file. Similarly, the "Features" directory has 8 rar files "i3D_vecs.part01.rar,..." that contain the i3d features of input videos. You should extract all of these rar files in a unique directory named "i3D_vecs" and set its address of it in the training loop file. Note that the process of generating vggish and i3d features is offline and will be done separately. During the training process, we only read those features and used them for the training of the model and prediction.

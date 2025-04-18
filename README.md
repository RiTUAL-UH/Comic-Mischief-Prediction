# Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model
<p align="right"><i>Authors: Elaheh Baharlouei, Mahsa Shafaei, Yigeng Zhang, Hugo Jair Escalante, Thamar Solorio </i></p> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the dataset and implementations of the model proposed in the paper ["Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model"]() on [The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation][(https://lrec-coling-2024.org/)] at the [LREC-COLING 2024] conference.

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

## Installation
We have updated the code to work with Python 3.9, Pytorch 1.9, and CUDA 11.2. If you use conda, you can set up the environment as follows:

```bash
conda create -n multimodal_NER python==3.8
conda activate multimodal_NER
conda install pytorch==1.9 cudatoolkit=11.2 -c pytorch
```

Also, install the dependencies specified in the requirements.txt:
```
pip install -r requirements.txt
```

## Data
In this repository, we provide some toy examples to play with the code. Due to the policy, we are not allowed to release the data. If you need, please email Shuguang Chen ([schen52@uh.edu](schen52@uh.edu)) and we will provide the following data:

```
1. original twitter text
2. associated images
3. associated image captions
```

<u>Image feature extraction</u>:
- Global image features: you can use the script under `src/data/global_image_feature_extraction` for extract image features from VGG16 or ResNet152.
- Regional image features: we provide the script under `src/data/regional_image_feature_extraction`. Please follow the instructions in [VisualBERT](https://github.com/uclanlp/visualbert/tree/master/visualbert) for extracting features for image objects.



## Running

We use config files to specify the details for every experiment (e.g., hyper-parameters, datasets, etc.). You can modify config files in the `configs` directory and run experiments with following command:

```
CUDA_VISIBLE_DEVICES=[gpu_id] python src/main.py --config /path/to/config
```

If you would like to run experiments with VisualBERT, please download the pretrained weights from [VisualBERT](https://github.com/uclanlp/visualbert/tree/master/visualbert) and replace `pretrained_weights` in the config file:

```json
    ...
    "model": {
        "name": "mner",
        "model_name_or_path": "bert-base-uncased",
        "pretrained_weights": "path/to/pretrained_weights",
        "do_lower_case": true,
        "output_attentions": false,
        "output_hidden_states": false
    },
    ...
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
Feel free to get in touch via email to schen52@uh.edu.



# Comic-Mischief-Prediction
Repository for the work Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model (LREC-COLING 2024). Figure 1 shows the overall structure of the proposed HIerarchical Cross Attention with CAPtion (HICCAP) model. Totally we have 3 directories 1) Binary, 2) Multi-Task, and 3) Features. We will discuss the content of directories in the next sections.


![HICCAP](https://github.com/user-attachments/assets/c1b725aa-7ca7-4ab7-b579-7d46d550c3ec)

## Binary
This directory contains the binary implementation of our approach. Inside of this directory, we can find two directories 1) source, and 2) processed_data. The source directory contains all the Python files needed to run the model for binary prediction. In the Python file "nlp_comic_fine_tuning_binary.py", we have the training loop for binary classification. The cinfig.py contains all the required configurations for training the model. The "models" directory contains the Python file of the proposed model and also the attention file implemented in this project. The "experiments" directory also contains a couple of helper functions that we utilized in our training process. The processed_data directory contains the json files needed for training, validation, and testing the model. These files contain a dictionary that has the name of all videos with assigned binary labels and also the related description. 

## Multi-Task
Similar to the "Binary" directory, this directory contains the multi-task implementation of our approach. Inside of this directory, we can find two directories 1) source, and 2) processed_data. The source directory contains all the Python files needed to run the model for multi-task prediction. In the Python file "nlp_comic_multi_task.py", we have the training loop for multi-task classification. The cinfig.py contains all the required configurations for training the model. The "models" directory contains the Python file of the proposed model and also the attention file implemented in this project. The "experiments" directory also contains a couple of helper functions that we utilized in our training process. The processed_data directory contains the json files needed for training, validation, and testing the model. These files contain a dictionary that has the name of all videos with assigned labels for different categories and also the related description. 

## Features
During the training of the model, we load the i3d and vggish features of input videos and audio respectively. These files have been generated offline and stored in separate directories which will be read during the training and testing process. The "Features" directory has a vgg_vecs.zip file which contains the vggish audio features of input videos. For using these features you should extract this file to the "vgg_vecs" directory and set the address inside of the training loop file. Similarly, the "Features" directory has 8 rar files "i3D_vecs.part01.rar,..." that contain the i3d features of input videos. You should extract all of these rar files in a unique directory named "i3D_vecs" and set its address of it in the training loop file. Note that the process of generating vggish and i3d features is offline and will be done separately. During the training process, we only read those features and used them for the training of the model and prediction.

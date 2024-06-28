import os
import sys

import random
import torch
sys.path.append('/home/ebaharlo/comic_mischief/Bert_Based_Model/source')

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

from torch import nn
import time
import json
import numpy as np
#import pandas as pd
from random import shuffle
from torch import optim
from torch.nn import functional as F
from experiments import utils as U
from experiments.TorchHelper import TorchHelper
from experiments.tf_logger import Logger
#from experiments.dataloader import *
from models.unified_model_hybrid_LREC import *
import config as C
from sklearn.metrics import f1_score
import warnings
from sklearn.metrics import confusion_matrix
from pytorch_pretrained_bert import BertAdam
from transformers import AdamW
warnings.filterwarnings('ignore')

torch_helper = TorchHelper()

loss_weights1 = torch.Tensor([1,3])


#run_mode = 'run'
run_mode = 'resume'
# run_mode = 'test_resume'
criterian = nn.CrossEntropyLoss()
# exp_mode =

start_epoch = 0
# plot_interval = 5
# limit_movies_for_rank_plot = 500
# test_data_limit = 100
batch_size = 20 #C.batch_size_kinetics

max_epochs = 40
learning_rate = 1e-5
clip_grad = 0.5
weight_decay_val = 0
optimizer_type = 'adam'  # sgd

# if run_mode == 'test' or run_mode == 'test_resume':
#     max_epochs = 25
#     plot_interval = 1
#     batch_size = 8
# if 'resume' in run_mode:
#     start_epoch = 2

collect_attention = True
run_multitask = False

l2_regularize = True
l2_lambda = 0.1

# Learning rate scheduler
lr_schedule_active = True
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau

# Creates the directory where the results, logs, and models will be dumped.
run_name = 'Text_Audio_Image_binary/'
run_name = 'Pretraining_Hybrid_Skip_sched'+str(learning_rate)
description = ''

output_dir_path = '/path-to-results/'+ run_name+'/'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)
#logger = Logger(output_dir_path + 'logs')

path_res_out = os.path.join(output_dir_path, 'res_'+run_name+'.out')
f = open(path_res_out, "a")
f.write('-------------------------\n')
f.close()

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------


#features_dict_train = json.load(open(C.training_features_bert_kinetics))
#features_dict_val = json.load(open(C.val_features_bert_kinetics))
#features_dict_test = json.load(open(C.test_features_bert_kinetics))

features_dict_train = json.load(open(C.training_features_bert_HowTo100M))
features_dict_val = json.load(open(C.val_features_bert_HowTo100M))
features_dict_test = json.load(open(C.test_features_bert_HowTo100M))

#features_dict_test.update(features_dict_test)
#features_dict_test.update(features_dict_val)
#features_dict_test.update(features_dict_train)


train_set = features_dict_train
print (len(train_set))
print('Train Loaded')

validation_set = features_dict_val
print (len(validation_set))
print('Validation Loaded')

#total_id_list = train_id_list +val_id_list+test_id_list
test_set = features_dict_test
print (len(test_set))
print('test Loaded')

#===========================================================================================================

# ------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def create_model():
    """
    Creates and returns the EmotionFlowModel.
    Moves to GPU if found any.
    :return:

    """

    model =  Bert_Model_Matching()
    if run_mode == 'resume' or run_mode == 'test_resume' or run_mode == 'test':
        model1 =  Bert_Model_CL()
        #model1 = Bert_Model_Pretraining()
        
        checkpoint_path = "path-to-checkpoint/Contrastive_Loss_Kinetics_Skip_Connection1e-05/"#Pretraining_Kinetics
        torch_helper.load_saved_model(model1, checkpoint_path + 'best_contrastive_loss.pth') #last_pretrain_All.pth
        print('model loaded')
    
    
        model.features = model1.features
        
    model.cuda()
    return model


def compute_l2_reg_val(model):
    if not l2_regularize:
        return 0.

    l2_reg = None

    for w in model.parameters():
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2)

    return l2_lambda * l2_reg.item()


from torch.utils.data import TensorDataset, DataLoader
def masking(docs_ints, seq_length=500):

    # getting the correct rows x cols shape
    masks = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        #mask[i, :len(row)] = 1
        masks[i, -len(row):] = 1

    return masks

def mask_vector(max_size,arr):
    # print (arr,arr.shape)
    if arr.shape[0] > max_size:
       output = [1]*max_size
    else:
       len_zero_value = max_size -  arr.shape[0]
       output = [1]*arr.shape[0] + [0]*len_zero_value
    
    return np.array(output)

def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    #print (S, D)
    try:
       pad_l =  max_feature_len - S
       # pad
       pad_segment = np.zeros((pad_l, D))
       feature = np.concatenate((feature,pad_segment), axis=0)
       #print (feature.shape)
    except:
       feature = feature[0:max_feature_len]
       #print (feature.shape)
    return feature

def pad_features(docs_ints, seq_length=500):

    # getting the correct rows x cols shape
    features = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """

    start_time = time.time()

    model.train()

    batch_idx = 1
    total_loss = 0
    batch_x = []
    batch_image, batch_mask_img = [],[]
    batch_audio, batch_mask_audio = [],[]
    batch_emo_deepMoji = []
    batch_mask = []
    batch_y = []
    batch_text = []
    train_imdb = []
    sh_train_set = train_set
    #random.Random(2).shuffle(sh_train_set)

    for index, i in enumerate(sh_train_set):
        # if index == 100:
        #   break
        #list(np.int_(batch_x))
        mid = sh_train_set[i]['IMDBid']
        if mid == "_HvRp2w0SHM_32" or mid == "a6L2XzFljgE_62" or mid == "nLI3n_fzPMw_100":
          print(mid)
          continue

        #path = "/home/ebaharlo/comic_mischief/BMT/submodules/video_features/out_i3D_vecs/"
        path = "path-to-video_features/out_i3D_vecs/"
        #image_vec = np.load("./deepMoji_out/"+mid+".npy")
        try:
           a1 = np.load(path+mid+"_rgb.npy")
        except:
           a1 = np.array([1024*[0.0]])
           
        try:
           a2 = np.load(path+mid+"_flow.npy")
        except:
           a2 = np.array([1024*[0.0]])

        a = a1+a2
        masked_img = mask_vector(10,a)
        a = pad_segment(a, 10, 0)
        image_vec = a
        #masked_img = mask_vector(36,a)

        #path = "/home/ebaharlo/comic_mischief/BMT/submodules/video_features/out_vgg_vecs/"

        path = "path-to-audio_features/out_vgg_vecs/"
        try:
           audio_arr = np.load(path+mid+"_vggish.npy")
        except:
           audio_arr = np.array([128*[0.0]])

        masked_audio = mask_vector(10,audio_arr)
        #print (masked_audio)
        audio_vec = pad_segment(audio_arr, 10, 0)
        batch_audio.append(audio_vec)
        batch_mask_audio.append(masked_audio)

        train_imdb.append(mid)
        batch_x.append(np.array(sh_train_set[i]['indexes']))
        batch_mask_img.append(masked_img)
        batch_image.append(image_vec)
        batch_y.append(sh_train_set[i]['y'])

        if (len(batch_x) == batch_size): #or index == len(sh_train_set) - 1 ) and len(batch_x)>0:

            optimizer.zero_grad()

            mask = masking(batch_x)
            batch_x = pad_features(batch_x)
            batch_x = np.array(batch_x)
            batch_x = torch.tensor(batch_x).cuda()

            batch_image = np.array(batch_image)
            batch_image = torch.tensor(batch_image).cuda()

            batch_mask_img = np.array(batch_mask_img)
            batch_mask_img = torch.tensor(batch_mask_img).cuda()

            batch_audio = np.array(batch_audio)
            batch_audio = torch.tensor(batch_audio).cuda()

            batch_mask_audio = np.array(batch_mask_audio)
            batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

            # print("batch_x :", batch_x.shape)
            # print("batch_image :", batch_image.shape)
            # print("batch_audio :", batch_audio.shape)

            #batch_emo_deepMoji = np.array(batch_emo_deepMoji)
            #batch_emo_deepMoji = torch.tensor(batch_emo_deepMoji).cuda()

            
            out, mid, label_TAM, label_IAM, label_ITM, output_TAM, output_IAM, output_ITM = \
                  model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img, batch_audio.float(),batch_mask_audio)
            
            
            loss_TAM = F.binary_cross_entropy(output_TAM.cpu(), label_TAM.cpu())
            loss_IAM = F.binary_cross_entropy(output_IAM.cpu(), label_IAM.cpu())
            loss_ITM = F.binary_cross_entropy(output_ITM.cpu(), label_ITM.cpu())

            #y_pred1 = out.cpu()
            #loss_comic = F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))
            
            loss2 = compute_l2_reg_val(model) + loss_IAM+loss_TAM+loss_ITM
            #print("loss_ITM:", loss_ITM.item(), "loss_IAM:", loss_IAM.item(), "loss_TAM:", loss_TAM.item())
            # print("torch.Tensor(batch_y):", torch.Tensor(batch_y).shape)
            #loss2 = compute_l2_reg_val(model) + F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))

            total_loss += loss2.item()

            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)


            optimizer.step()

            torch_helper.show_progress(batch_idx, np.ceil(len(sh_train_set) / batch_size), start_time,
                                       round(total_loss / (index + 1), 4))
            batch_idx += 1
            batch_x, batch_y,batch_image,batch_mask_img = [], [], [],[]
            batch_audio, batch_mask_audio = [],[]

    return model




# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, dataset):
    model.eval()

    total_loss = 0
    total_loss1, total_loss2, total_loss3 = 0, 0, 0
    total_loss_ITM, total_loss_IAM, total_loss_TAM = 0, 0, 0

    batch_x, batch_y1,batch_image,batch_mask_img = [], [],[],[]
    batch_director = []
    batch_genre = []
    y1_true, y2_true, y3_true = [], [], []
    imdb_ids = []
    predictions = [[], [], []]
    id_to_vec = {}
    batch_audio, batch_mask_audio = [],[]
    #batch_audio, batch_mask_audio = [],[]
    vecs = []
    batch_text = []
    with torch.no_grad():
        for index,i in enumerate(dataset):
            mid = dataset[i]['IMDBid']
            #if mid == "tt1723121.03":
            #words = dataset[i]["words"]
            #emoji_vec = np.load("./deepMoji_out/"+mid+".npy")
            #mid = dataset[i]['IMDBid']
            imdb_ids.append(mid)
            batch_x.append(np.array(dataset[i]['indexes']))
            #batch_emo.append(np.array(dataset[i]['emo']))
            #batch_emo_deepMoji.append(emoji_vec)
            batch_y1.append(dataset[i]['y'])
            #y1_true.append(C.label_to_idx[dataset[i]['label']])
                
            path = "path-to-video_features/out_i3D_vecs/"
            try:
                a1 = np.load(path+mid+"_rgb.npy")
            except:
               a1 = np.array([1024*[0.0]])
           
            try:
               a2 = np.load(path+mid+"_flow.npy")
            except:
               a2 = np.array([1024*[0.0]])
             
            a = a1+a2
            masked_img = mask_vector(10,a)
            a = pad_segment(a, 10, 0)
            image_vec = a
            batch_image.append(image_vec)
            #masked_img = mask_vector(36,a)
            batch_mask_img .append(masked_img)

            path = "path-to-audio_features/out_vgg_vecs/"
            try:
                audio_arr = np.load(path+mid+"_vggish.npy")
            except:
                audio_arr = np.array([128*[0.0]])
            
            #audio_arr = np.load(path+mid+"_vggish.npy")
            masked_audio = mask_vector(10,audio_arr)
            #print (masked_audio)
            audio_vec = pad_segment(audio_arr, 10, 0)
            batch_audio.append(audio_vec)
            batch_mask_audio.append(masked_audio)

            if (len(batch_x) == batch_size):# or index == len(dataset) - 1) and len(batch_x)>0:

                mask = masking(batch_x)

                #print (mask)
                batch_x = pad_features(batch_x)
                batch_x = np.array(batch_x)
                batch_x = torch.tensor(batch_x).cuda()

                batch_image = np.array(batch_image)
                batch_image = torch.tensor(batch_image).cuda()

                batch_mask_img = np.array(batch_mask_img )
                batch_mask_img = torch.tensor(batch_mask_img ).cuda()

                batch_audio = np.array(batch_audio)
                batch_audio = torch.tensor(batch_audio).cuda()
 
                batch_mask_audio = np.array(batch_mask_audio)
                batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

                out, mid_level_out, label_TAM, label_IAM, label_ITM, output_TAM, output_IAM, output_ITM = \
                  model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img,batch_audio.float(),batch_mask_audio)


                loss_TAM = F.binary_cross_entropy(output_TAM, label_TAM)
                loss_IAM = F.binary_cross_entropy(output_IAM, label_IAM)
                loss_ITM = F.binary_cross_entropy(output_ITM, label_ITM)
                
                total_loss_ITM += loss_ITM.item()
                total_loss_IAM += loss_IAM.item()
                total_loss_TAM += loss_TAM.item()


                total_loss1 += 0#loss2.item()

                batch_x, batch_y1,batch_image,batch_mask_img  = [], [], [],[]
                batch_director = []
                batch_genre = []
                batch_mask = []
                batch_text = []
                batch_similar = []
                batch_description = []
                #imdb_ids = []
                batch_audio, batch_mask_audio = [],[]
                
    
  

    return predictions, \
           total_loss1 / len(dataset), \
           0, \
           total_loss_ITM / len(dataset), \
           total_loss_IAM / len(dataset), \
           total_loss_TAM / len(dataset)
    # attn_weights,\
    # total_loss2/len(dataset), \
    # total_loss3/len(dataset), \
    # total_loss/len(dataset), \

    # [micro_f1_1, micro_f1_2, micro_f1_3]


def training_loop():
    global optimizer_type
    """

    :return:
    """
    model = create_model()
    
    if optimizer_type == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
        
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay_val)
   
    lr_scheduler = reduce_on_plateau_lr_schdlr(optimizer, 'max', min_lr=1e-8, patience=3, factor=0.5)
    
    for epoch in range(start_epoch, max_epochs):
        print('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))

        f = open(path_res_out, "a")
        f.write('[Epoch %d] / %d : %s\n' % (epoch + 1, max_epochs, run_name))

        # train model
        """
        if epoch == 115:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        """
        model = train(model, optimizer)

        val_pred, val_loss1, val_f1, val_loss_ITM, val_loss_IAM, val_loss_TAM = evaluate(model, validation_set)
        #train_pred, train_loss1, train_f1, train_loss_ITM, train_loss_IAM, train_loss_TAM = evaluate(model, train_set)

        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        print('Validation Loss %.5f, Validation F1 %.5f' % (val_loss1, val_f1))
        print('Image-Text-Matching Loss %.5f' % (val_loss_ITM))
        print('Image-Audio-Matching Loss %.5f' % (val_loss_IAM))
        print('Text-Audio-Matching Loss %.5f' % (val_loss_TAM))

        print('Learning Rate', current_lr)

        f.write('Validation Loss%.5f, Validation loss ITM %.5f, Validation loss IAM %.5f,Validation loss TAM %.5f,' % (val_loss1, val_loss_ITM, val_loss_IAM, val_loss_TAM))
        #f.write('Test Loss %.5f, Test F1 %.5f\n' % (test_loss1, test_f1))
        f.write('Learning Rate: %f\n' % (current_lr))
        f.close()
        
        if lr_schedule_active:
            lr_scheduler.step(val_loss_TAM + val_loss_IAM+ val_loss_ITM) #

        is_best = torch_helper.checkpoint_model_pretrain_matching(model, optimizer, output_dir_path, val_loss_TAM + val_loss_IAM+ val_loss_ITM, epoch + 1,
                                                'min')


        print()

        # -------------------------------------------------------------
        # Tensorboard Logging
        # -------------------------------------------------------------
        info = {#'training loss': train_loss1,
                'validation loss': val_loss1,
                # 'train_loss1' : train_loss1,
                # 'val_loss1'   : val_loss1,
                # 'train_loss2': train_loss1,
                # 'val_loss2': val_loss2,
                # 'train_loss3'  : train_loss1,
                # 'val_loss3'  : val_loss3,

                #'train_f1_1': train_f1,
                'val_f1_1': val_f1,
                # 'train_f1_2': train_f1[1],
                # 'val_f1_2'  : val_f1[1],
                # 'train_f1_3': train_f1[2],
                # 'val_f1_3'  : val_f1[2],

                'lr': current_lr
                }
        

def test():
    model = create_model()

    val_pred, val_loss1, val_f1 = evaluate(model, test_set)
    print('Validation Loss %.5f, Validation F1 %.5f' % (val_loss1, val_f1))

if __name__ == '__main__':
    #"""i
    if run_mode != 'test':
        U.copy_files(backup_file_list, output_dir_path)
        with open(output_dir_path + 'description.txt', 'w') as f:
                f.write(description)
                f.close()

        training_loop()
    #"""
    else:
        print("test")
        test()

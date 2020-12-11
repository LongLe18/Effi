# Effi

import os
import sys
sys.path.append("Monk_Object_Detection/4_efficientdet/lib/");
from train_detector import Detector
gtf = Detector();
#directs the model towards file structure
root_dir = "./";
coco_dir = "Drone";
img_dir = "./";
set_dir = "Images";

gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=416, use_gpu=True)
gtf.Model();
gtf.Set_Hyperparams(lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0)
%%time
gtf.Train(num_epochs=100, model_output_dir="trained/");

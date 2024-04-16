from torchvision import transforms
from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger
from natsort import natsorted
import glob
import os,sys

if __name__ == '__main__':
    args = sys.argv
    pred_file=args[1]
    result=args[2]
    print(pred_file,result)

    model_name = "pspnet_vgg16"
    device = 'cuda'
    n_classes=2
    batch_size=8
    image_axis_minimum_size = 200
    pretrained = True
    fixed_feature = False
    epoch="epoch_100"

    print(model_name,n_classes,batch_size,epoch)
    logger = Logger(model_name=model_name, data_name='uroko_w')

    ### Model
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)
    logger.load_model(model, epoch)

    #### Writing the predict result.
    os.makedirs(result,exist_ok=True)
    file_list=glob.glob(pred_file+"/"+"*.jpg")
    files=natsorted(file_list)
    # print(files)
    for name in files:
        print(name)
        f=name.split("/")
        uid,num,key=f[-1].split("-")
        dir_name=key.split(".")[0]
        if int(num) == 1:
            print(result+dir_name)
            os.makedirs(result+dir_name,exist_ok=True)
        predict(model,name, result+"/"+dir_name+"/"+f[-1])
    




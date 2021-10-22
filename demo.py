import os
import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

import datasets.cityscapes_labels as cityscapes_labels

from GPUtil import showUtilization as gpu_usage

import matplotlib.pyplot as plt

from gtts import gTTS
from playsound import playsound


gpu_usage()                             

torch.cuda.empty_cache() 


def navigation(image, obj_name):
    
    # Preprocess the image to reduce the size of it
    picture = Image.open(image)
    picture.save("c" + image,optimize=True,quality=65)
    image = "c" + image


    # Add the arguments
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--demo-image', type=str, default=image, help='path to demo image', required=False)
    parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=False)
    parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
    parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
    args, unknown = parser.parse_known_args()
    assert_and_infer_cfg(args, train_mode=False)
    cudnn.benchmark = False
    
    
    # get net
    args.dataset_cls = cityscapes
    net = network.get_net(args, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Net built.')
    net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
    net.eval()
    print('Net restored.')
    
    # get data
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    img = Image.open(args.demo_image).convert('RGB')
    img_tensor = img_transform(img)
    
    # predict
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).cuda()
        pred = net(img)
        print('Inference done.')
    
    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # make color mask image and save it
    colorized = args.dataset_cls.colorize_mask(pred)
    colorized.save(os.path.join(args.save_dir, 'color_mask.png'))
    
    label_out = np.zeros_like(pred)
    for label_id, train_id in args.dataset_cls.id_to_trainid.items():
        label_out[np.where(pred == train_id)] = label_id
        cv2.imwrite(os.path.join(args.save_dir, 'pred_mask.png'), label_out)
        
    # Get train_id of the object from it's name
    id = name2trainId(obj_name)
    
    # Get indexes of  the object
    obj = []
    for i in range (len(pred)):
        for j in range (len(pred[i])):
            if(pred[i][j] == id):
                obj.append((j, i))

    # Get center point in the object 
    mid_point = int(len(obj) / 2)
    midx, midy = obj[mid_point]
    
    # gray image and threshold
    img = cv2.imread(image, 0)
    ret,thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh)
    plt.show()
    cv2.imwrite("thresh.png", thresh)
    thresh = thresh.astype(np.uint8)
    
    width = len(pred[0])
    height = len(pred)
    
    # indexes of the box in center of the image
    start_mid_x = int((width/2) - (width/8))
    end_mid_x = int((width/2) + (width/8))
    start_mid_y = int((height/2) - (height/8))
    end_mid_y = int((height/2) + (height/8))
    
    output = ""
    
    # Get the place of the person
    person_point = (int(width/2), height)
    
    # Check if person arrives
    if(person_point in obj):
        output = "you arrive"
    else:
        # Get the place of the object
        if midx in range(start_mid_x,  end_mid_x):
            if midy in range(start_mid_y, end_mid_y):
                output = "Go straight"
            elif midy > start_mid_y:
                output = "Go straight"
            else:
                output = "Go straight"
        
        elif midx < start_mid_x:
            if midy in range(start_mid_y, end_mid_y):
                output = "Go straight and turn left"
            elif midy > start_mid_y:
                output = "turn left"
            else:
                output = "Go straight and turn left"
        
        else:
            if midy in range(start_mid_y, end_mid_y):
                output = "Go straight and turn right"
            elif midy > start_mid_y:
                output = "Turn right"
            else:
                output = "Go straight and turn right"

    print('Results saved.')
    
    language = 'en'
  
    # make request to google to get synthesis
    myobj = gTTS(text = output, lang=language, slow=False)
  
    myobj.save("navigation.mp3")
    
    # play the audio file
    playsound("navigation.mp3")
    
    return output

# Get train_id of the object from it's name
def name2trainId(name):
    for train_id, label_name in cityscapes_labels.trainId2name.items():
        if(name == label_name):
            return train_id

# call method navigation and send the name of image and the object
navigation('frame1.jpg', "car")

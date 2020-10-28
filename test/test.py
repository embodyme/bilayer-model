import sys
sys.path.append('../')
import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from infer import InferenceWrapper2
import cv2
import numpy 
import time
from test.image_feed import CVFeed
def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))


args_dict = {
    'project_dir': '../',
    'init_experiment_dir': '../runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}


module = InferenceWrapper2(args_dict)

source_data_dict = {
        'source_imgs': np.asarray(Image.open('images/target.jpg')) # H x W x 3
        }

module.initialization(source_data_dict)

post_processing = lambda x: Image.fromarray(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))  
vs = CVFeed(size=(1280, 720), post_processing=post_processing)

while(True):
    tgt_image, _, _ = vs.read()

    target_data_dict = {
        'target_imgs': np.asarray(tgt_image)[None]} # B x H x W x # 3

    output_data_dict = module(target_data_dict)

    pred_img = to_image(output_data_dict['pred_enh_target_imgs'][0, 0], output_data_dict['pred_target_segs'][0, 0])

    output_img = cv2.cvtColor(numpy.asarray(pred_img),cv2.COLOR_RGB2BGR) 
    cv2.imshow("cv", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
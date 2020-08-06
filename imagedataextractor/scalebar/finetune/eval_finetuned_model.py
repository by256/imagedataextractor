import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.append('../../')
from scalebar.ocr import OCR


parser = argparse.ArgumentParser()
parser.add_argument('--eval-list', 
                    type=str, 
                    default='./tesstrain/data/scalebar/list.eval',
                    help='path to input image directory')
args = parser.parse_args()

with open(args.eval_list, 'r') as f:
    eval_paths = f.readlines()

eval_paths = ['./tesstrain/'+x.strip('\n') for x in eval_paths]

ocr = OCR()

base_model_acc = 0
ft_model_acc = 0

for eval_path in tqdm(eval_paths):
    image_path = eval_path.split('.lstmf')[0]+'.tif'
    gt_path = eval_path.split('.lstmf')[0]+'.gt.txt'

    image = np.array(Image.open(image_path))
    with open(gt_path, 'r') as f:
        gt_text = f.read()

    # evaluate with base model
    bm_pred_text = ocr.perform_ocr(image, lang='eng')
    bm_pred_text_inv = ocr.perform_ocr(255-image, lang='eng')
    base_model_acc += int((bm_pred_text == gt_text) | (bm_pred_text_inv == gt_text))

    # evaluate with finetuned model
    ft_pred_text = ocr.perform_ocr(image, lang='scalebar')
    ft_pred_text_inv = ocr.perform_ocr(255-image, lang='scalebar')
    # print(gt_text, '\t', ft_pred_text, '\t', ft_pred_text == gt_text)
    ft_model_acc += int((ft_pred_text == gt_text) | (ft_pred_text_inv == gt_text))

base_model_acc = base_model_acc / len(eval_paths)
ft_model_acc = ft_model_acc / len(eval_paths)


print('Base model accuracy: {}\nFinetuned model accuracy: {}'.format(base_model_acc, ft_model_acc))
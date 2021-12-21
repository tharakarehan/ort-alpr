
import sys 
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import cv2
import numpy as np
import onnxruntime
import time
from lpd.utils import im2single
from lpd.infer_utils import detect_lp_onnx
from lpd.label import Shape

t0 = time.time()
lp_threshold = 0.5
wpod_net_onnx_path = 'models/lpd.onnx'
sess = onnxruntime.InferenceSession(wpod_net_onnx_path)

t1 = time.time()
img_num = 'Cars391.png'
img_path = '/Users/tharakarehan/Desktop/Object_Tracking/License-Plate-Recognition-Library/License-Plate-Detection-Test/License-Plate-Dataset/images/'+img_num
Ivehicle = cv2.imread(img_path)
blurred_img = cv2.GaussianBlur(Ivehicle, (99, 99), 0)
output_dir = 'Results'
output_lp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'Extracted_LPs')
bname = img_path.split('/')[-1][:-4]
ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
side = int(ratio*288.)
bound_dim = min(side + (side % (2**4)), 608)

bbox,Llp, LlpImgs, t = detect_lp_onnx(sess, im2single(
    Ivehicle),bound_dim, 2**4, 300, lp_threshold)

mask = np.zeros(Ivehicle.shape, dtype=np.uint8)

if len(bbox):
    Ilp = LlpImgs[0]
    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
    pts = bbox[0]
    s = Shape(Llp[0].pts)
    # xmin = int(min(pts[0]))
    # ymin = int(min(pts[1]))
    # xmax = int(max(pts[0]))
    # ymax = int(max(pts[1]))
    pt1 = [int(pts[0][0]),int(pts[1][0])]
    pt2 = [int(pts[0][1]),int(pts[1][1])]
    pt3 = [int(pts[0][2]),int(pts[1][2])]
    pt4 = [int(pts[0][3]),int(pts[1][3])]
    # mask = cv2.rectangle(mask, (xmin,ymin),(xmax,ymax), (255, 255, 255), -1)
    ptslist = np.array([[pt1,pt2,pt3,pt4]],dtype=np.int32)
    mask = cv2.fillPoly(mask,ptslist,(255,255,255))
    out = np.where(mask==0, Ivehicle, blurred_img)
    # cv2.imwrite('%s/%s_lp.png' % (output_dir, bname),out)
    cv2.imwrite('%s/%s_lp.png' % (output_lp_dir, bname),Ilp*255.)
    # writeShapes('%s/%s_lp.txt' % (output_lp_dir, bname), [s])
    t2 = time.time()
    print('Success','time_elapsed',t2-t1)
    print('Success','time_elapsed',t2-t0)
    print('Success','time_elapsed',t1-t0)
else:
    print('No Detections')



import sys 
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


import onnxruntime
from lpd.utils import im2single
from lpd.infer_utils import detect_lp_onnx

class LisencePlateDetector():
    def __init__(self) -> None:
        self.lp_threshold = 0.5
        self.lpd_onnx_path = 'models/lpd.onnx'
        self.sess = onnxruntime.InferenceSession(self.lpd_onnx_path)
    
    def __call__(self, img):
        # ori_im = img.copy()
        ratio = float(max(img.shape[:2]))/min(img.shape[:2])
        side = int(ratio*288.)
        bound_dim = min(side + (side % (2**4)), 608)

        bbox,Llp, LlpImgs, t = detect_lp_onnx(self.sess, im2single(
            img),bound_dim, 2**4, 300, self.lp_threshold)
        
        return bbox,Llp, LlpImgs, t

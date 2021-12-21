import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import cv2
import numpy as np
import math
import time
import tools.ocr_utility as utility
from ocr.postprocess import build_post_process
from ocr.utils.utility import get_image_file_list, check_and_read_gif
import onnxruntime


class TextRecognizer():
    def __init__(self):
        self.rec_image_shape = [int(v) for v in "3, 32, 320".split(",")]
        self.character_type = 'ch'
        self.rec_batch_num = 6
        self.rec_algorithm = 'CRNN'
        self.max_text_length = 25
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": 'ch',
            "character_dict_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ocr/utils/ppocr_keys_v1.txt'),
            "use_space_char": True
        }
      
        self.postprocess_op = build_post_process(postprocess_params)
        self.limited_max_width = 1280
        self.limited_min_width = 16

        self.ort_session = onnxruntime.InferenceSession("./models/ocr_rec.onnx")
       
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((32 * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(ratio_imgH)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                # if self.rec_algorithm != "SRN":
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            starttime = time.time()
         
            input_shape = norm_img_batch.shape
            print(input_shape)

            ort_inputs = {self.ort_session.get_inputs()[0].name:norm_img_batch}
            ort_outs = self.ort_session.run(None, ort_inputs)
            outputs1 = ort_outs[0]
          
            # print(outputs1)
        
        rec_result = self.postprocess_op(outputs1)
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        elapse += time.time() - starttime
        return rec_res, elapse
    
    def convert_to_onnx(self):
        # Input to the model
        X_i = torch.randn(3, 3, 32, 320, requires_grad=True)

        # Export the model
        torch.onnx.export(self.net,             # model being run
                  X_i,                         # model input (or a tuple for multiple inputs)
                  "ppocr_rec_infer_latest.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 
                                           3 : 'width_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size', 
                                            1 : 'width_size'}})


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, predict_time = text_recognizer(img_list)
    except Exception as e:
        print(
            "ERROR!!!! \n"
            "Please read the FAQ：https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' ")
        print(e)
        exit()
    for ino in range(len(img_list)):
        print("Predicts of {}:{}".format(valid_image_file_list[ino], rec_res[
            ino]))
    print("Total predict time for {} images, cost: {:.3f}".format(
        len(img_list), predict_time))


if __name__ == '__main__':
    main(utility.parse_args())
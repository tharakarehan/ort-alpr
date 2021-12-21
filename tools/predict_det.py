import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

# import copy
import cv2
import numpy as np
import time
import onnxruntime
import tools.ocr_utility as utility
from ocr.utils.utility import get_image_file_list, check_and_read_gif
from ocr.data import create_operators, transform
from ocr.postprocess import build_post_process



class TextDetector():
    def __init__(self):
        self.det_algorithm = 'DB'
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': 'max',
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.5
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.6
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"
        

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        self.ort_session = onnxruntime.InferenceSession("./models/ocr_det.onnx")
       

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes


    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        # print('###########################',img.shape)
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0).astype('float32')
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()
        input_shape = img.shape
       
        ort_inputs = {self.ort_session.get_inputs()[0].name:img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        outputs1 = ort_outs[0]
        # print(outputs1)
        # print(outputs['maps'])

        preds = {}
        
        preds['maps'] = outputs1
            
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse

    def convert_to_onnx(self):
        # Input to the model
        X_i = torch.randn(1, 3, 960, 1280, requires_grad=True)

        # Export the model
        torch.onnx.export(self.net,             # model being run
                        X_i,                         # model input (or a tuple for multiple inputs)
                        "ppocr_server.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size', 
                                                2 : 'height_size', 
                                                3 : 'width_size'},  # variable lenght axes
                                        'output' : {0 : 'batch_size', 
                                                    2 : 'height_size', 
                                                    3 : 'width_size'}})
        

if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    # text_detector.convert_to_onnx()
    count = 0
    total_time = 0
    draw_img_save = "./inference_ocr_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        dt_boxes, elapse = text_detector(img)

        if count > 0:
            total_time += elapse
        count += 1
        # print("Predict time of {}: {}".format(image_file, elapse))
        # print(dt_boxes)
        src_im = utility.draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        cv2.imshow('Image',src_im)
        cv2.waitKey(0)
    
        # print("The visualized image saved in {}".format(img_path))
    cv2.destroyAllWindows()
    if count > 1:
        print("Avg Time: {}".format(total_time / (count - 1)))


def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
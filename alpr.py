import os 
import cv2
import time
import numpy as np
import tools.predict_system as predict_sys
import tools.predict_lpd as predict_lpd
from PIL import Image, ImageFont, ImageDraw
import re


class ALPR():
    def __init__(self, out_dir: str) -> None:
        self.text_system = predict_sys.TextSystem()
        self.lp_detector = predict_lpd.LisencePlateDetector()
        self.output_directory = out_dir

    def detect_lp(self,path: str, Bbox : bool, save : bool, show : bool):
        
        img = cv2.imread(path)
        Name = os.path.basename(path).split('.')[0]
        bbox, _, _, _ = self.lp_detector(img)
        if len(bbox):
            pts = bbox[0]
            if Bbox:
                xmin = int(min(pts[0]))
                ymin = int(min(pts[1]))
                xmax = int(max(pts[0]))
                ymax = int(max(pts[1]))
                cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (0, 255, 0), 2)
            else:
                pt1 = [int(pts[0][0]),int(pts[1][0])]
                pt2 = [int(pts[0][1]),int(pts[1][1])]
                pt3 = [int(pts[0][2]),int(pts[1][2])]
                pt4 = [int(pts[0][3]),int(pts[1][3])]
                ptslist = np.array([[pt1,pt2,pt3,pt4]],dtype=np.int32)
                cv2.drawContours(img, ptslist, -1, (0, 255, 0), 2)
            
            if save:
                cv2.imwrite('%s/%s_lpd.png' % (self.output_directory, Name), img)

            if show:
                cv2.imshow(Name,img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print('No License Plates Detected')
    
    def blur_lp(self,path: str, save : bool, show : bool):
        
        img = cv2.imread(path)
        Name = os.path.basename(path).split('.')[0]
        max_len = max(img.shape[0],img.shape[1])
        w_k = int(0.2*max_len)
        if w_k%2 == 0:
            w_k = w_k+1
        blurred_img = cv2.GaussianBlur(img, (w_k,w_k), 0)
        mask = np.zeros(img.shape, dtype=np.uint8)
        bbox, _, _, _ = self.lp_detector(img)
        if len(bbox):
            pts = bbox[0]
            
            pt1 = [int(pts[0][0]),int(pts[1][0])]
            pt2 = [int(pts[0][1]),int(pts[1][1])]
            pt3 = [int(pts[0][2]),int(pts[1][2])]
            pt4 = [int(pts[0][3]),int(pts[1][3])]
            ptslist = np.array([[pt1,pt2,pt3,pt4]],dtype=np.int32)
            mask = cv2.fillPoly(mask,ptslist,(255,255,255))
            img = np.where(mask==0, img, blurred_img)
            
            if save:
                cv2.imwrite('%s/%s_blurred.png' % (self.output_directory, Name), img)

            if show:
                cv2.imshow(Name,img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print('No License Plates Detected')
        
    def recognize_lp(self, path: str, save : bool, show : bool, f_scale : float):

        img = cv2.imread(path)
        Name = os.path.basename(path).split('.')[0]
        bbox, _,  LlpImgs, _ = self.lp_detector(img)
        if len(bbox):
            pts = bbox[0]
            xmin = int(min(pts[0]))
            ymin = int(min(pts[1]))
            xmax = int(max(pts[0]))
            ymax = int(max(pts[1]))
            Width = int(xmax-xmin)
            cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (0, 255, 0), int(2/1.5*f_scale))
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            _, rec_res = self.text_system(Ilp*255.)
            print(rec_res)
            if rec_res is not None:
                text_sum = 0
                for text, score in rec_res[::-1]:
                    text = strip_chinese(text)
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 2)
                    text_w, text_h = text_size
                    text_sum+=text_h
                    img = draw_text(img, text,
                                    pos=(xmin, ymin-int(text_sum)),
                                    font=cv2.FONT_HERSHEY_PLAIN,
                                    font_scale=f_scale,
                                    text_color=(0, 0, 0),
                                    font_thickness=2,
                                    text_color_bg=(0, 255, 0)
                                    )
            
                if save:
                    cv2.imwrite('%s/%s_alpr.png' % (self.output_directory, Name), img)

                if show:
                    cv2.imshow(Name,img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print('No License Plates Detected')



        
        else:
            print('No License Plates Detected')
     
        

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 0, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (int(x + 1.1*text_w), y + 2*text_h), text_color_bg, -1)
    im_p = Image.fromarray(img)
    draw = ImageDraw.Draw(im_p)
    font = ImageFont.truetype("fonts/simfang.ttf",int(32*font_scale/1.5))
    draw.text((x, y ),text,text_color,font=font)
    result_o = np.array(im_p)
    # cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)
    return result_o

def strip_chinese(string):
    en_list = re.findall(u'[^\u4E00-\u9FA5]', string)
    for c in string:
        if c not in en_list:
            string = string.replace(c, '')
    return string
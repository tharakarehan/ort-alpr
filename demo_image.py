from alpr import ALPR

x = ALPR(out_dir='lpd_results')
# x.detect_lp(path ='Test_Images/Cars438.jpeg',Bbox=False,show=True,save=False)
# x.blur_lp(path ='Test_Images/Cars422.png',show=True,save=False)
x.recognize_lp(path ='Test_Images/Cars450.jpeg',show=True,save=False,f_scale=1.5)
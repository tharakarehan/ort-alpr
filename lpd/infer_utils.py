import numpy as np
import cv2
import time

from os.path import splitext

from lpd.label import Label
from lpd.utils import getWH, nms
from lpd.projection_utils import getRectPts, find_T_matrix
import onnxruntime

class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)


def reconstruct(Iorig,I,Y,out_size,threshold=.9):

	net_stride 	= 2**4
	side 		= ((208. + 40.)/2.)/net_stride # 7.75

	Probs = Y[...,0]
	Affines = Y[...,2:]
	rx,ry = Y.shape[:2]
	ywh = Y.shape[1::-1]
	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

	xx,yy = np.where(Probs>threshold)

	WH = getWH(I.shape)
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha

	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))

		labels.append(DLabel(0,pts_prop,prob))

	final_labels = nms(labels,.1)
	TLps = []
	bbox = []
	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i,label in enumerate(final_labels):

			# t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
			pt1 = [int(ptsh[0][0]),int(ptsh[1][0])]
			pt2 = [int(ptsh[0][1]),int(ptsh[1][1])]
			pt3 = [int(ptsh[0][2]),int(ptsh[1][2])]
			pt4 = [int(ptsh[0][3]),int(ptsh[1][3])]
			w1 = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
			w2 = ((pt3[0]-pt4[0])**2 + (pt3[1]-pt4[1])**2)**0.5
			h1 = ((pt1[0]-pt4[0])**2 + (pt1[1]-pt4[1])**2)**0.5
			h2 = ((pt2[0]-pt3[0])**2 + (pt2[1]-pt3[1])**2)**0.5
			W = int(max(w1,w2))
			Hgt = int(max(h1,h2))
			Mwh = max(W,Hgt)
			W = int(W*out_size/Mwh)
			Hgt = int(Hgt*out_size/Mwh)
			t_ptsh 	= getRectPts(int(W/10),int(Hgt/10),int(W/10)+W,int(Hgt/10)+Hgt)
			H 		= find_T_matrix(ptsh,t_ptsh)
			Ilp 	= cv2.warpPerspective(Iorig,H,(2*int(W/10)+W,2*int(Hgt/10)+Hgt),borderValue=.0)
			bbox.append(ptsh)
			TLps.append(Ilp)

	return bbox,final_labels,TLps
	

def detect_lp_onnx(sess,I,max_dim,net_step,out_size,threshold):

	min_dim_img = min(I.shape[:2])
	factor 		= float(max_dim)/min_dim_img

	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)
	Iresized = cv2.resize(I,(w,h))

	T = Iresized.copy()
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	start 	= time.time()
	# Yr 		= model.predict(T)
	# Yr 		= np.squeeze(Yr)
	T = T if isinstance(T, list) else [T]
	feed = dict([(input.name, T[n]) for n, input in enumerate(sess.get_inputs())])
	pred_onnx = sess.run(None, feed)[0]
	Yr  = np.squeeze(pred_onnx)
	elapsed = time.time() - start
	bbox,L,TLps = reconstruct(I,Iresized,Yr,out_size,threshold)

	return bbox,L,TLps,elapsed
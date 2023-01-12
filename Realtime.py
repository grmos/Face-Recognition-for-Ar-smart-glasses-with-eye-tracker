from imutils.video import VideoStream
import argparse
import numpy as np
import time
import cv2
import os
import pickle
import torch
from face_alignment import align
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from helper_functions import load_pretrained_model,to_input
#import tobii_research   ##Library for eyetracking Feature Plan

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-wv", "--boolWriteVideo", type=bool, default=False,
	help="Write video in a *.mp4 file")
ap.add_argument("-v", "--boolVideo", type=bool, default=True,
	help="True-> Input a Video.mp4 ,False-> Input from camera ")
ap.add_argument("-cR", "--cameraResolution", type=str, default="4K",
	help="Choose camera Resolution 4K or FHD")
ap.add_argument("-mp", "--modelpath", type=str, default="pretrained/mydatabaseextra2model.hdf5",
	help="the path of the model ")
ap.add_argument("-ef", "--embeddingsfolder", type=str, default="output/train_Embeddings.pickle",
	help="path to empbeddingsfolder for taking the labels(names)")
ap.add_argument("-vf", "--videofile", type=str, default="videos/1.mp4",
	help="path to videofile ")
args = vars(ap.parse_args())

refPt=[]

## Function for eyetracking simulation 
def onMouse(event,x,y,flags,param):
	global mouseX,mouseY,refPt
	if event == cv2.EVENT_MOUSEMOVE:
		refPt=[]
		mouseX,mouseY = x,y
		refPt.append((x,y))

## Creates a Region (Bounding Box) with Center the output of EyeTracker/onMouse Function 	
def createROI(refPt,window_size,frame):
	roi= frame[int(refPt[0][1]-window_size/2):int(refPt[0][1]+window_size/2), int(refPt[0][0]-window_size/2):int(refPt[0][0]+window_size/2)]
	return roi

## Load Classifier (Neural Network) and labels
model = load_pretrained_model('ir_101')
data = pickle.loads(open(os.path.abspath(args["embeddingsfolder"]), "rb").read())
lab = LabelEncoder()
lab.fit_transform(data["names"])
model_loaded = load_model(args["modelpath"])#mydatabaseextramodel mydatabaseextra2model mydatabaseDA2model mydatabasemodel

knownEmbeddings = []
knownNames = []
facial_names=[0]
cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
print("[INFO] starting video stream...")
if(args["boolVideo"]):
	vs = cv2.VideoCapture(args["videofile"])
else:
	vs = cv2.VideoCapture(0)
	vs.set(3, 1920) 
	vs.set(4, 1080)
FRAMEPROC=1       
window_size=225   ##Size of BB ROI
W_bb,h_bb,distance=0,0,0  ##used for Distance estimation
if(args[ "boolWriteVideo"]):
	img_array = []
box=[]
while (vs.isOpened()):
	r,frame = vs.read()
	if(r!=1):break
	size = (frame.shape[1],frame.shape[0])
	roi=frame.copy()
	cv2.imshow("output", frame)
	if(args[ "boolWriteVideo"]):
		img_array.append(cv2.resize(frame,(990,540)))
	cv2.setMouseCallback('output',onMouse)
	if(len(refPt))>0:
		roi=createROI(refPt,window_size,frame)
		cv2.circle(frame, (refPt[0][0],refPt[0][1]), 100, (255,0,0), 5)
		if(facial_names[-1]):
			cv2.putText(frame, facial_names[-1], (refPt[0][0]-int(w_bb/2)-40, refPt[0][1]-int(h_bb/2)-70), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 4)
			cv2.putText(frame, str(distance)+"m", (refPt[0][0]-int(w_bb/2)-40, refPt[0][1]-int(h_bb/2)-150), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 4)
		cv2.imshow("output", frame)
		if(args[ "boolWriteVideo"]):
			img_array.append(cv2.resize(frame,(990,540)))
	(h, w) = roi.shape[:2]
	if(h==0 or w==0 or FRAMEPROC%10!=0): 
		key = cv2.waitKey(5) & 0xFF
		FRAMEPROC=FRAMEPROC+1
		continue;
	facial_names=[0]
	if(FRAMEPROC%1==0):
		FRAMEPROC=FRAMEPROC+1
		box=[]
		aligned_rgb_img,box = align.get_aligned_face(roi)
		if(aligned_rgb_img==None):
			continue;
		bgr_tensor_input = to_input(aligned_rgb_img)
		feature, _ = model(bgr_tensor_input)
		feature=[feature.detach().numpy().flatten()]
		nninput=np.array(feature)
		predictions=model_loaded.predict(nninput)
		index=predictions.argmax(axis=1)
		nnscore=predictions.max(axis=1)
		p_name = lab.classes_[index[0]]
		score=nnscore[0]
		print(score)
		if(score>=0.94):
			facial_names.append(p_name)
			
		for (top, right, bottom, left) in box:
		    w_bb=right-left
		    h_bb=bottom-top
		if(args["cameraResolution"]=="4K"):
			distance=(845/(w_bb))*0.32 
		else:
			distance=(348/(w_bb))*0.5 ##FHD
		distance = float("{:.2f}".format(distance))

	key = cv2.waitKey(5) & 0xFF
	
	if key == ord("q"):
		break
if(args[ "boolWriteVideo"]):
	out = cv2.VideoWriter('demo2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 45, (990,540))
	for i in range(len(img_array)):
		out.write(img_array[i])
		out.release()
vs.release()
cv2.destroyAllWindows()
	

from sort import *
import torch
import cv2

tracker  = Sort()

model = torch.hub.load('C:/Users/MicroApt/Desktop/radome/YOLOFAMILY/yolov5', 'custom',   
			path='bestperson.onnx', source='local')                         # -------------> object detection model YOLOV5
# frame = "P1033651.mp4"
cap = cv2.VideoCapture("C:/Users/MicroApt/Desktop/SARVESHCOLLEGTOPPR/Doteye/P1033651.mp4")     #--------->>> PASSING THE VIDEO

while (True):

	ret, frame = cap.read()
	preds = model(frame,size=640)
	print("preds",preds)
	preds.save()
	detections = preds.pred[0].numpy()

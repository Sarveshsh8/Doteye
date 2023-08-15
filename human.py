
from sort import *
import torch
import cv2

tracker  = Sort()

model = torch.hub.load('C:/Users/MicroApt/Desktop/radome/YOLOFAMILY/yolov5', 'custom', 
								path='bestperson.onnx', source='local')                   #--------------------------> loading the models

# vid? = cv2.VideoCapture(0)
vid = cv2.VideoCapture("C:/Users/MicroApt/Desktop/SARVESHCOLLEGTOPPR/Doteye/P1033651.mp4") #------->> input video

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
   
size = (frame_width, frame_height)                #--------> frame width and height
result = cv2.VideoWriter('humanvideo.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)                         #---------------------->>>> saving the final results (video)                         

i = 0
while (True):

	ret, frame = vid.read()
	preds = model(frame,size=640)    # ---------->> model 
	detections = preds.pred[0].numpy()
	track = tracker.update(detections)    # --------------->> SORT tracker to track the humans

	for j in range(len(track.tolist())):
		coord = track.tolist()[j]
		x1,y1,x2,y2 = int(coord[0]),int(coord[1]),int(coord[2]),int(coord[3])   #---------> coords
		name_idx = int(coord[4])                                                # ---------> class name
		name = "ID : {}".format(str(name_idx))
		# color = colours[name_idx]
		cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
		cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,0),2)
		result.write(frame)
		# cv2.imshow("image",frame)
		name = "C:/Users/MicroApt/Desktop/SARVESHCOLLEGTOPPR/Doteye/results/"+ str(i) + str(j) + ".jpg"
		cv2.imwrite(name,frame)

	i +=1
	if i==409:
		break

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()


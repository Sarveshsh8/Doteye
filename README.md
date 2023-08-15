# Yolov5 model for person detection using onnx 

steps to setup
1. create a anaconda environment python==3.8
2. Clone the yolov5 repositiory using git command
3. Install the requirements.txt using pip install command
4. Use python train.py --source filepath --weight "small/medium/large" to train the model
5. Use export.py to export the models to onnx format for faster inference on cpu device
6. Use val.py to validate the model performance

Annotation format

1. USE LABEL IMG / LABEL ME TO ANNOTATE IMAGES (COCO,PASCALVOC,YOLO)
2. If the annotation is in coco json, just login to roboflow account and convert the json file to yolovtxt format or any format required.



run python_video.py to check the person detection
run python_video_track.py for person tracking



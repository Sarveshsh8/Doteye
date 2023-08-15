# Yolov5 model for person detection using onnx 

#steps to setup
1. create a anaconda environment python==3.8
2. Clone the yolov5 repositiory using git command
3. Install the requirements.txt using pip install command
4. Use python train.py --source filepath --weight "small/medium/large" to train the model
5. Use export.py to export the models to onnx format for faster inference on cpu device
6. Use val.py to validate the model performance


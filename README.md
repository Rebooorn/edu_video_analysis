# Facial and hand expression detection

In this project, we will design a pipeline to:  
1. detect the face/hand keypoints in each frame of a video.
2. analyse the expression change based on the keypoints.
3. research the correlation of the expression changes and the 

## Usage 
1. install the requirements  
```console
pip install -r requirements
```

2. Extract the holistic keypoints from the video by mediapipe.  
```console
python mp_utils.py -vf ./videos/xx.mp4 -of ./videos_outputs
```
This will output the xxx_body.csv, xxx_face.csv and xxx_hand.csv, recording the time-series keypoints of the 

3. Analyse the head and facial expression in the detected time-series.  
```console
python detect_utils.py 
```
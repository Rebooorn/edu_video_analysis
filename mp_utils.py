# This script is modified from https://github.com/WimPouw/envisionBOX_modulesWP/blob/main/Mediapipe_Optional_Masking/Masking_Mediapiping.ipynb

import mediapipe as mp #mediapipe
import cv2 #opencv
import math #basic operations
import numpy as np #basic operations
import pandas as pd #data wrangling
import csv #csv saving
import os #some basic functions for inspecting folder structure etc.
from skimage.transform import resize

from argparse import ArgumentParser
from detect_utils import get_truncated_face_connections

from pathlib import Path
#list all videos in input_videofolder
from os import listdir
from os.path import isfile, join
from utils import markershands, markersbody, facemarks

#load in mediapipe modules
mp_holistic = mp.solutions.holistic
# Import drawing_utils and drawing_styles.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

##################FUNCTIONS AND OTHER VARIABLES
#landmarks 33x that are used by Mediapipe (Blazepose)

#check if there are numbers in a string
def num_there(s):
    return any(i.isdigit() for i in s)

#take some google classification object and convert it into a string
def makegoginto_str(gogobj):
    gogobj = str(gogobj).strip("[]")
    gogobj = gogobj.split("\n")
    return(gogobj[:-1]) #ignore last element as this has nothing

#make the stringifyd position traces into clean numerical values
def listpostions(newsamplemarks):
    newsamplemarks = makegoginto_str(newsamplemarks)
    tracking_p = []
    for value in newsamplemarks:
        if num_there(value):
            stripped = value.split(':', 1)[1]
            stripped = stripped.strip() #remove spaces in the string if present
            tracking_p.append(stripped) #add to this list  
    return(tracking_p)

def cli_debug(args):
    mypath = "./videos" #this is your folder with (all) your video(s)
    vfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #loop through the filenames and collect them in a list
    #time series output folder
    inputfol = "./videos"
    outputf_mask = "./video_outputs"
    outputf_ts = "./video_outputs/"
    pass

# mypath = "./videos" #this is your folder with (all) your video(s)
# vfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #loop through the filenames and collect them in a list
# #time series output folder
# inputfol = "./videos"
# outputf_mask = "./video_outputs"
# outputf_ts = "./video_outputs/"

# #check videos to be processed
# print("The following folder is set as the output folder where all the pose time series are stored")
# print(os.path.abspath(outputf_ts))
# print("\n The following folder is set as the output folder for saving the masked videos ")
# print(os.path.abspath(outputf_mask))
# print("\n The following video(s) will be processed for masking: ")
# print(vfiles)

#initialize modules and functions

def cli_save_video():
    vfile = r'./videos/T4_CXY_en.mp4'
    capture = cv2.VideoCapture(vfile) #load in the videocapture
    n = 0
    while (True):
        ret, image = capture.read() #read frame
        if ret == True: #if there is a frame
            # image_ = resize(image, output_shape=(180, 270)) * 255
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #make sure the image is in RGB format
            # original_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("resizedimage", image)
            image_ = resize(image, output_shape=(180, 270)) * 255
            cv2.imwrite(f'./video_outputs/frames/{n}.jpg', image_.astype(np.uint8))
            # cv2.imwrite(f'./video_outputs/frames/{n}.jpg', image)
        n += 1
        if n == 1000:
            break
    print('done')
    


def cli_mp_main(args):
    # mypath =  args.mypath    #this is your folder with (all) your video(s)
    vfile = args.vfile
    # vfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #loop through the filenames and collect them in a list
    #time series output folder
    outputf_ts = args.outputf_ts
    
    verbose = args.verbose
    save_video = args.save_video

    print("Note that we have the following number of pose keypoints for markers body")
    print(len(markersbody))

    print("\n Note that we have the following number of pose keypoints for markers hands")
    print(len(markershands))

    print("\n Note that we have the following number of pose keypoints for markers face")
    print(len(facemarks ))

    #set up the column names and objects for the time series data (add time as the first variable)
    markerxyzbody = ['time']
    markerxyzhands = ['time']
    markerxyzface = ['time']

    for mark in markersbody:
        for pos in ['X', 'Y', 'Z', 'visibility']: #for markers of the body you also have a visibility reliability score
            nm = pos + "_" + mark
            markerxyzbody.append(nm)
    for mark in markershands:
        for pos in ['X', 'Y', 'Z']:
            nm = pos + "_" + mark
            markerxyzhands.append(nm)
    for mark in facemarks:
        for pos in ['X', 'Y', 'Z']:
            nm = pos + "_" + mark
            markerxyzface.append(nm)
    masking = False

    #We will now loop over all the videos that are present in the video file
    # for vidf in vfiles:
    print("We will now process video:")
    print(vfile)
    # print("This is video number" + str(vfiles.index(vidf))+ "of" + str(len(vfiles)) + "videos in total")
    #capture the video, and check video settings
    videoname = Path(vfile).name
    videoloc = vfile
    capture = cv2.VideoCapture(videoloc) #load in the videocapture
    frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH) #check frame width
    frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) #check frame height
    samplerate = capture.get(cv2.CAP_PROP_FPS)   #fps = frames per second

    #make an 'empty' video file where we project the pose tracking on
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #for different video formats you could use e.g., *'XVID'
    outputloc = str(Path(outputf_ts) / videoname)
    if save_video:
        out = cv2.VideoWriter(outputloc, fourcc, 
                            fps = samplerate, frameSize = (int(frameWidth), int(frameHeight)))
    # print(f'video info: fps={samplerate}, framesize=[{frameHeight}, {frameWidth}]')
    # break
    # Run MediaPipe frame by frame using Holistic with `enable_segmentation=True` to get pose segmentation.
    time = 0
    tsbody = [markerxyzbody]   #these will be your time series objects, which start with collumn names initialized above
    tshands = [markerxyzhands] #these will be your time series objects, which start with collumn names initialized above
    tsface = [markerxyzface]   #these will be your time series objects, which start with collumn names initialized above
    with mp_holistic.Holistic(
            static_image_mode=False, enable_segmentation=True, refine_face_landmarks=True) as holistic:
        nframe=0
        while (True):
            ret, image = capture.read() #read frame
            if ret == True: #if there is a frame
                image_ = resize(image, output_shape=(180, 270)) * 255
                # cv2.imwrite(str(Path(outputf_mask)/'frames'/'{}.jpg'.format(nframe)), image_.astype(np.uint8))
                nframe+=1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #make sure the image is in RGB format
                results = holistic.process(image) #apply Mediapipe holistic processing
                # Draw pose segmentation
                h, w, c = image.shape
                if  np.all(results.segmentation_mask) != None: #check if there is a pose found
                    if masking == False:
                        original_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if masking == True:
                        original_image = np.concatenate([image, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
                        mask_img = np.zeros_like(image, dtype=np.uint8) #set up basic mask image
                        mask_img[:, :] = (255,255,255) #set up basic mask image
                        segm_2class = 0.2 + 0.8 * results.segmentation_mask #set up a segmentation of the results of mediapipe
                        segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2) #set up a segmentation of the results of mediapipe
                        annotated_image = mask_img * segm_2class * (1 - segm_2class) #take the basic mask image and make a sillhouette mask
                        # append Alpha channel to sillhouetted mask so that we can overlay it to the original image
                        mask = np.concatenate([annotated_image, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
                        # Zero background where we want to overlay
                        original_image[mask==0]=0 #for the original image we are going to set everything at zero for places where the mask has to go
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    #now lets draw on the original_image the left and right hand landmarks, the facemesh and the body poses
                        #left hand
                    if verbose:
                        mp_drawing.draw_landmarks(original_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                            #right hand
                        mp_drawing.draw_landmarks(original_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                            #face
                        mp_drawing.draw_landmarks(
                                original_image,
                                results.face_landmarks,
                                # mp_holistic.FACEMESH_TESSELATION,
                                get_truncated_face_connections(mp_holistic.FACEMESH_TESSELATION),
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                            #body
                        mp_drawing.draw_landmarks(
                                original_image,
                                results.pose_landmarks,
                                mp_holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.
                                get_default_pose_landmarks_style())
                    #######################now save everything to a time series
                        #make a variable list with x, y, z, info where data is appended to
                    samplebody = listpostions(results.pose_landmarks)
                    samplehands = listpostions([results.left_hand_landmarks, results.right_hand_landmarks])
                    sampleface = listpostions(results.face_landmarks)
                    samplebody.insert(0, time)
                    samplehands.insert(0, time)
                    sampleface.insert(0, time)
                    tsbody.append(samplebody)   #append to the timeseries object
                    tshands.append(samplehands) #append to the timeseries object
                    tsface.append(sampleface)   #append to the timeseries object
                #show the video as we process (you can comment this out, if you want to run this process in the background)
                if verbose:
                    cv2.imshow("resizedimage", original_image)
                if save_video:
                    out.write(original_image) #save the frame to the new masked video
                time = time+(1000/samplerate)#update the time variable  for the next frame
                print('frame {} done.'.format(nframe))
            if verbose:
                if cv2.waitKey(1) == 27: #allow the use of ESCAPE to break the loop
                    break
            if ret == False: #if there are no more frames, break the loop
                break
            

    #once done de-initialize all processes
    if save_video:
        out.release()
    capture.release()
    if verbose:
        cv2.destroyAllWindows()
    ####################################################### data to be written row-wise in csv fil
    # opening the csv file in 'w+' mode
    filebody = Path(outputf_ts) / (videoname[:-4]+'_body.csv')
    filebody = open(filebody, 'w+', newline ='')
    #write it
    with filebody:    
        write = csv.writer(filebody)
        write.writerows(tsbody)
    # opening the csv file in 'w+' mode
    filehands = Path(outputf_ts) / (videoname[:-4]+'_hands.csv')
    filehands = open(filehands, 'w+', newline ='')
    #write it
    with filehands:
        write = csv.writer(filehands)
        write.writerows(tshands)
    # opening the csv file in 'w+' mode
    fileface = Path(outputf_ts) / (videoname[:-4]+'_face.csv')
    fileface = open(fileface, 'w+', newline ='')
    #write it
    with fileface:    
        write = csv.writer(fileface)
        write.writerows(tsface)

    print("Done with video")    


if __name__ == '__main__':
    cli_save_video()

    # parser = ArgumentParser()
    # # parser.add_argument('-mp', '--mypath', type=str, default='./videos')
    # parser.add_argument('-vf', '--vfile', type=str, required=True)
    # # parser.add_argument('-if', '--ifolder', type=str, default='./videos')
    # parser.add_argument('-of', '--outputf_ts', type=str, default='./video_outputs')
    # parser.add_argument('--debug', action='store_true', default=False)
    # parser.add_argument('-sv', '--save-video', action='store_true', default=False)
    # parser.add_argument('-v', '--verbose', action='store_true', default=False)
    # args = parser.parse_args()

    # cli_mp_main(args)
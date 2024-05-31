# from plot_utils import pt
import numpy as np
import cv2
from utils import get_truncated_face_connections
import mediapipe as mp  
mp_holistic = mp.solutions.holistic

'''
ref: demo 2 here https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
We will solve the homography for face perspective correction.The left/fight eye inner/outer and the nose are considered to be the anchor point. The face is then considered to be a plane.

useful: https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
'''
def solve_H(src, ref):
    # src: [pt, pt, pt, pt, pt] (left eye=le, right eye=re, nose=no)
    # ref: [pt, pt, pt, pt, pt]
    # print(src)
    # print(ref)
    H, _ = cv2.findHomography(src, ref)
    # print(H)
    return H

def warp_img(img, H, framesize=None):
    if framesize is None:
        framesize = img.shape
    return cv2.warpPerspective(img, H, (framesize[1], framesize[0]))
    
def warp_pt(pt: np.ndarray, H):
    ppt = np.ones([1,3])
    ppt[0,:2] = pt
    ppt_warp = (H @ ppt.T).T
    ppt_warp = ppt_warp[0]
    ppt_warp /= ppt_warp[-1]
    return ppt_warp[:2]


if __name__ == '__main__':
    # cv2.imshow('src', img)
    # cv2.waitKey(0)
    import pandas as pd
    from plot_utils import load_xy_from_df
    framesize=[270, 180]
    # csv_file = r'video_outputs\T4_CXY_en_body.csv'
    csv_file = r'video_outputs\T4_CXY_en_face.csv'
    df = pd.read_csv(csv_file)
    # print(df.columns)
    time = df['time']               # ms
    time = time / 1000              # s
   
    # x_left_eye, y_left_eye = load_xy_from_df('LEFT_EYE', df, framesize)        # left eye
    # left_eye_inner = load_xy_from_df('LEFT_EYE_INNER', df, framesize)[7]
    left_eye_outer = load_xy_from_df('263', df, framesize)
   
    # right_eye_inner = load_xy_from_df('RIGHT_EYE_INNER', df, framesize)
    right_eye_outer = load_xy_from_df('33', df, framesize)

    mouth_left = load_xy_from_df('291', df, framesize)
    mouth_right = load_xy_from_df('61', df, framesize)
    # nose = load_xy_from_df('94', df, framesize)

    ref = np.array([
        [200.0, 40.0],      # left eye outer
        [70.0, 40.0],       # right eye out
        [100.0, 140.0],     # mouth right
        [170.0, 140.0],     # mouth left
        # [135.0, 90.0]
    ], dtype=np.float32)

    for i in range(0, 100):
        if left_eye_outer[i] is not np.nan:
            src = np.array([
                left_eye_outer[i],
                right_eye_outer[i],
                mouth_right[i],
                mouth_left[i],
                # nose[i]
            ], dtype=np.float32)

            H = solve_H(src, ref)
        src_p = np.concatenate([src, np.ones([4,1])], axis=1)
        ref_p = (H @ src_p.T).T
        p = np.stack([ref_p[:,-1],]*3).T
        print(p)
        print(ref_p / p)

        print('use warp_pt():')
        # for i in src:
        #     print(warp_pt(i, H))

        img = cv2.imread(r'D:\Chang\education\video_outputs\frames\{}.jpg'.format(i))
        img_warp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        cv2.line(img, [int(i) for i in src[0]], [int(i) for i in src[1]], (0,0,255), 2)
        cv2.line(img, [int(i) for i in src[1]], [int(i) for i in src[2]], (0,0,255), 2)
        cv2.line(img, [int(i) for i in src[2]], [int(i) for i in src[3]], (0,0,255), 2)
        cv2.line(img, [int(i) for i in src[3]], [int(i) for i in src[0]], (0,0,255), 2)
        # cv2.line(img, [int(i) for i in src[4]], [int(i) for i in src[0]], (0,0,255), 2)
        cv2.imshow('src', img)

        cv2.line(img_warp, [int(i) for i in ref[0]], [int(i) for i in ref[1]], (0,0,255), 2)
        cv2.line(img_warp, [int(i) for i in ref[1]], [int(i) for i in ref[2]], (0,0,255), 2)
        cv2.line(img_warp, [int(i) for i in ref[2]], [int(i) for i in ref[3]], (0,0,255), 2)
        cv2.line(img_warp, [int(i) for i in ref[3]], [int(i) for i in ref[0]], (0,0,255), 2)
        # cv2.line(img_warp, [int(i) for i in ref[4]], [int(i) for i in ref[0]], (0,0,255), 2)

        cv2.imshow('warpped', img_warp)
        cv2.waitKey(0)
import numpy as np
import mediapipe as mp  
from utils import pt, load_xy_from_df, load_face_mesh
from utils import lm_to_remove, get_truncated_face_connections
from homography_utils import warp_pt, solve_H
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

def measure_eye_area(pts: list):
    l = len(pts)
    A = 0
    for i in range(l-1):
        A += pts[i][0]*pts[i+1][1] - pts[i+1][0]*pts[i][1]
    A += pts[l][0] * pts[0][1] - pts[0][0] * pts[l][1]
    return np.abs(A)

def distance_xy(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def measure_point_speed(pts: list):
    # pts: time sequence of the same pts
    l = len(pts)
    sp = np.zeros(l)
    for i in range(l-1):
        sp[i] = distance_xy(pts[i], pts[i+1])
    sp[-1] = sp[-2]
    return sp

# detection func of face/head/hand/finger expression


def l2_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def measure_face_expression(pts, ref_pts):
    # the face expression is denoted by the average L2 distance of each points
    d = [l2_distance(i,j) for i, j in zip(pts, ref_pts)]
    return sum(d) / len(d)

def measure_head_expression(pts, ref_pts):
    # the head expression is denoted by the 
    d = [l2_distance(i,j) for i, j in zip(pts, ref_pts)]
    return sum(d) / len(d)

def cli_detect_main(face_csv):
    framesize=[270, 180]
    df = pd.read_csv(face_csv)
    time = df['time']               # ms
    time = time / 1000              # s
    face_mesh = load_face_mesh(df, framesize)

    left_eye_outer = load_xy_from_df('263', df, framesize)
   
    # right_eye_inner = load_xy_from_df('RIGHT_EYE_INNER', df, framesize)
    right_eye_outer = load_xy_from_df('33', df, framesize)

    mouth_left = load_xy_from_df('291', df, framesize)
    mouth_right = load_xy_from_df('61', df, framesize)

    ref = np.array([
        [200.0, 40.0],      # left eye outer
        [70.0, 40.0],       # right eye out
        [100.0, 140.0],     # mouth right
        [170.0, 140.0],     # mouth left
        # [135.0, 90.0]
    ], dtype=np.float32)

    face_exp = [0,]
    head_exp = [0,]
    face_mesh_src_last = None
    face_mesh_warp_last = None
    # for i in tqdm(range(1000)):
    for i in range(len(face_mesh)):
        src = np.array([
                left_eye_outer[i],
                right_eye_outer[i],
                mouth_right[i],
                mouth_left[i],
                # nose[i]
            ], dtype=np.float32)
        H = solve_H(src, ref)
        face_mesh_src = [p[i] for p in face_mesh]
        face_mesh_warp = [warp_pt(p[i], H) for p in face_mesh]
        if face_mesh_src_last is not None:
            face_exp_ = measure_face_expression(face_mesh_warp, face_mesh_warp_last)
            head_exp_ = measure_head_expression(face_mesh_src, face_mesh_src_last)
            face_exp.append(face_exp_)
            head_exp.append(head_exp_)
        face_mesh_src_last = face_mesh_src
        face_mesh_warp_last = face_mesh_warp

    # print(face_exp)
    fig, axs = plt.subplots(1,2, figsize=[16,4])
    axs[0].plot(face_exp)
    axs[1].plot(head_exp)
    axs[0].set_title('{} face expression'.format(face_csv.name.replace('_face.csv', '')))
    axs[1].set_title('head expression')
    # plt.show()
    plt.tight_layout()
    plt.savefig('./video_outputs/plots/{}'.format(face_csv.name.replace('_face.csv', '.png')))
    vname = face_csv.name.replace('_face.csv', '')
    mean_face_exp = np.nanmean(face_exp)
    mean_head_exp = np.nanmean(head_exp)
    print('{},{},{}'.format(vname, mean_face_exp, mean_head_exp))
    plt.close()

if __name__ == '__main__':
    # get_truncated_face_connections(mp_holistic.FACEMESH_TESSELATION)
    # face_csv = r'video_outputs\T4_CXY_en_face.csv'
    # cli_detect_main(face_csv)

    from pathlib import Path
    face_csvs = Path(r'./video_outputs').glob('*_face.csv')
    face_csvs = list(face_csvs)
    for f in face_csvs:
        cli_detect_main(f)

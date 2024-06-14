import numpy as np
import mediapipe as mp  
from utils import pt, load_xy_from_df, load_face_mesh
from utils import lm_to_remove, get_truncated_face_connections
from homography_utils import warp_pt, solve_H
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


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

    face_exp = []
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

def load_face_keypoints(face_csv):
    framesize=[270, 180]
    df = pd.read_csv(face_csv)
    time = df['time']               # ms
    time = time / 1000              # s
    face_mesh = load_face_mesh(df, framesize)

    left_eye_outer = load_xy_from_df('263', df, framesize)
    right_eye_outer = load_xy_from_df('33', df, framesize)

    mouth_left = load_xy_from_df('291', df, framesize)
    mouth_right = load_xy_from_df('61', df, framesize)
    return time, face_mesh, left_eye_outer, right_eye_outer, mouth_left, mouth_right


def cli_face_detect(args):
    face_csv = args.time_series_csv
    face_csv = Path(face_csv)
    fpsm = args.fps_multiplier          # base fps is 30

    clip = [args.clip_time_start, args.clip_time_end]

    time, face_mesh, left_eye_outer, right_eye_outer, mouth_left, mouth_right = load_face_keypoints(face_csv)

    ref = np.array([
        [200.0, 40.0],      # left eye outer
        [70.0, 40.0],       # right eye out
        [100.0, 140.0],     # mouth right
        [170.0, 140.0],     # mouth left
        # [135.0, 90.0]
    ], dtype=np.float32)

    face_exp = []
    face_mesh_warp_last = None
    # for i in tqdm(range(1000)):
    t = time.to_list()
    # print(min(t), max(t))
    # t = t[::fpsm]

    # clip out the region of interest
    # print(t)
    clip_ = np.logical_and([i > clip[0] for i in t], [i < clip[1] for i in t])
    # print(clip_)
    # print(len(face_mesh[0]))
    face_mesh = [p.clip(clip_) for p in face_mesh]
    # print(len(face_mesh[0]))
    t = np.array(t)[clip_]
    # print(min(t), max(t))
    left_eye_outer = left_eye_outer.clip(clip_)
    right_eye_outer = right_eye_outer.clip(clip_)
    mouth_left = mouth_left.clip(clip_)
    mouth_right = mouth_right.clip(clip_)
    tt = []
    for i in range(0, len(face_mesh[0]), fpsm):
        src = np.array([
                left_eye_outer[i],

                right_eye_outer[i],
                mouth_right[i],
                mouth_left[i],
                # nose[i]
            ], dtype=np.float32)
        # print(src)
        H = solve_H(src, ref)
        face_mesh_warp = [warp_pt(p[i], H) for p in face_mesh]
        # print(face_mesh_warp)
        if face_mesh_warp_last is not None:
            face_exp_ = measure_face_expression(face_mesh_warp, face_mesh_warp_last)
            face_exp.append(face_exp_)
            tt.append(t[i])
        face_mesh_warp_last = face_mesh_warp
        # print(i, face_exp, tt)

    # print(face_exp)
    # fig, axs = plt.subplots(1,2, figsize=[16,4])
    # print(len(tt))
    plt.plot(tt, face_exp)
    # axs[1].plot(t, head_exp)
    plt.title('{} face expression'.format(face_csv.name.replace('_face.csv', '')))
    plt.tight_layout()
    plt.savefig('./video_outputs/plots/{}'.format(face_csv.name.replace('_face.csv', '_face.png')))
    # plt.show()
    vname = face_csv.name.replace('_face.csv', '')
    mean_face_exp = np.nanmean(face_exp)
    # print('{},{}'.format(vname, mean_face_exp))
    plt.close()

    return vname, mean_face_exp, tt, face_exp


def cli_head_detect(args):
    face_csv = args.time_series_csv
    face_csv = Path(face_csv)
    fpsm = args.fps_multiplier          # base fps is 30

    clip = [args.clip_time_start, args.clip_time_end]

    time, face_mesh, left_eye_outer, right_eye_outer, mouth_left, mouth_right = load_face_keypoints(face_csv)

    head_exp = []
    t = time.to_list()
    # t = time[::fpsm]
    clip_ = np.logical_and([i > clip[0] for i in t], [i < clip[1] for i in t])
    # clip out the region of interest
    face_mesh = [p.clip(clip_) for p in face_mesh]
    t = np.array(t)[clip_]
    # t = t[t > clip[0] and t < clip[1]]
    tt = []
    face_mesh_src_last = None
    for i in range(0, len(face_mesh[0]), fpsm):

        face_mesh_src = [p[i] for p in face_mesh]
        if face_mesh_src_last is not None:
            head_exp_ = measure_head_expression(face_mesh_src, face_mesh_src_last)
            head_exp.append(head_exp_)
            tt.append(t[i])
        face_mesh_src_last = face_mesh_src

    # print(face_exp)
    # fig, axs = plt.subplots(1,2, figsize=[16,4])
    # axs[0].plot(t, face_exp)
    plt.plot(tt, head_exp)
    plt.title('{} head expression'.format(face_csv.name.replace('_face.csv', '')))
    # plt.show()
    plt.tight_layout()
    plt.savefig('./video_outputs/plots/{}'.format(face_csv.name.replace('_face.csv', '_head.png')))
    # plt.show()
    vname = face_csv.name.replace('_face.csv', '')
    mean_head_exp = np.nanmean(head_exp)
    # print('{},{}'.format(vname, mean_head_exp))
    plt.close()
    return vname, mean_head_exp, tt, head_exp

def cli_finger_detect(args):
    pass

def cli_hand_detect(args):
    pass

def cli_main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-tsc', '--time-series-csv', type=str, required=True)
    parser.add_argument('-fm', '--fps_multiplier', type=int, default=1)
    parser.add_argument('-cts', '--clip-time-start', type=float, default=0)         # in second
    parser.add_argument('-cte', '--clip-time-end', type=float, default=10000)
    parser.add_argument('--face', action='store_true', default=False)
    parser.add_argument('--head', action='store_true', default=False)
    parser.add_argument('--finger', action='store_true', default=False)
    parser.add_argument('--hand', action='store_true', default=False)
    args = parser.parse_args()

    if args.face:
        vname, metric = cli_face_detect(args)
        return vname, metric
    if args.head:
        vname, metric = cli_head_detect(args)
        return vname, metric
    if args.finger:
        cli_finger_detect(args)
    if args.hand:
        cli_hand_detect(args)
    else:
        ValueError('Unrecognized mode, ethier --face, --head, --finger or --hand.')


if __name__ == '__main__':
    # cli_main()

    # face_csv = r'video_outputs\T4_CXY_en_face.csv'
    # cli_detect_main(face_csv)

    from argparse import Namespace
    from pathlib import Path
    import json
    from utils import convert_time_to_seconds

    face_csvs = Path(r'./video_outputs').glob('*_face.csv')
    face_csvs = list(face_csvs)

    # load video clip info
    with open('video_clip.json', 'r') as f:
        video_clip_dict = json.load(f)

    for f in face_csvs:
        vname = str(f).split('_face')[0]
        args = Namespace()
        args.time_series_csv = f
        args.fps_multiplier = 15
        if vname in video_clip_dict.keys():
            args.clip_time_start = convert_time_to_seconds(video_clip_dict[vname].split('/')[0])
            args.clip_time_end = convert_time_to_seconds(video_clip_dict[vname].split('/')[1])
        else:
            args.clip_time_start = 0
            args.clip_time_end = 10000
        vname, face_exp, _, _= cli_face_detect(args)
        vname, head_exp, _, _ = cli_head_detect(args)
        print(f'{vname},{face_exp},{head_exp}')
   
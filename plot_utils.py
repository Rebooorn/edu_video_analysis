import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from skimage.transform import resize
from homography_utils import solve_H, warp_img, warp_pt
from detect_utils import get_truncated_face_connections
import mediapipe as mp #mediapipe
from utils import pt, load_xy_from_df, load_face_mesh, load_eye_or_mouth_mesh


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles


# useful wrappers 
def plot_line(plot_obj, pt1, pt2):
    plot_obj.set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])

def plot_circle(plot_objs, pts):
    assert len(plot_objs) == len(pts)
    for i in range(len(pts)-1):
        plot_line(plot_objs[i], pts[i], pts[i+1])
    plot_line(plot_objs[-1], pts[0], pts[-1])

def get_init_face_mesh(ax):
    face_mesh_plot = []
    for i in range(1936):
        face_mesh_plot.append(ax.plot([], [], 'r-')[0])
    return face_mesh_plot

def update_face_mesh(face_plot, pts, connections):
    for i in range(len(face_plot)):
        plot_line(face_plot[i], pts[connections[i][0]], pts[connections[i][1]])

left_eye_ind = [33, 246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
right_eye_ind = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]
mouth_ind = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
###########################

def plot_face(csv_file, face_csv, fps=30, framesize=[270, 180]):

    df = pd.read_csv(csv_file)
    df_face = pd.read_csv(face_csv)
    print(df.columns)
    time = df['time']               # ms
    time = time / 1000              # s
    nose = load_xy_from_df('NOSE', df, framesize)
   
    # x_left_eye, y_left_eye = load_xy_from_df('LEFT_EYE', df, framesize)        # left eye
    # left_eye = load_xy_from_df('LEFT_EYE', df, framesize)        # left eye
    # left_eye_inner = load_xy_from_df('LEFT_EYE_INNER', df, framesize)
    # left_eye_outer = load_xy_from_df('LEFT_EYE_OUTER', df, framesize)
   
    # right_eye = load_xy_from_df('RIGHT_EYE', df, framesize)        # right eye
    # right_eye_inner = load_xy_from_df('RIGHT_EYE_INNER', df, framesize)
    # right_eye_outer = load_xy_from_df('RIGHT_EYE_OUTER', df, framesize)
    
    # left_mouth = load_xy_from_df('MOUTH_LEFT', df, framesize)  # mouth
    # right_mouth = load_xy_from_df('MOUTH_RIGHT', df, framesize)
    left_eye_outer = load_xy_from_df('263', df_face, framesize)
   
    # right_eye_inner = load_xy_from_df('RIGHT_EYE_INNER', df, framesize)
    right_eye_outer = load_xy_from_df('33', df_face, framesize)

    mouth_left = load_xy_from_df('291', df_face, framesize)
    mouth_right = load_xy_from_df('61', df_face, framesize)

    left_eye_mesh = load_eye_or_mouth_mesh(df_face, framesize, left_eye_ind)
    right_eye_mesh = load_eye_or_mouth_mesh(df_face, framesize, right_eye_ind)
    mouth_mesh = load_eye_or_mouth_mesh(df_face, framesize, mouth_ind)

    face_mesh = load_face_mesh(df_face, framesize)

    ref = np.array([
        [200.0, 40.0],      # left eye outer
        [70.0, 40.0],       # right eye out
        [100.0, 140.0],     # mouth right
        [170.0, 140.0],     # mouth left
        # [135.0, 90.0]
    ], dtype=np.float32)

    # plt.plot(time, x_nose)
    # plt.plot(time, y_nose, 'red')
    # plt.show()

    print('loading frames')
    images = [Image.open('./video_outputs/frames/{}.jpg'.format(i)) for i in range(5000)]
    images = [np.array(i) for i in images]
    # images = [resize(i, framesize[::-1]) for i in images]

    # for n,i in enumerate(images):
    #     ii = Image.fromarray((i*255).astype(np.uint8))
    #     ii.save('./video_outputs/frames/{}.jpg'.format(n))

    plt.ion()

    fig, ax = plt.subplots(1,3, figsize=(15, 4), )
    for i in range(3):
        ax[i].set_xlim(0,framesize[0])
        ax[i].set_ylim(0,framesize[1])
        ax[i].invert_yaxis()
        ax[i].invert_xaxis()
    # eye1, = ax[1].plot([], [], 'ro-')  
    # eye2, = ax[1].plot([], [], 'ro-')
    # eye3, = ax[1].plot([], [], 'ro-')
    # eye4, = ax[1].plot([], [], 'ro-')
    # mouth, = ax[1].plot([], [], 'bo-')
    # nose_, = ax[1].plot([],[],'bo-')
    bg = ax[0].imshow(images[0])

    # eyes/mouth mesh: ax[1]
    left_eye_lines = []
    for i in range(len(left_eye_mesh)):
        left_eye_lines.append(ax[1].plot([], [], 'r-')[0])
    right_eye_lines = []
    for i in range(len(right_eye_mesh)):
        right_eye_lines.append(ax[1].plot([], [], 'r-')[0])
    mouth_lines = []
    for i in range(len(mouth_mesh)):
        mouth_lines.append(ax[1].plot([], [], 'r-')[0])

    # homography: ax[2]
    bg2 = ax[2].imshow(images[0])
    left_eye_lines_warp = []
    for i in range(len(left_eye_mesh)):
        left_eye_lines_warp.append(ax[2].plot([], [], 'r-')[0])
    right_eye_lines_warp = []
    for i in range(len(right_eye_mesh)):
        right_eye_lines_warp.append(ax[2].plot([], [], 'r-')[0])
    mouth_lines_warp = []
    for i in range(len(mouth_mesh)):
        mouth_lines_warp.append(ax[2].plot([], [], 'r-')[0])
    face_plot = get_init_face_mesh(ax[2])
    face_mesh_connections = get_truncated_face_connections(mp_holistic.FACEMESH_TESSELATION)

    for i in range(100):
        
        bg.set_data(images[i])

        # plot_line(nose_, nose[i], nose[i])

        # plot_line(eye1, left_eye_inner[i], left_eye[i])
        # plot_line(eye2, left_eye_outer[i], left_eye[i])

        # plot_line(eye3, right_eye_inner[i], right_eye[i])
        # plot_line(eye4, right_eye_outer[i], right_eye[i])

        # plot_line(mouth, left_mouth[i], right_mouth[i])
        
        # map the face
        plot_circle(left_eye_lines, pts=[p[i] for p in left_eye_mesh])
        plot_circle(right_eye_lines, pts=[p[i] for p in right_eye_mesh])
        plot_circle(mouth_lines, pts=[p[i] for p in mouth_mesh])

        # get homography H
        src = np.array([
                left_eye_outer[i],
                right_eye_outer[i],
                mouth_right[i],
                mouth_left[i],
                # nose[i]
            ], dtype=np.float32)
        H = solve_H(src, ref)
        face_mesh_warp = [warp_pt(p[i], H) for p in face_mesh]
        img_warp = warp_img(images[i], H)
        bg2.set_data(img_warp)
        left_eye_mesh_warp = [warp_pt(p[i], H) for p in left_eye_mesh]
        right_eye_mesh_warp = [warp_pt(p[i], H) for p in right_eye_mesh]
        mouth_mesh_warp = [warp_pt(p[i], H) for p in mouth_mesh]

        plot_circle(left_eye_lines_warp, pts=left_eye_mesh_warp)
        plot_circle(right_eye_lines_warp, pts=right_eye_mesh_warp)
        plot_circle(mouth_lines_warp, pts=mouth_mesh_warp)

        update_face_mesh(face_plot, face_mesh_warp, face_mesh_connections)
        
        ax[1].set_title('time = {}s'.format(str(time[i])[:4]))

        plt.draw()
        plt.pause(0.03)

        # plt.show()



if __name__ == '__main__':
    csv_file = r'video_outputs\T4_CXY_en_body.csv'
    face_csv = r'video_outputs\T4_CXY_en_face.csv'
    plot_face(csv_file, face_csv)


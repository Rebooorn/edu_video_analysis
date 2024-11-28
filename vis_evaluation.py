import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from detect_utils import cli_face_detect, cli_head_detect
from argparse import Namespace
from time import sleep
import json
from utils import convert_time_to_seconds
from homography_utils import warp_img, warp_pt
from plot_utils import paint_line
from copy import deepcopy
import os
import shutil
from pathlib import Path
import json

def save_png(img, fname):
    img.save(str(fname))

def save_json(data, fname):
    with open(str(fname), "w") as file:
        json.dump(data, file, indent=4)  # indent=4 makes the file pretty-printed

class evaluation_worker:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("expression extraction")
        self.slice = 0

        # vid = r'T1_HJH_en'
        # vid = r'T2_HJC_en'
        # vid = r'T3_HNS_en'
        vid = r'T4_YJH_en'
        self.cap = cv2.VideoCapture(r'./videos/{}.mp4'.format(vid))
        args = Namespace()
        args.time_series_csv = r'./video_outputs/{}_face.csv'.format(vid)
        self.vid = vid
        args.fps_multiplier = 60

        with open('video_clip.json', 'r') as f:
            video_clip_dict = json.load(f)

        if vid in video_clip_dict.keys():
            args.clip_time_start = convert_time_to_seconds(video_clip_dict[vid].split('/')[0])
            args.clip_time_end = convert_time_to_seconds(video_clip_dict[vid].split('/')[1])
        else:
            args.clip_time_start = 0            # sec
            args.clip_time_end = 10000

        print('Video clip from {} to {} s'.format(args.clip_time_start, args.clip_time_end))
        self.fpsm = args.fps_multiplier
        print('Loading face and head expression curve')

        _, ret_face, self.tt, self.face_exp = cli_face_detect(args)
        self.ret_face_mesh = ret_face[3]
        self.ret_H = ret_face[4]
        self.ret_face_mesh_warp = ret_face[5]

        # print(len(self.tt), len(self.face_exp))
        _, _, self.tt, self.head_exp = cli_head_detect(args)
        # print(len(self.tt), len(self.head_exp))
        # self.image_size = [540, 360]
        # self.canvas_size = [800, 360]
        self.image_size = [270, 180]
        self.canvas_size = [400, 180]
        # print(self.face_exp)
        self.face_exp = np.array(self.face_exp)
        self.head_exp = np.array(self.head_exp)
        # print(self.face_exp.max())
        self.face_exp = np.nan_to_num(self.face_exp, nan=0.0)
        self.head_exp = np.nan_to_num(self.head_exp, nan=0.0)
        # print(self.face_exp.max())
        # args.fps_multiplier = 10
        self.preload_frames(args.fps_multiplier, args)            # all frames are now in self.frames

        self.total_frames = len(self.tt)

        # print(self.tt)
        print(self.total_frames, len(self.frames))

        self.autoplay = True

        self.fval = 0
        self.hval = 0

        # Create image widget
        self.image = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1], bg="white")
        self.image.create_image(0, 0, anchor=tk.NW, image=self.frames[0])
        self.image.grid(row=0, column=0, padx=10, pady=10)

        # create a image widget to show homography
        self.hm_image = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1], bg="white")
        # hm_image = 
        self.hm_image.create_image(0, 0, anchor=tk.NW, image=self.frames[0])
        self.hm_image.grid(row=1, column=0, padx=10, pady=10)

        # create label to show time
        self.time_text = tk.StringVar(value='Start')
        self.label = tk.Label(self.root, textvariable=self.time_text, font=("Helvetica", 16))
        self.label.grid(row=3, column=1, padx=10, pady=10)

        # image_label2 = tk.Label(self.root, image=tk_image2)
        self.canvas1 = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1], bg="white")          # fval
        self.canvas1.grid(row=0, column=1, padx=10, pady=10)
        self.canvas2 = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1], bg="white")          # hval
        self.canvas2.grid(row=1, column=1, padx=10, pady=10)

        self.show_mesh = False

        # Create a slider
        self.slider = ttk.Scale(self.root, from_=0, to=100, orient="horizontal", command=self.on_slider_change)
        self.slider.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Create buttons
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.grid(row=3, column=0)


        button1 = ttk.Button(buttons_frame, text="Play", command=self.on_play_button_click)
        button1.pack(side=tk.LEFT, padx=5)

        button2 = ttk.Button(buttons_frame, text="Pause", command=self.on_pause_button_click)
        button2.pack(side=tk.LEFT, padx=5)

        button3 = ttk.Button(buttons_frame, text="Next", command=self.on_next_frame_button_click)
        button3.pack(side=tk.LEFT, padx=5)
        
        button4 = ttk.Button(buttons_frame, text="Last", command=self.on_last_frame_button_click)
        button4.pack(side=tk.LEFT, padx=5)

        button5 = ttk.Button(buttons_frame, text="Save", command=self.on_save_button_click)
        button5.pack(side=tk.LEFT, padx=5)

        button6 = ttk.Button(buttons_frame, text="DetectPts", command=self.on_detectpts_button_click)
        button6.pack(side=tk.LEFT, padx=5)

        # self.x_norm, self.y_norm, self.curve_indicator = self.draw_face_curve()
        self.face_x_norm, self.face_y_norm, self.face_exp_indicator = self.draw_face_curve(width=self.canvas_size[0], height=self.canvas_size[1])
        self.head_x_norm, self.head_y_norm, self.head_exp_indicator = self.draw_head_curve(width=self.canvas_size[0], height=self.canvas_size[1])
        
        # Run the Tkinter event loop
        self.root.mainloop()

    # Function to handle button click
    def on_play_button_click(self):
        # print("Button 1 clicked")
        self.autoplay = True
        self.auto_next_frame()
    
    def on_save_button_click(self):
        # when the 'save' button is clicked:
        # 1. save the current frame, the previous frame, and the homography
        # 2. the current f/h-val
        # 3. the detected keypoints **overlay** the original image, and the homography
        # 3. tag the vid and time.
        tag = self.vid + '@'+ str(self.this_time)[:5] + 's'
        fval = str(self.fval)[:5]
        hval = str(self.hval)[:5]
        det = {
            'fval': fval,
            'hval': hval
        }

        frame = ImageTk.getimage(self.frames[self.frame_number])
        hm_frame = ImageTk.getimage(self.hm_frames[self.frame_number])
        prev_frame = ImageTk.getimage(self.frames[self.frame_number-1])
        prev_hm_frame = ImageTk.getimage(self.hm_frames[self.frame_number-1])

        framePts = ImageTk.getimage(self.framesPts[self.frame_number])
        hm_framePts = ImageTk.getimage(self.hm_framesPts[self.frame_number])
        prev_framePts = ImageTk.getimage(self.framesPts[self.frame_number-1])
        prev_hm_framePts = ImageTk.getimage(self.hm_framesPts[self.frame_number-1])

        # save
        save_path = Path('./paper') / tag
        save_path.mkdir(exist_ok=True)
        save_png(frame, save_path / 'this_frame.png')
        save_png(hm_frame, save_path / 'hm_this_frame.png')
        save_png(prev_frame, save_path / 'prev_frame.png')
        save_png(prev_hm_frame, save_path / 'hm_prev_frame.png')

        save_png(framePts, save_path / 'this_framePts.png')
        save_png(hm_framePts, save_path / 'hm_this_framePts.png')
        save_png(prev_framePts, save_path / 'prev_framePts.png')
        save_png(prev_hm_framePts, save_path / 'hm_prev_framePts.png')

        save_json(det, save_path / 'det.json')

        print('clicked save')

    def on_detectpts_button_click(self):
        if self.show_mesh:
            print('Mesh off')
            self.show_mesh = False
            self.update_frame(self.slice)
        else:
            print('Mesh on')
            self.show_mesh = True
            self.update_frame(self.slice)

    def update_time(self):
        time = self.slice / 100 * self.total_frames
        time = self.tt[int(time)]
        fval = str(self.fval)[:5]
        hval = str(self.hval)[:5]
        self.time_text.set('{} s, fval={}, hval={}'.format(str(time)[:5], fval, hval))
        self.this_time = time
        
    def auto_next_frame(self):
        if self.autoplay:
            self.update_time()
            self.on_next_frame_button_click()
            self.root.after(int(1000/30*self.fpsm), self.auto_next_frame)
    
    def on_pause_button_click(self):
        self.autoplay = False
    
    def on_next_frame_button_click(self):
        self.slice += 100.0 / self.total_frames
        self.slider.set(self.slice)
        self.face_exp_indicator = self.update_curve(self.face_x_norm, self.face_y_norm, self.canvas1, self.face_exp_indicator)
        self.head_exp_indicator = self.update_curve(self.head_x_norm, self.head_y_norm, self.canvas2, self.head_exp_indicator)
        self.update_frame(self.slice)
        self.update_time()

    def on_last_frame_button_click(self):
        self.slice -= 100.0 / self.total_frames
        self.slider.set(self.slice)
        self.face_exp_indicator = self.update_curve(self.face_x_norm, self.face_y_norm, self.canvas1, self.face_exp_indicator)
        self.head_exp_indicator = self.update_curve(self.head_x_norm, self.head_y_norm, self.canvas2, self.head_exp_indicator)
        self.update_frame(self.slice)
        self.update_time()
        
    # Function to handle slider change
    def on_slider_change(self, slice):
        self.slice = float(slice)
        self.face_exp_indicator = self.update_curve(self.face_x_norm, self.face_y_norm, self.canvas1, self.face_exp_indicator)
        self.head_exp_indicator = self.update_curve(self.head_x_norm, self.head_y_norm, self.canvas2, self.head_exp_indicator)
        self.update_frame(self.slice)
        self.update_time()
        print(f"Slider value: {self.slice} %")

    # Function to draw the sine curve on the canvas
    def draw_face_curve_debug(self, width=270, height=180):
        # Generate data points for the sine curve
        x = np.linspace(0, 4 * np.pi, 1000)  
        y = np.sin(x)                        # y values as sin(x)
        r=3

        # Normalize data to fit the canvas
        x_norm = (x - x.min()) / (x.max() - x.min()) * width
        y_norm = (1 - (y - y.min()) / (y.max() - y.min())) * height  # Invert y-axis for Tkinter

        # Draw the curve on the canvas
        for i in range(len(x_norm) - 1):
            self.canvas1.create_line(x_norm[i], y_norm[i], x_norm[i + 1], y_norm[i + 1], fill="blue")
        
        circle = self.canvas1.create_oval(x_norm[0]-r, y_norm[0]-r, x_norm[0]+r, y_norm[0]+r, fill='red')

        return x_norm, y_norm, circle
    
    def draw_curve(self, x, y, canvas, width, height):
        r=3
        y = np.array(y)
        
        # Normalize data to fit the canvas
        x_norm = (x - x.min()) / (x.max() - x.min()) * width
        y_norm = (1 - (y - y.min()) / (y.max() - y.min())) * height  # Invert y-axis for Tkinter
        # print(y.max(), y.min())
        # Draw the curve on the canvas
        # i=0
        # print(x_norm[i], y_norm[i], x_norm[i + 1], y_norm[i + 1])
        for i in range(len(x_norm) - 1):
            canvas.create_line(x_norm[i], y_norm[i], x_norm[i + 1], y_norm[i + 1], fill="blue")
        
        circle = canvas.create_oval(x_norm[0]-r, y_norm[0]-r, x_norm[0]+r, y_norm[0]+r, fill='red')

        return x_norm, y_norm, circle

    def draw_face_curve(self, width=270, height=180):
        x = np.linspace(0, 1, len(self.face_exp))  
        y = self.face_exp                        # y values as sin(x)
        return self.draw_curve(x, y, self.canvas1, width, height)
    
    def draw_head_curve(self, width=270, height=180):
        x = np.linspace(0, 1, len(self.head_exp))  
        y = self.head_exp                        # y values as sin(x)
        return self.draw_curve(x, y, self.canvas2, width, height)      

    def update_curve(self, x_norm, y_norm, canvas, curve_indicator):
        # ind = len(self.x_norm) * self.slice / 100
        # ind = int(ind) 
        # self.canvas1.delete(self.curve_indicator)
        # r = 3
        # self.curve_indicator = self.canvas1.create_oval(self.x_norm[ind]-r, self.y_norm[ind]-r,self.x_norm[ind]+r, self.y_norm[ind]+r,fill='red')
        # print(self.curve_indicator)
        ind = len(x_norm) * self.slice / 100
        ind = int(ind) 
        canvas.delete(curve_indicator)
        r = 3
        curve_indicator = canvas.create_oval(x_norm[ind]-r, y_norm[ind]-r,x_norm[ind]+r, y_norm[ind]+r,fill='red')
        # print(self.curve_indicator)
        self.fval = self.face_exp[ind]
        self.hval = self.head_exp[ind]
        return curve_indicator

    def update_frame(self, prog):
        self.frame_number = self.total_frames * prog / 100
        self.frame_number = int(self.frame_number)
        if not self.show_mesh:
            self.image.create_image(0, 0, anchor=tk.NW, image=self.frames[self.frame_number])
            self.image.image_tk = self.frames[self.frame_number]
            self.hm_image.create_image(0, 0, anchor=tk.NW, image=self.hm_frames[self.frame_number])
            self.hm_image.image_tk = self.hm_frames[self.frame_number]
        else:
            self.image.create_image(0, 0, anchor=tk.NW, image=self.framesPts[self.frame_number])
            self.image.image_tk = self.framesPts[self.frame_number]
            self.hm_image.create_image(0, 0, anchor=tk.NW, image=self.hm_framesPts[self.frame_number])
            self.hm_image.image_tk = self.hm_framesPts[self.frame_number]
            


    def preload_frames(self, fpsm, args):
        # load all frames according to the fpsm, a bit faster
        self.frames = []
        self.hm_frames = []
        self.framesPts = []         # frame with pts
        self.hm_framesPts = []      # homography frames with pts

        nframe = 0
        idx = 0
        t = -0.033333

        print(len(self.ret_H))
        print('Loading frames...')
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if t > args.clip_time_start and t < args.clip_time_end:
                if nframe % fpsm == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # print(t)
                    # Convert the frame to an image object
                    image = Image.fromarray(frame)
                    image = image.resize(self.image_size)
                    image = np.array(image)
                    if idx+1 < len(self.ret_H):
                        hm_image = warp_img(np.array(image), self.ret_H[idx])

                        # TODO: also paint the points?
                        imagePts = deepcopy(image)
                        hm_imagePts = deepcopy(hm_image)
                        for p, hp in zip(self.ret_face_mesh[idx], self.ret_face_mesh_warp[idx]):
                            # print(p)
                            # check whether nan in p and hp
                            if np.isnan(np.min(p)) or np.isnan(np.min(hp)):
                                continue

                            p = [int(i) for i in p]
                            hp = [int(i) for i in hp]
                            imagePts = paint_line(imagePts, p, p)
                            hm_imagePts = paint_line(hm_imagePts, hp, hp)

                        image_tk = ImageTk.PhotoImage(Image.fromarray(image))
                        imagePts_tk = ImageTk.PhotoImage(Image.fromarray(imagePts))
                        hm_image_tk = ImageTk.PhotoImage(Image.fromarray(hm_image))
                        hm_imagePts_tk = ImageTk.PhotoImage(Image.fromarray(hm_imagePts))
                        self.frames.append(image_tk)
                        self.hm_frames.append(hm_image_tk)
                        self.framesPts.append(imagePts_tk)
                        self.hm_framesPts.append(hm_imagePts_tk)
                    idx = idx + 1
                nframe = nframe + 1
                # print(idx)
            t += 0.033333

        self.cap.release()
        print('Done.')

# Create the main window

if __name__ == '__main__':
    evaluation_worker()
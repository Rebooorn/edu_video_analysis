import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from detect_utils import cli_face_detect, cli_head_detect
from argparse import Namespace
from time import sleep


class evaluation_worker:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("expression extraction")
        self.slice = 0

        self.cap = cv2.VideoCapture(r'./videos/T1_CZY_2_en.mp4')
        args = Namespace()
        args.time_series_csv = r'./video_outputs/T1_CZY_2_en_face.csv'
        args.fps_multiplier = 30
        args.clip_time_start = 0
        args.clip_time_end = 10000

        self.fpsm = args.fps_multiplier
        print('Loading face and head expression curve')
        _, _, self.tt, self.face_exp = cli_face_detect(args)
        _, _, self.tt, self.head_exp = cli_head_detect(args)
        self.image_size = [549, 360]
        self.canvas_size = [800, 360]
        # print(self.face_exp)
        self.face_exp = np.array(self.face_exp)
        self.head_exp = np.array(self.head_exp)
        # print(self.face_exp.max())
        self.face_exp = np.nan_to_num(self.face_exp, nan=0.0)
        self.head_exp = np.nan_to_num(self.head_exp, nan=0.0)
        # print(self.face_exp.max())
        self.preload_frames(args.fps_multiplier)            # all frames are now in self.frames

        self.total_frames = len(self.tt)

        # print(self.tt)
        print(self.total_frames, len(self.frames))

        self.autoplay = True

        # Create image widget
        self.image = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1], bg="white")
        self.image.create_image(0, 0, anchor=tk.NW, image=self.frames[0])
        self.image.grid(row=0, column=0, padx=10, pady=10)

        # create label to show time
        self.time_text = tk.StringVar(value='Start')
        self.label = tk.Label(self.root, textvariable=self.time_text, font=("Helvetica", 16))
        self.label.grid(row=1, column=0, padx=10, pady=10)

        # image_label2 = tk.Label(self.root, image=tk_image2)
        self.canvas1 = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1], bg="white")      
        self.canvas1.grid(row=0, column=1, padx=10, pady=10)
        self.canvas2 = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1], bg="white")      
        self.canvas2.grid(row=1, column=1, padx=10, pady=10)

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

    def update_time(self):
        time = self.slice / 100 * self.total_frames
        time = self.tt[int(time)]
        self.time_text.set(str(time)[:5]+'s')
        

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
        print(f"Slider value: {self.slice} %")
        self.face_exp_indicator = self.update_curve(self.face_x_norm, self.face_y_norm, self.canvas1, self.face_exp_indicator)
        self.head_exp_indicator = self.update_curve(self.head_x_norm, self.head_y_norm, self.canvas2, self.head_exp_indicator)
        self.update_frame(self.slice)
        self.update_time()

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
        return curve_indicator

    def update_frame(self, prog):
        self.frame_number = self.total_frames * prog / 100
        self.frame_number = int(self.frame_number)
        self.image.create_image(0, 0, anchor=tk.NW, image=self.frames[self.frame_number])
        self.image.image_tk = self.frames[self.frame_number]

        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # ret, frame = self.cap.read()
        # if ret:
        #     # Convert the frame to RGB format
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        #     # Convert the frame to an image object
        #     image = Image.fromarray(frame)
        #     image = image.resize(self.image_size)
        #     image_tk = ImageTk.PhotoImage(image)
            
        #     # self.image.configure(image = image_tk)
        #     # Update the canvas with the new image
        #     self.image.create_image(0, 0, anchor=tk.NW, image=image_tk)
        #     self.image.image_tk = image_tk

    def preload_frames(self, fpsm):
        # load all frames according to the fpsm, which can be faster
        self.frames = []
        nframe = 0
        print('Loading frames...')
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if (nframe - 1)  % fpsm == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                # Convert the frame to an image object
                image = Image.fromarray(frame)
                image = image.resize(self.image_size)
                image_tk = ImageTk.PhotoImage(image)
                self.frames.append(image_tk)

            nframe = nframe + 1
        self.cap.release()

# Create the main window

if __name__ == '__main__':
    evaluation_worker()
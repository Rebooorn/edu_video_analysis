import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from detect_utils import cli_face_detect, cli_head_detect
from argparse import Namespace

class evaluation_worker:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("expression extraction expression")
        self.slice = 0

        self.cap = cv2.VideoCapture(r'./videos/T1_CZY_2_en.mp4')
        args = Namespace()
        args.time_series_csv = r'./video_outputs/T1_CZY_2_en_face.csv'
        args.fps_multiplier = 30
        args.clip_time_start = 0
        args.clip_time_end = 10000

        _, _, self.tt, self.face_exp = cli_face_detect(args)
        _, _, self.tt, self.head_exp = cli_head_detect(args)
        self.image_size = [270, 180]
        self.canvas_size = [400, 180]

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(self.tt)
        print(self.total_frames)


        # ret, frame = self.cap.read()
        # if ret:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     # Convert the frame to an image object
        #     image = Image.fromarray(frame)
        #     image = image.resize(self.image_size)
        #     # Convert images to Tkinter format
        #     image_tk = ImageTk.PhotoImage(image)

        # # Create image labels
        # self.image = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1], bg="white")
        # self.image.create_image(0, 0, anchor=tk.NW, image=image_tk)
        # self.image.grid(row=0, column=0, padx=10, pady=10)

        # # image_label2 = tk.Label(self.root, image=tk_image2)
        # self.canvas1 = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1], bg="white")      
        # self.canvas1.grid(row=0, column=1, padx=10, pady=10)
        # self.canvas2 = tk.Canvas(self.root, width=self.image_size[0], height=self.image_size[1], bg="white")      
        # self.canvas2.grid(row=0, column=2, padx=10, pady=10)

        # # Create a slider
        # slider = ttk.Scale(self.root, from_=0, to=100, orient="horizontal", command=self.on_slider_change)
        # slider.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # # Create buttons
        # button1 = ttk.Button(self.root, text="Play", command=self.on_play_button_click)
        # button1.grid(row=2, column=0, padx=10, pady=10)

        # button2 = ttk.Button(self.root, text="Pause", command=self.on_pause_button_click)
        # button2.grid(row=2, column=1, padx=10, pady=10)

        # self.x_norm, self.y_norm, self.curve_indicator = self.draw_face_curve()

        # # Run the Tkinter event loop
        # self.root.mainloop()

    # Function to handle button click
    def on_play_button_click(self):
        print("Button 1 clicked")

    def on_pause_button_click(self):
        print("Button 2 clicked")

    # Function to handle slider change
    def on_slider_change(self, slice):
        self.slice = float(slice)
        print(f"Slider value: {self.slice} %")
        self.update_curve()
        self.update_frame(self.slice)

    # Function to draw the sine curve on the canvas
    def draw_face_curve(self, width=270, height=180):
        # Generate data points for the sine curve
        x = np.linspace(0, 4 * np.pi, 1000)  # x values from 0 to 4*pi
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

    def update_curve(self):
        ind = len(self.x_norm) * self.slice / 100
        ind = int(ind) 
        self.canvas1.delete(self.curve_indicator)
        r = 3
        self.curve_indicator = self.canvas1.create_oval(self.x_norm[ind]-r, self.y_norm[ind]-r,self.x_norm[ind]+r, self.y_norm[ind]+r,fill='red')
        print(self.curve_indicator)

    def update_frame(self, prog):
        frame_number = self.total_frames * prog / 100
        frame_number = int(frame_number)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the frame to an image object
            image = Image.fromarray(frame)
            image = image.resize(self.image_size)
            image_tk = ImageTk.PhotoImage(image)
            
            # self.image.configure(image = image_tk)
            # Update the canvas with the new image
            self.image.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.image.image_tk = image_tk

    def preload_frames(self):
        # load all frames according to the fpsm, which can be faster
        self.frames = []
        nframe = 0
        print('Loading frames')
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if (nframe - 1)  % self.fpsm == 0:
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
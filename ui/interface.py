import tkinter as tk 
import ttkbootstrap as ttk 
from tkinter import filedialog

import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageTk

import warnings
warnings.filterwarnings("ignore")

import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

import torch
from torchvision import transforms
import imagetransfer.illation_image as ti
from imagetransfer.utils_image import read_image,imshow,recover_image
import videotransfer.transformer_video as tv
from videotransfer.utils_video import load_cam_image, show_cam_image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StyleTransfer_UI(ttk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

        self.title("Style Transfer")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (ImageStyleTransfer_UI, VideoStyleTransfer_UI):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(ImageStyleTransfer_UI)

        menu_bar = tk.Menu(self)
        menu_bar.add_command(label="Image", command=lambda: self.show_frame(ImageStyleTransfer_UI))
        menu_bar.add_command(label="Video", command=lambda: self.show_frame(VideoStyleTransfer_UI))
        self.config(menu=menu_bar)

        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        

class ImageStyleTransfer_UI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # 上下两个PanedWindow（垂直方向）
        top_bottom_paned_window = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        top_bottom_paned_window.pack(fill=tk.BOTH, expand=True)

        # 上部分
        top_frame = tk.Frame(top_bottom_paned_window)
        top_bottom_paned_window.add(top_frame, minsize=600)

        # 下部分
        bottom_frame = tk.Frame(top_bottom_paned_window)
        top_bottom_paned_window.add(bottom_frame, minsize=50)

        # 上部分的PanedWindow（水平方向）
        top_paned_window = tk.PanedWindow(top_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        top_paned_window.pack(fill=tk.BOTH, expand=True)

        # 左侧部分
        self.left_frame = tk.Frame(top_paned_window, borderwidth=1, relief="flat")
        top_paned_window.add(self.left_frame, minsize=450)

        # 中间部分
        self.middle_frame = tk.Frame(top_paned_window, borderwidth=1, relief="flat")
        top_paned_window.add(self.middle_frame, minsize=500)

        # 右侧部分
        self.right_frame = tk.Frame(top_paned_window, borderwidth=1, relief="flat")
        top_paned_window.add(self.right_frame, minsize=100)

        # 添加你的图形界面元素和功能代码

        # 在底部部分添加标签
        Style_label = ttk.Label(bottom_frame, text="Style Image")
        Style_label.pack(side="left", padx=180, pady=10)
        Content_label = ttk.Label(bottom_frame, text="Content Image")
        Content_label.pack(side="left", padx=230, pady=10)
        Output_label = ttk.Label(bottom_frame, text="Output Image")
        Output_label.pack(side="left", padx=100, pady=10)

        # 为标签绑定事件
        Style_label.bind("<Button-1>", lambda event: self.load_and_display_image(self.left_frame, event, "style"))
        Content_label.bind("<Button-1>", lambda event: self.load_and_display_image(self.middle_frame, event, "content"))
        Output_label.bind("<Button-1>",lambda event: self.transfer_and_display_image())

        # 初始化属性
        self.style_image_path = None
        self.content_image_path = None

    def get_model_path(self,style_path):
        style_image_name, _ = os.path.splitext(os.path.basename(style_path))
        model_name = 'transform_' + style_image_name + '.pth'
        current_script_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        for root, dirs, files in os.walk(current_script_directory):
            for file in files:
                if file == model_name:
                    model_path = os.path.join(root, file)
        
        return model_path

    def load_and_display_image(self, frame, event, image_type):
        for widget in frame.winfo_children():
            widget.destroy()

        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg")])

        if image_path:
            if image_type == "style":
                self.style_image_path = image_path
            elif image_type == "content":
                self.content_image_path = image_path

            image = Image.open(image_path)

            frame_width = frame.winfo_width()
            frame_height = frame.winfo_height()
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:
                new_width = frame_width
                new_height = int(frame_width / aspect_ratio)
            else:
                new_width = int(frame_height * aspect_ratio)
                new_height = frame_height
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(image)

            image_label = tk.Label(frame, image=image_tk)
            image_label.image = image_tk
            image_label.pack(anchor="center", fill="both", expand=True)

    def transfer_and_display_image(self):

        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
        ])

        # 加载风格图像
        style_path = self.style_image_path

        # 获取模型路径
        model_path = self.get_model_path(style_path)

        # 加载预训练模型参数
        transform_net = ti.TransformNet(32).to(device)
        transform_net.load_state_dict(torch.load(model_path))

        # 将模型设为评估模式
        transform_net.eval()

        # 加载图像并进行推理
        content_path = self.content_image_path
        content_image = Image.open(content_path).convert('RGB')
        content_image = data_transform(content_image).unsqueeze(0).to(device)
        output_image = transform_net(content_image)

        # 将张量恢复为图像
        output_image = recover_image(output_image)

        plt.imshow(output_image)
        plt.show()


        output_image = Image.fromarray(output_image)

        # 调整输出图像大小
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        frame_width = self.right_frame.winfo_width()
        frame_height = self.right_frame.winfo_height()
        aspect_ratio = output_image.width / output_image.height
        if aspect_ratio > 1:
            new_width = frame_width
            new_height = int(frame_width / aspect_ratio)
        else:
            new_width = int(frame_height * aspect_ratio)
            new_height = frame_height
        
        output_image = output_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 显示推理后的图像
        image_tk = ImageTk.PhotoImage(output_image)

        image_label = tk.Label(self.right_frame, image=image_tk)
        image_label.image = image_tk
        image_label.pack(anchor="center", fill="both", expand=True)

                
        


class VideoStyleTransfer_UI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


        right_frame_1 = ttk.Frame(self)
        right_frame_1.pack(side="left")

        right_frame_2 = ttk.Frame(self)
        right_frame_2.pack(side="right")


        self.canvas_1 = tk.Canvas(right_frame_1, bg="white", width=600, height=450, highlightthickness=0)
        self.canvas_1.pack(expand=True, fill="both",pady=50,padx=(60, 20))
        self.canvas_2 = tk.Canvas(right_frame_2, bg="white", width=600, height=450, highlightthickness=0)
        self.canvas_2.pack(expand=True, fill="both", pady=50, padx=(0, 70))

        border_width = 3
        self.canvas_1.create_line(0, 0, 0, self.canvas_1.winfo_reqheight(), width=border_width, fill="gray")
        self.canvas_1.create_line(0, 0, self.canvas_1.winfo_reqwidth(), 0, width=border_width, fill="gray")
        self.canvas_1.create_line(0, self.canvas_1.winfo_reqheight(), self.canvas_1.winfo_reqwidth(), self.canvas_1.winfo_reqheight(), width=border_width, fill="gray")
        self.canvas_1.create_line(self.canvas_1.winfo_reqwidth(), 0, self.canvas_1.winfo_reqwidth(), self.canvas_1.winfo_reqheight(), width=border_width, fill="gray")
        self.canvas_2.create_line(0, 0, 0, self.canvas_2.winfo_reqheight(), width=border_width, fill="gray")
        self.canvas_2.create_line(0, 0, self.canvas_2.winfo_reqwidth(), 0, width=border_width, fill="gray")
        self.canvas_2.create_line(0, self.canvas_2.winfo_reqheight(), self.canvas_2.winfo_reqwidth(), self.canvas_2.winfo_reqheight(), width=border_width, fill="gray")
        self.canvas_2.create_line(self.canvas_2.winfo_reqwidth(), 0, self.canvas_2.winfo_reqwidth(),self.canvas_2.winfo_reqheight(), width=border_width, fill="gray")


        button_1 = ttk.Button(right_frame_1, text="source video", style="outline",command=lambda: self.open_video())
        button_1.pack(pady=10)
        button_2 = ttk.Button(right_frame_2, text="transferred video", style="outline")
        button_2.pack(pady=10)

        self.vid = None
        self.update_video()

        self.bind("<Button-3>", self.change_model)
    
    def change_model(self, event):
        new_model_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Model Files", "*.model")])

        if new_model_path:
            self.model_path = new_model_path
            print(f"Selected new model: {self.model_path}")
            self.load_model(self.model_path)
    
    def load_model(self, model_path):
        print("Loading Transformer Network")
        self.net = tv.TransformNet()
        self.net.load_state_dict(torch.load(self.model_path))
        print(self.model_path)
        self.net.to(device)
        print("Loading Transformer Network")

    def open_video(self):
        self.vid = cv2.VideoCapture(0)

    def display_original_video(self,frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        scale_factor = 0.92  
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        photo = ImageTk.PhotoImage(image=Image.fromarray(small_frame))

        image_width = photo.width()
        image_height = photo.height()
        
        canvas_center_x = self.canvas_1.winfo_reqwidth() // 2
        canvas_center_y = self.canvas_1.winfo_reqheight() // 2
        
        image_x = canvas_center_x - image_width // 2
        image_y = canvas_center_y - image_height // 2
        
        self.canvas_1.create_image(image_x, image_y, anchor=tk.NW, image=photo)
        self.canvas_1.photo = photo

    def display_transferred_video(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        torch.cuda.empty_cache()

        content_tensor = load_cam_image(frame)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_frame = content_transform(content_tensor)
        content_frame = content_frame.unsqueeze(0).to(device)

        output = self.net(content_frame).cpu()
        frame = show_cam_image(output[0].detach())

        scale_factor = 0.92  
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        photo = ImageTk.PhotoImage(image=Image.fromarray(small_frame))

        image_width = photo.width()
        image_height = photo.height()
        
        canvas_center_x = self.canvas_2.winfo_reqwidth() // 2
        canvas_center_y = self.canvas_2.winfo_reqheight() // 2

        image_x = canvas_center_x - image_width // 2
        image_y = canvas_center_y - image_height // 2
        
        self.canvas_2.create_image(image_x, image_y, anchor=tk.NW, image=photo)
        self.canvas_2.photo = photo

        
    def update_video(self):
        if self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                self.display_original_video(frame)

                self.display_transferred_video(frame)


        self.after(10, self.update_video)


if __name__ == "__main__":
    app = StyleTransfer_UI(themename="cosmo")
    app.geometry("1400x650")
    #app.resizable(0,0)
    app.mainloop()
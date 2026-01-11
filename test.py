import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import torch
import time
from utils import convert_image
from models import Generator

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预训练模型
srgan_checkpoint = "./results/checkpoint_srgan.pth"
checkpoint = torch.load(srgan_checkpoint)

generator = Generator(large_kernel_size=large_kernel_size,
                      small_kernel_size=small_kernel_size,
                      n_channels=n_channels,
                      n_blocks=n_blocks,
                      scaling_factor=scaling_factor)
generator = generator.to(device)
generator.load_state_dict(checkpoint['generator'])
generator.eval()
model = generator


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        global img, original_img
        original_img = Image.open(file_path).convert('RGB')
        img = original_img.copy()
        display_image(img, panelA)
        bicubic_img = img.resize((img.width * scaling_factor, img.height * scaling_factor), Image.Resampling.BICUBIC)
        display_image(bicubic_img, panelB)


def process_image():
    if img is None:
        messagebox.showerror("Error", "Please open an image first.")
        return

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')

    print('用时  {:.3f} 秒'.format(time.time() - start))

    display_image(sr_img, panelC)


def display_image(img, panel):
    img = img.resize((400, 400), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk


# 创建主窗口
root = tk.Tk()
root.title("SRGAN Image Super-Resolution")
root.geometry("1500x600")

# 使用 grid 布局
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

# 显示原始图像
panelA = tk.Label(root, bg="white")
panelA.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# 显示双线性上采样图像
panelB = tk.Label(root, bg="white")
panelB.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

# 显示处理后的图像
panelC = tk.Label(root, bg="white")
panelC.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")

# 按钮框架
button_frame = tk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=3, pady=20)

# 打开图像按钮
btnOpen = tk.Button(button_frame, text="Open Image", command=open_image, height=2, width=15)
btnOpen.pack(side="left", padx=20)

# 处理图像按钮
btnProcess = tk.Button(button_frame, text="Process Image", command=process_image, height=2, width=15)
btnProcess.pack(side="right", padx=20)

# 初始化图像变量
img = None
original_img = None

# 启动GUI
root.mainloop()

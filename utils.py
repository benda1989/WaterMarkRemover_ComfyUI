import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from torch.hub import download_url_to_file
from comfy.model_management import get_torch_device

DEVICE = get_torch_device()

LAMA_MODEL_PATH = Path(__file__).parent.absolute()
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"


def tensor2video(tensor: torch.Tensor, output_prefix: str, fps: int = 30):
    import folder_paths
    import time
    ourfile =os.path.join(folder_paths.get_output_directory(),f"{output_prefix}_{time.time_ns()}.mp4")

    frames = tensor.cpu().numpy()  
   
    if frames.dtype == np.float32 or frames.dtype == np.float64:
        frames = (frames * 255).astype(np.uint8)
    h, w = frames.shape[1:3]
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #（MP4）
    video_writer = cv2.VideoWriter(ourfile, fourcc, fps, (w, h))
    

    for frame in frames:
        video_writer.write(frame[..., ::-1])  # OpenCV  RGB -> BGR
    
    video_writer.release()
    print(f"video saved：  {ourfile}")
    return ourfile

#if img length is not a multiple of 8, then return the length divided by 8
def fitlength(x) -> int:
    if x % 8 == 0:
        return x
    return int(x // 8 + 1) * 8


# pad image
def pad(img,mask=False):
    # w, h are original image size
    w, h = img.size
    #x, y are padded image size
    x = fitlength(w)
    y = fitlength(h)

    if x!= w or y!= h:
        if mask:
            bgimg = Image.new("L", (x, y), 0)
        else:
            bgimg = Image.new("RGB", (x, y), (0, 0, 0))
        bgimg.paste(img, (0, 0, w, h))
        return bgimg    
    return img

# Tensor to PIL
def tensor2pil(image):
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


# Convert PIL to Tensor
def pil2tensor(image, device=DEVICE):
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise Exception("Input image should be either PIL Image!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    try:
        img = img.astype(np.float32) / 255
    except:
        img = img.astype(np.float16) / 255
    
    out_image = torch.from_numpy(img).unsqueeze(0).to(device)
    return out_image

# crop image
def cropimage(img, x, y):
    return img.crop((0, 0, x, y))


# pil to comfy (i, 3, w, h) -> (i, h, w, 3)
def pil2comfy(img):
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

# download models
def get_model_path(filename, url=LAMA_URL, localdir=LAMA_MODEL_PATH):
    model_path = localdir.joinpath(filename)
    if not os.path.exists(model_path):
        print(f"biglama model not found, downloading...")
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            download_url_to_file(url, model_path) 
        except:
            print(f"model download failed, please download it manually")

    return model_path


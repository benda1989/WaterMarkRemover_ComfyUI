import torch
from torchvision import transforms
from PIL import ImageOps, ImageFilter
from .utils import *


class Base:
    def __init__(self):
        self.device = DEVICE
        model_path = get_model_path(filename="big-lama.pt")
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
        except:
            print(f"can't use comfy device: {model_path}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.jit.load(model_path, map_location=self.device)

        self.model.eval()
        self.model.to(self.device)

    def Mask(self, image, mask):
        with torch.inference_mode():
            result = self.model(image, mask)
            return result[0]


class Remover(Base):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_threshold": ("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gaussblur_radius": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "GKK·WaterMark"

    def image(self, image):
        ori_image = tensor2pil(image)
        self.size = ori_image.size
        p_image = pad(ori_image)
        self.psize = p_image.size
        return pil2tensor(p_image)

    def mask(self, image, isMask=True):
        ten2pil = transforms.ToPILImage()
        if isMask:
            mask = image.unsqueeze(0)
        else:
            mask = image.movedim(0, -1).movedim(0, -1)

        ori_mask = ten2pil(mask).convert('L')

        p_mask = pad(ori_mask, True)
        if p_mask.size != self.psize:
            print("resize:", self.psize, p_mask.size)
            p_mask = p_mask.resize(self.psize)

        if not self.invert_mask:
            p_mask = ImageOps.invert(p_mask)

        p_mask = p_mask.filter(ImageFilter.GaussianBlur(
            radius=self.gaussblur_radius))
        gray = p_mask.point(lambda x: 0 if x > self.mask_threshold else 255)
        self.pt_mask = pil2tensor(gray)
        return self.pt_mask

    def one(self, image, mask=None):
        pt_image = self.image(image)
        if mask is not None:
            print(f"input image size :{image.shape}, mask size :{mask.shape}")
            self.mask(mask)
   
        result = self.Mask(pt_image, self.pt_mask)
        if result.shape[1] > self.size[0] or result.shape[2] > self.size[1]:
            return pil2comfy(cropimage(transforms.ToPILImage()(result), self.size[0], self.size[1]))
        result = result.permute(0, 3, 2, 1)  # (i, 3, w, h) -> (i, h, w, 3)
        print(result.shape)
        return result

    def run(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        self.mask_threshold, self.gaussblur_radius, self.invert_mask = mask_threshold, gaussblur_radius, invert_mask
        results = [self.one(images[0], masks[0])]
        return (torch.cat(results, dim=0),)


class RemoverVideo(Remover):
    def __init__(self):
        super().__init__()

    def run(self, images, masks, mask_threshold, gaussblur_radius, invert_mask):
        from tqdm import tqdm
        self.mask_threshold, self.gaussblur_radius, self.invert_mask = mask_threshold, gaussblur_radius, invert_mask
        tbar = tqdm(total=len(images), unit='frame', position=0)
        for i, frame in enumerate(images):
            if i == 0:
                result = self.one(frame, masks[0])
            else:
                result = torch.cat((result, self.one(frame)), dim=0)
            torch.cuda.empty_cache()
            tbar.update(1)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "Remover": Remover,
    "VideoRemover": RemoverVideo
}

__all__ = ['NODE_CLASS_MAPPINGS']

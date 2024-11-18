from PIL import Image
import torch

def center_crop_resize(img: Image.Image, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)

    left = (width - new_dimension) // 2
    top = (height - new_dimension) // 2
    right = (width + new_dimension) // 2
    bottom = (height + new_dimension) // 2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img

def common_upscale(samples, width, height, upscale_method: str, center=True):
    if center:
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:,:,y:old_height-y,x:old_width-x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

def upscale(samples, upscale_method: str, scale_by: float | int):
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, center=False)
    return s
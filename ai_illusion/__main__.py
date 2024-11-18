import pathlib
import logging

from PIL import Image

from infer import infer

imgs_dir = pathlib.Path(__file__).parent.parent.joinpath("imgs")
out_dir = pathlib.Path(__file__).parent.parent.joinpath("out")

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    out_image, seed = infer(
        Image.open(imgs_dir.joinpath("explosion.png")),
        "Mountainous landscape",
        negative_prompt="poor quality; low resolution"
    )

    out_image.save(out_dir.joinpath("out.png"))
import numpy as np
from PIL import ImageDraw


class RandomMask(object):

    def __call__(self, img):
        width, height = img.size
        mask_xs = np.random.randint(0, width, size=2)
        mask_ys = np.random.randint(0, height, size=2)

        mask_x0, mask_x1 = np.min(mask_xs), np.max(mask_xs)
        mask_y0, mask_y1 = np.min(mask_ys), np.max(mask_ys)

        draw = ImageDraw.Draw(img)
        draw.rectangle((mask_x0, mask_y0, mask_x1, mask_y1), fill=0)
        del draw

        return img

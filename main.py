import numpy as np
import cv2
import os
from tqdm import tqdm


class LSB(object):
    def __init__(self, background_addr, watermark_addr, embed_bit=2):
        self.background = cv2.imread(background_addr)
        self.watermark = cv2.imread(watermark_addr)
        self.embed_bit = embed_bit
        self.synthesis = None
        self.extract_background = None
        self.extract_watermark = None

        self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2RGB)
        self.watermark = cv2.cvtColor(self.watermark, cv2.COLOR_BGR2RGB)
        self.background_backup = self.background.copy()
        self.watermark_backup = self.watermark.copy()

        watermark_height, watermark_width, watermark_color = self.watermark.shape
        background_height, background_width, background_color = self.background.shape
        if watermark_width > background_width or watermark_height > background_height:
            raise ValueError("水印图片大小应小于背景图片")

    @staticmethod
    def bit_space(img):
        """
        将图像的rgb数值转化为长度为8的向量
        eg: 33 -> (0, 0, 0, 1, 0, 0, 0, 1)
        """
        height, width, color = img.shape
        space = np.zeros(shape=(height, width, 8, color))
        for color_i in range(color):
            flag = 0b00000001
            for bit_i in range(8):
                bit_plane = img[..., color_i] & flag
                bit_plane[bit_plane != 0] = 1
                space[..., bit_i, color_i] = bit_plane
                flag <<= 1
        return space.astype(np.uint8)

    def embed(self):
        watermark_height, watermark_width, watermark_color = self.watermark.shape
        background_height, background_width, background_color = self.background.shape

        bs_background = self.bit_space(self.background)
        bs_watermark = self.bit_space(self.watermark)

        for c_i in range(background_color):
            for embed_bit_i in range(self.embed_bit):
                bs_background[0:watermark_height, 0:watermark_width, embed_bit_i, c_i] = \
                    bs_watermark[0:watermark_height, 0:watermark_width, (8 - self.embed_bit) + embed_bit_i, c_i]

        __synthesis = np.zeros_like(self.background)
        for c_i in range(background_color):
            for bit_i in range(8):
                __synthesis[..., c_i] += bs_background[..., bit_i, c_i] * np.power(2, bit_i)
        self.synthesis = __synthesis.astype(np.uint8)

    def extract(self):
        bs_synthesis = self.bit_space(self.synthesis)
        self.extract_watermark = np.zeros_like(self.synthesis)
        self.extract_background = np.zeros_like(self.synthesis)
        for color_i in range(3):
            for i in range(8):
                if i < self.embed_bit:
                    self.extract_watermark[..., color_i] += bs_synthesis[..., i, color_i] * np.power(2, (8 - self.embed_bit) + i)
                else:
                    self.extract_background[..., color_i] += bs_synthesis[..., i, color_i] * np.power(2, i)
        self.extract_watermark = self.extract_watermark.astype(np.uint8)
        self.extract_background = self.extract_background.astype(np.uint8)

    def run(self):
        self.embed()
        self.extract()
        self.background_backup = cv2.cvtColor(self.background_backup, cv2.COLOR_RGB2BGR)
        self.watermark_backup = cv2.cvtColor(self.watermark_backup, cv2.COLOR_RGB2BGR)
        self.synthesis = cv2.cvtColor(self.synthesis, cv2.COLOR_RGB2BGR)
        self.extract_background = cv2.cvtColor(self.extract_background, cv2.COLOR_RGB2BGR)
        self.extract_watermark = cv2.cvtColor(self.extract_watermark, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    if not os.path.exists("result"):
        os.mkdir("result")
    for i in tqdm(range(1, 12)):
        lsb = LSB(f"images/bg{i}.png", "images/wm.png")
        lsb.run()
        cv2.imwrite(f"result/extract_watermark_{i}.jpg", lsb.extract_watermark)
        cv2.imwrite(f"result/extract_background_{i}.jpg", lsb.extract_background)


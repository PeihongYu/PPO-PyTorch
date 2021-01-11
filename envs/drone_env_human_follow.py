import airsim
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import OrderedDict
from matplotlib import pyplot as plt

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

from envs.drone_env import drone_env

np.set_printoptions(precision=3, suppress=True)

class HumanFollow(drone_env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # connect to the AirSim simulator
        super().__init__()

        start_time = time.time()
        self.module_handle = '/home/yu/Music/Surreal/surreal/tensorflow_hub_model/openimages_v4_ssd_mobilenet_v2_1'
        self.detector = hub.load(self.module_handle).signatures['default']
        end_time = time.time()
        print("time cost of ssd tensorflow hub loading: ", end_time - start_time, 's')

        self.state = self.getState()

        self.out_img = np.zeros((480, 640, 3))
        self.area = 0
        self.reward = None

        self.MAX_NULL_STEPS = 10
        self.CURR_NULL_STEPS = 0

    def reset(self):
        super().reset()
        self.state = self.getState()
        return self.state

    def getState(self):
        np_rgb_image = self.getImg('rgb')
        inp_img = cv2.resize(np_rgb_image, (640, 480))
        np_rgb_image, box_loc, area = self.run_detector_local(inp_img)

        state = OrderedDict()
        state["depth"] = self.getImg('depth')
        state["image"] = np_rgb_image
        if box_loc.size == 0:
            state["bbx"] = np.zeros(4)
            state["robot-state"] = np.zeros(4)
            state["area"] = 0
        else:
            state["bbx"] = box_loc[0]
            state["robot-state"] = self.normalize_bbx(box_loc[0])
            state["area"] = area
        return state

    def step(self, action):
        self.moveByDist(action)
        state_cur = self.getState()

        self.out_img = state_cur["image"]
        self.area = state_cur["area"]
        self.reward = self.get_reward(state_cur)

        done = False

        if self.CURR_NULL_STEPS == self.MAX_NULL_STEPS:
            self.CURR_NULL_STEPS = 0
            done = True
        elif self.area == 0:
            self.CURR_NULL_STEPS += 1
        else:
            self.CURR_NULL_STEPS = 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        self.cur_step += 1
        self.state = state_cur

        print("cur_step:", self.cur_step, "bbx:", self.state["robot-state"], "action:", action,
              "  reward: ", self.reward, " null step: ", self.CURR_NULL_STEPS)
        # self.render()

        return self.state, self.reward, done, info

    def get_reward(self, state_cur):
        state_pre = self.state
        area_pre = state_pre["area"]
        area_cur = state_cur["area"]
        area_gain = area_cur - area_pre

        img_center = np.array([480, 640]) / 2
        bbx_center_pre = np.array(
            [(state_pre["bbx"][0] + state_pre["bbx"][2])/2, (state_pre["bbx"][1] + state_pre["bbx"][3])/2])
        bbx_center_cur = np.array(
            [(state_cur["bbx"][0] + state_cur["bbx"][2]) / 2, (state_cur["bbx"][1] + state_cur["bbx"][3]) / 2])
        center_gain = np.linalg.norm(bbx_center_pre - img_center) - np.linalg.norm(bbx_center_cur - img_center)
        print("area_gain: ", area_gain, "center_gain: ", center_gain)
        return area_gain + center_gain

    def normalize_bbx(self, bbx):
        # ymin, xmin, ymax, xmax = tuple(bbx)
        im_size = np.array([480, 640, 480, 640])
        bbx = (bbx - im_size/2) / im_size
        return bbx

    def render(self, mode='human'):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)     # Blue color in BGR
        thickness = 2           # Line thickness of 2 px
        self.out_img = cv2.putText(self.out_img, f'Area:{self.area}', org, font, fontScale, color, thickness, cv2.LINE_AA)
        plt.imshow(self.out_img[:, :, ::-1])
        plt.show()

    def close(self):
        pass

    def run_detector_local(self, img):
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        result = self.detector(converted_img)
        end_time = time.time()

        result = {key: value.numpy() for key, value in result.items()}

        my_filter = result['detection_class_entities'] == b'Person'
        for key, val in result.items():
            result[key] = val[my_filter]

        # print("Found %d objects." % len(result["detection_scores"]))
        # print("Inference time: ", end_time-start_time)

        image_with_boxes, score = self.draw_boxes(
            img, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"])

        box_loc = result["detection_boxes"]

        return image_with_boxes, box_loc, score

    def draw_bounding_box_on_image(self, image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
        """Adds a bounding box to an image."""
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                  width=thickness, fill=color)

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=font)
            text_bottom -= text_height - 2 * margin

    def draw_boxes(self, image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
        """Overlay labeled boxes on an image with formatted scores and label names."""
        colors = list(ImageColor.colormap.values())

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                      25)
        except IOError:
            print("Font not found, using default font.")
            font = ImageFont.load_default()

        area = 0.
        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                               int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                self.draw_bounding_box_on_image(
                    image_pil,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                    color,
                    font,
                    display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
                area += (xmax - xmin) * (ymax - ymin)
        return image, area



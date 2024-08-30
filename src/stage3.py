import os
import shutil

from PIL import Image
import cv2
import numpy as np
import torch

import utils
import random
from transformers import SamModel, SamProcessor


def vlm_label_read(file_path, image_path):

    # read bbox from file
    bbox = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            bbox.append(float(line.strip()))

    # convert the bbox coords from percentages to pixels
    img_h, img_w, _ = cv2.imread(image_path).shape
    _, cx, cy, w, h = bbox
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h

    return [int(x1), int(y1), int(x2), int(y2)]


def step_1(output_img_path, output_mask_path, real_imgs_path, real_imgs_labels):
    # USE OWL to get bounding boxes of the unseen object and use the detected bounding boxes to crop the object

    # load needed model from transformers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor_sam = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    for img_name in os.listdir(real_imgs_path):

        img_path = real_imgs_path + img_name
        file_name = img_name.replace("jpeg", "").replace("jpg", "")
        save_img_name = img_name.replace(".jpeg", "").replace(".jpg", "")

        real_imgs_labels_path = real_imgs_labels + file_name + "txt"
        bbox = vlm_label_read(real_imgs_labels_path, img_path)
        image = Image.open(img_path)

        # ignore the small bbox
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 100:
            continue

        input_boxes = [bbox]
        inputs = processor_sam(image, input_boxes=[input_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_sam(**inputs)

        masks = processor_sam.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )

        masks_imgs = utils.mask_high_score(masks[0])

        # extract the object from the image and set background to white
        # convert torch tensor to cv2 image
        image_np = np.array(image)

        # get object through mask
        for idx, mask_image in enumerate(masks_imgs):
            whole_obj = image_np * np.array(mask_image)
            whole_mask_cv2 = np.array(mask_image) * 255

            # crop obj by bbox
            img = whole_obj[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            mask_cv2 = whole_mask_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]]

            # filter out masks and boxes that are too small
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            mask_per = utils.cal_mask_per(mask_cv2)
            if mask_per <= 0.5 and bbox_area < 2000:
                continue

            # convert mask to np.unit8
            mask_cv2 = np.uint8(mask_cv2)

            # save data to img and mask folders
            cv2.imwrite(f"{output_img_path}{idx}_{save_img_name}.jpeg", img)
            cv2.imwrite(f"{output_mask_path}{idx}_{save_img_name}.jpeg", mask_cv2)


def step_2(
    bg_imgs_path,
    bg_labels_path,
    blend_img_path,
    blend_labels_path,
    output_img_path,
    output_mask_path,
    class_id,
    num_bg_imgs,
):
    """
    Create synthetic training data by blending images of a class of interest onto randomly
    selected background image, and save the resulting images and labels.
    """

    bg_imgs = os.listdir(bg_imgs_path)
    class_imgs = os.listdir(output_img_path)

    if len(bg_imgs) < num_bg_imgs:
        print(f"Provide more background images in the {bg_imgs_path} folder.")
    else:
        for rand_img in random.sample(bg_imgs, num_bg_imgs):

            # get random bg img and labels
            rand_labels = rand_img.replace("jpeg", "txt").replace("jpg", "txt")
            bg_img = bg_imgs_path + rand_img
            bg_labels = bg_labels_path + rand_labels

            # get random class of interest img and labels
            rand_class = random.choice(class_imgs)
            class_img = output_img_path + rand_class
            class_mask = output_mask_path + rand_class

            # generate synthetic image and labels
            bbox = utils.bbox_gen(bg_img, class_img)
            syn_img, syn_labels = utils.gradient_synobj(bg_img, bg_labels, class_img, class_mask, bbox, class_id)

            # generate path for new image and labels
            syn_bg_img_name = f"{rand_img[:-5]}_{rand_class[:-5]}.jpeg"
            syn_labels_name = f"{rand_img[:-5]}_{rand_class[:-5]}.txt"
            syn_bg_img_path = blend_img_path + syn_bg_img_name
            syn_labels_path = blend_labels_path + syn_labels_name

            # write syn_img and syn_labels
            cv2.imwrite(syn_bg_img_path, syn_img)
            np.savetxt(syn_labels_path, syn_labels)


def step_3(bg_imgs, bg_labels, syn_imgs, syn_labels, dest_imgs, dest_labels):

    # copy synthesised images and labels to the train folder
    for i in os.listdir(syn_imgs):
        shutil.copy(syn_imgs + i, dest_imgs + i)
    for i in os.listdir(syn_labels):
        shutil.copy(syn_labels + i, dest_labels + i)

    # copy background images and labels to the train folder
    for i in os.listdir(bg_imgs):
        shutil.copy(bg_imgs + i, dest_imgs + i)
    for i in os.listdir(bg_labels):
        shutil.copy(bg_labels + i, dest_labels + i)


if __name__ == "__main__":

    # params
    pre_path = "/app/"
    class_id = 80
    num_bg_imgs = 10

    output_img_path = pre_path + "data/stage3/objs/obj/"
    output_mask_path = pre_path + "data/stage3/objs/mask/"
    bg_imgs_path = pre_path + "data/stage2/false_pos/images/"
    bg_labels_path = pre_path + "data/stage2/false_pos/labels/"
    syn_imgs_path = pre_path + "data/stage3/synthesised/images/"
    syn_labels_path = pre_path + "data/stage3/synthesised/labels/"
    real_imgs_path = pre_path + "data/stage2/vlm_detect/images/"
    real_imgs_labels = pre_path + "data/stage2/vlm_detect/labels/"
    dest_img_path = pre_path + "trainer/train/images/"
    dest_label_path = pre_path + "trainer/train/labels/"

    os.makedirs(output_img_path, exist_ok=True)
    os.makedirs(output_mask_path, exist_ok=True)
    os.makedirs(syn_imgs_path, exist_ok=True)
    os.makedirs(syn_labels_path, exist_ok=True)

    step_1(output_img_path, output_mask_path, real_imgs_path, real_imgs_labels)
    step_2(
        bg_imgs_path,
        bg_labels_path,
        syn_imgs_path,
        syn_labels_path,
        output_img_path,
        output_mask_path,
        class_id,
        num_bg_imgs,
    )
    step_3(bg_imgs_path, bg_labels_path, syn_imgs_path, syn_labels_path, dest_img_path, dest_label_path)

    print("Stage 3 is complete")

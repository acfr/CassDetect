import os
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import utils
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor
from tqdm import tqdm
import random


def step_1(bg_imgs_path, bg_labels_path):
    """
    Label background images.
    """
    print("Processing initial background field images...")

    model_pretrain = YOLO("yolov8m.pt")
    bg_labels = os.listdir(bg_labels_path)

    for img_name in os.listdir(bg_imgs_path):
        file_name = img_name.replace("jpeg", "").replace("jpg", "")

        # skip images that have already been processed
        if file_name + "txt" in bg_labels:
            continue

        # get labels
        img_path = bg_imgs_path + img_name
        results = model_pretrain(img_path)
        img_h, img_w, _ = cv2.imread(img_path).shape
        labels = utils.get_labels(results, img_h, img_w)

        # save labels
        if len(labels) > 0:
            np.savetxt(bg_labels_path + file_name + "txt", labels, fmt="%d %f %f %f %f")
        else:
            with open(bg_labels_path + file_name + "txt", "w") as f:
                f.write("")

    print("Initial background field images are processed!")


def step_2(pre_path, output_img_path, output_mask_path):
    """
    Generate synthetic data.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import sam
    model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor_sam = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # import owlv2
    model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
    processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
	
    # cassowary_prompt = [["black bird"],['flightless bird']]  # prompt for open word vocabulary cassowary detection for fast processing
    # prompt from wordnet: flightless bird,ratite,ratite bird ,black flightless bird,black ratite ,black ratite bird,cassowary
    aug_cassowary_prompt = [['flightless bird'],['atite'],['ratite bird '],['black flightless bird'],['black ratite '],['black ratite bird'],['cassowary']]
    real_imgs_path = pre_path + "data/stage1/class_images/"

    # generate synthetic data
    print("Generating synthetic data...")

    for img_name in tqdm(os.listdir(real_imgs_path)):

        img_path = real_imgs_path + img_name
        save_img_name = img_name.replace(".jpeg", "").replace(".jpg", "")

        #image, scores, boxes = utils.open_voc_detect(img_path, cassowary_prompt, model_owl, processor_owl)
        image, scores, boxes = utils.open_voc_detect_lbaug(img_path, aug_cassowary_prompt,model_owl, processor_owl, threshould=0.5)
        bbox = utils.cv2_plot(image, scores, boxes)
        bbox = utils.inside_remove(bbox)

        if len(bbox) > 0:

            # keep a numpy version of the image available
            image_np = np.array(image)  # (H, W, C)

            # pass image through sam model
            inputs = processor_sam(image, input_boxes=[bbox], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_sam(**inputs)

            # get masks and convert them to a numpy array (-1, H, W)
            masks = processor_sam.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0].numpy()
            masks = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))

            # get object through mask
            for idx, mask in enumerate(masks):

                # filter out masks that are too small
                if utils.cal_mask_per_v2(mask) < 0.1:
                    continue

                # convert the mask to an image
                mask = np.expand_dims(mask, -1)
                img = image_np * np.array(mask)

                mask_cv2 = np.array(mask) * 255
                mask_cv2 = np.uint8(mask_cv2)

                # save data to img and mask folders
                cv2.imwrite(f"{output_img_path}{idx}_{save_img_name}.jpeg", img)
                cv2.imwrite(f"{output_mask_path}{idx}_{save_img_name}.jpeg", mask_cv2)


def step_3(pre_path, bg_imgs_path, bg_labels_path, output_img_path, output_mask_path, class_id, num_bg_imgs):
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
            syn_bg_img_path = f"{pre_path}data/stage1/synthesised/images/{syn_bg_img_name}"
            syn_labels_path = f"{pre_path}data/stage1/synthesised/labels/{syn_labels_name}"

            # write syn_img and syn_labels
            cv2.imwrite(syn_bg_img_path, syn_img)
            np.savetxt(syn_labels_path, syn_labels)


def step_4(pre_path, bg_imgs, bg_labels, syn_imgs, syn_labels):

    dest_imgs = pre_path + "trainer/train/images/"
    dest_labels = pre_path + "trainer/train/labels/"

    os.makedirs(dest_imgs, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

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
    output_img_path = pre_path + "data/stage1/objs/obj/"
    output_mask_path = pre_path + "data/stage1/objs/mask/"
    bg_imgs_path = pre_path + "data/stage1/background/images/"
    bg_labels_path = pre_path + "data/stage1/background/labels/"
    syn_imgs = pre_path + "data/stage1/synthesised/images/"
    syn_labels = pre_path + "data/stage1/synthesised/labels/"

    os.makedirs(output_img_path, exist_ok=True)
    os.makedirs(output_mask_path, exist_ok=True)
    os.makedirs(bg_labels_path, exist_ok=True)
    os.makedirs(syn_imgs, exist_ok=True)
    os.makedirs(syn_labels, exist_ok=True)

    step_1(bg_imgs_path, bg_labels_path)
    step_2(pre_path, output_img_path, output_mask_path)
    step_3(pre_path, bg_imgs_path, bg_labels_path, output_img_path, output_mask_path, 80, 10)
    step_4(pre_path, bg_imgs_path, bg_labels_path, syn_imgs, syn_labels)

    print("Stage 1 is complete")

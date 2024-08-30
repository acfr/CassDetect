### stage 2: deploy the model to field data and collection FP and TP
import os
import shutil

from PIL import Image
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import utils
from transformers import Owlv2ForObjectDetection, Owlv2Processor


def step_1(pre_path, field_data_path, target_classes):
    """
    Iterate over field data and copy data into new folder if image contains a bbox with target class.
    NOTE: This step is only here as a buffer to reduce the amount of data that the VLM needs to process.
    """

    # setup
    input_folder = pre_path + "data/stage2/raw_data/"  ##raw data downloaded from device
    raw_field_data = os.listdir(input_folder)
    processed_field_data = os.listdir(field_data_path)
    model_path = pre_path + "data/training/stage1/weights/best.pt"
    model = YOLO(model_path)

    # get list of class names
    img_path = input_folder + raw_field_data[0]
    img = cv2.imread(img_path)
    results = model(img)
    names = results[0].names

    for raw_img in raw_field_data:
        # skip images that have already been processed
        if raw_img in processed_field_data:
            continue

        # get prediction
        img_path = input_folder + raw_img
        img = cv2.imread(img_path)
        results = model(img)

        # copy image if containing a bbox with target class
        # for class_id in results[0].boxes.cls.tolist():
        # name = names[int(class_id)]
        # if name in target_classes:
        # shutil.copy(img_path, field_data_path + raw_img)
        # break

        # TODO Remove copy image always
        shutil.copy(img_path, field_data_path + raw_img)


def step_2(pre_path, field_data_path, prompts, label_id_write):
    """
    Iterate over collected data and generate VLM labels.
    """
    img_save_path = pre_path + "data/stage2/vlm_detect/images/"
    label_save_path = pre_path + "data/stage2/vlm_detect/labels/"

    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)

    ## define the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=device)

    img_files = os.listdir(field_data_path)
    for img_file in img_files:
        image_path = field_data_path + "/" + img_file

        #_, scores, boxes = utils.open_voc_detect(image_path, prompts, model, processor)
        _, scores, boxes = utils.open_voc_detect_lbaug(image_path, prompts, model, processor, threshould=0.5)
        vlm_labels = utils.write_label(boxes, scores, label_id_write, score_threshold=0.05)
        # save vlm labels
        if len(vlm_labels) > 0:
            img_file = img_file.replace("jpeg", "txt").replace("jpg", "txt")
            np.savetxt(label_save_path + img_file, vlm_labels)
            shutil.copy(image_path, img_save_path)


def step_3(pre_path, field_data_path, label_id_write):
    """
    Relabel the FP and TP data.
    """
    img_save_path = pre_path + "data/stage2/vlm_detect/images/"
    label_save_path = pre_path + "data/stage2/vlm_detect/labels/"

    field_data_files = os.listdir(field_data_path)
    tp_files = os.listdir(img_save_path)

    tp_imgs_path = pre_path + "data/stage2/true_pos/images/"
    tp_labels_path = pre_path + "data/stage2/true_pos/labels/"
    fp_imgs_path = pre_path + "data/stage2/false_pos/images/"
    fp_labels_path = pre_path + "data/stage2/false_pos/labels/"

    os.makedirs(tp_imgs_path, exist_ok=True)
    os.makedirs(tp_labels_path, exist_ok=True)
    os.makedirs(fp_imgs_path, exist_ok=True)
    os.makedirs(fp_labels_path, exist_ok=True)

    ## load models and class name
    model_pretrain = YOLO("yolov8m.pt")
    sel_id = [11, 2, 0, 7]  # stop sign, car, person, truck

    # Combine TP labels from VLM and YOLO
    for img_name in tp_files:

        # setup
        file_name = img_name.replace("jpeg", "").replace("jpg", "")
        img = Image.open(img_save_path + img_name)
        img_w, img_h = img.size

        results = model_pretrain(img, device="cuda")
        labels = utils.get_labels(results, img_h, img_w)

        # load existing labels from VLM
        vlm_label = np.loadtxt(label_save_path + file_name + "txt")
        vlm_label = np.expand_dims(vlm_label, 0)

        # get new labels from YOLO and filter for classes of interest
        filtered_labels = [la for la in labels if la[0] in sel_id]
        filtered_labels = np.array(filtered_labels)

        # merge existing labels with filtered labels
        if len(filtered_labels) > 0:
            merged_labels = np.vstack((vlm_label, filtered_labels))
            tp_label = utils.overlap_filter(merged_labels, label_id_write)
        else:
            tp_label = vlm_label

        # write to disk
        np.savetxt(tp_labels_path + file_name + "txt", tp_label, fmt="%d %f %f %f %f")
        shutil.copy(img_save_path + img_name, tp_imgs_path + img_name)
        print("")

    # Use the remaining images from the field data for the FP dataset
    fp_files = [file for file in field_data_files if file not in tp_files]

    for img_name in fp_files:

        # setup
        file_name = img_name.replace("jpeg", "").replace("jpg", "")
        img_path = field_data_path + img_name
        img = Image.open(img_path)
        img_w, img_h = img.size

        results = model_pretrain(img, device="cuda")
        labels = utils.get_labels(results, img_h, img_w)

        shutil.copy(img_path, fp_imgs_path + img_name)
        if len(labels) > 0:
            np.savetxt(fp_labels_path + file_name + "txt", labels, fmt="%d %f %f %f %f")
        else:
            with open(fp_labels_path + file_name + "txt", "w") as f:
                f.write("")


def step_4(path):
    """
    Merge all dataset.
    """
    stage1_imgs = path + "data/stage1/synthesised/images/"
    stage1_labels = path + "data/stage1/synthesised/labels/"

    stage2_tp_imgs = path + "data/stage2/true_pos/images/"
    stage2_tp_labels = path + "data/stage2/true_pos/labels/"

    stage2_fp_imgs = path + "data/stage2/false_pos/images/"
    stage2_fp_labels = path + "data/stage2/false_pos/labels/"

    final_imgs = path + "trainer/train/images/"
    final_labels = path + "trainer/train/labels/"

    for img in os.listdir(stage1_imgs):
        shutil.copy(stage1_imgs + img, final_imgs + img)
    for label in os.listdir(stage1_labels):
        shutil.copy(stage1_labels + label, final_labels + label)

    for img in os.listdir(stage2_tp_imgs):
        shutil.copy(stage2_tp_imgs + img, final_imgs + img)
    for label in os.listdir(stage2_tp_labels):
        shutil.copy(stage2_tp_labels + label, final_labels + label)

    for img in os.listdir(stage2_fp_imgs):
        shutil.copy(stage2_fp_imgs + img, final_imgs + img)
    for label in os.listdir(stage2_fp_labels):
        shutil.copy(stage2_fp_labels + label, final_labels + label)


if __name__ == "__main__":

    # set root path and parameters
    pre_path = "/app/"
    label_id_write = 80
    target_classes = ["cassowary"]
    #prompts = [["black bird"],['flightless bird']]
    prompts = [['flightless bird'],['atite'],['ratite bird '],['black flightless bird'],['black ratite '],['black ratite bird'],['cassowary']]
    field_data_path = pre_path + "data/stage2/processed_data/"

    os.makedirs(field_data_path, exist_ok=True)

    step_1(pre_path, field_data_path, target_classes)
    step_2(pre_path, field_data_path, prompts, label_id_write)
    step_3(pre_path, field_data_path, label_id_write)
    step_4(pre_path)

    print("Stage 2 is complete")

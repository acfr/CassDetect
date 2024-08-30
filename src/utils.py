import math
import os

from PIL import Image
import cv2
import numpy as np
import torch

from transformers.image_utils import ImageFeatureExtractionMixin

mixin = ImageFeatureExtractionMixin()


def open_voc_detect(image_path, texts, model, processor):
    """
    Detect objects in an image using a given model and processor.

    Args:
        image_path (str): Path to the image.
        texts (list): List of text queries in the format [['text1'], ['text2']].
        model: The model used for detection.
        processor: The processor used for preparing inputs.

    Returns:
        tuple: A tuple containing the image, scores of detected objects, and bounding boxes.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open(image_path)
    image = image.resize((512, 512))
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Load example image
    image_size = model.config.vision_config.image_size
    image = mixin.resize(image, image_size)

    # Get prediction logits
    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get boundary boxes
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
    return image, scores, boxes
    
### write a function calcuate the IoU of two boxes
def compute_iou(box1_xcycwh, box2_xcycwh):
    box1 = [box1_xcycwh[0]-box1_xcycwh[2]/2, box1_xcycwh[1]-box1_xcycwh[3]/2, box1_xcycwh[0]+box1_xcycwh[2]/2, box1_xcycwh[1]+box1_xcycwh[3]/2]
    box2 = [box2_xcycwh[0]-box2_xcycwh[2]/2, box2_xcycwh[1]-box2_xcycwh[3]/2, box2_xcycwh[0]+box2_xcycwh[2]/2, box2_xcycwh[1]+box2_xcycwh[3]/2]



    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def NMS(bboxes, scores, iou_threshold=0.5):
   ## todo: non maximum suppression
    ## input:
    ## bboxes: list of boxes in the format of [[cx,cy,w,h],[cx,cy,w,h]]
    ## scores: list of scores in the format of [0.9,0.8]
    ## iou_threshold: threshold to remove the boxes
    ## output:
    ## bboxes: list of boxes in the format of [[cx,cy,w,h],[cx,cy,w,h]]
    ## scores: list of scores in the format of [0.9,0.8]
    ## labels: list of labels in the format of [0,1]
    ## iou: list of iou in the format of [0.9,0.8]

    boxes_id = [i for i in range(len(bboxes))] 
    # sort id by score
    boxes_id = sorted(boxes_id, key=lambda x: scores[x], reverse=True)
    nms_id = []
    if boxes_id:
        nms_id.append(boxes_id.pop(0))
    while len(boxes_id) > 0:
        box_id = boxes_id.pop(0)
        for i in range(len(nms_id)):
            iou = compute_iou(bboxes[box_id], bboxes[nms_id[i]])
            if iou > iou_threshold:
                continue
            elif iou <= iou_threshold:
              nms_id.append(box_id)
    # NMS algorithm, sort by score, remove the boxes which have iou > threshold   
    bboxes_nms = [bboxes[i] for i in nms_id]
    scores_nms = [scores[i] for i in nms_id]

    ## put refine here
    scores_nms = socre_refine(bboxes,scores,bboxes_nms,scores_nms,iou_threshold=0.7,refine_param=0.1)
    return bboxes_nms, scores_nms,nms_id


def socre_refine(bboxes,scores,bboxes_nms,scores_nms,iou_threshold=0.7,refine_param=0.2):
    socres_refine = []
    visited = [False]*len(bboxes)
    for bbox_nms,score_nms in zip(bboxes_nms,scores_nms):
        for i, bbox in enumerate(bboxes):
            if visited[i]:
                continue    
            iou = compute_iou(bbox,bbox_nms)
            
            if iou > iou_threshold:
                score_nms = score_nms + scores[i]*refine_param       
        
        socres_refine.append(score_nms)
    return socres_refine

def open_voc_detect_lbaug(image_path, autoprt,model, processor, threshould):
    ## intput:
    ## image_path: path to image
    ## texts: list of text queries in the format of [['text1'],['text2']]
    ## model: model
    ## processor: processor
    
    ### output:
    ## image: image
    ## scores: scores of detection object
    ## boxes: boxes of detection object
    ## labels: labels of detection object, in the format of [0,1] where 0 is the first text query, 1 is the second text query

    aug_boxes = []
    aug_scores = []
    aug_labels = []


    for lb in autoprt:
        texts = [lb]
        image,scores,boxes = open_voc_detect(image_path, texts, model, processor)
        for i in range(len(scores)):
            if scores[i] > threshould:
                aug_boxes.append(boxes[i])
                aug_scores.append(scores[i])
                #aug_labels.append(labels[i])
  # convert to numpy array
    aug_boxes = np.array(aug_boxes)
    aug_scores = np.array(aug_scores)
    #aug_labels = np.array(aug_labels)
    #print('before nms:',len(aug_boxes))
    nms_boxes, nms_scores, nms_id = NMS(aug_boxes, aug_scores)
    #print('after nms',len(nms_boxes))
    #nms_labels = aug_labels[nms_id]
    return image, nms_scores, nms_boxes    



    
    


def cv2_plot(input_image, scores, boxes, score_threshold=0.05):
    """
    Filter out bboxes below the score_threshold.
    """
    img_h, img_w, _ = np.array(input_image).shape

    bbox = []
    for score, box in zip(scores, boxes):
        if score < score_threshold:
            continue
        # convert to x1y1x2y2 format
        cx, cy, w, h = box
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        # denormalize
        x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        bbox.append([int(x1), int(y1), int(x2), int(y2)])

    # sort bbox according to the area, from largest to smallest
    bbox = sorted(bbox, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    return bbox


def write_label(boxes, scores, label_id, score_threshold=0.05):
    """
    Filter out bboxes below the score_threshold.
    """
    vlm_label = []
    for score, box in zip(scores, boxes):
        if score < score_threshold:
            continue
        cx, cy, w, h = box
        vlm_label.append([label_id, cx, cy, w, h])

    # 2. get the biggest box
    if len(vlm_label) > 1:
        vlm_label = sorted(vlm_label, key=lambda x: x[3] * x[4], reverse=True)

    if len(vlm_label) > 0:
        vlm_label = vlm_label[0]

    return np.array(vlm_label)


def mask_high_score(masks):
    """
    Convert the masks from a torch tensor to a list of numpy arrays.
    """
    masks_imgs = []
    for mask in masks:
        for submask in mask:
            m = submask.unsqueeze(-1)
            masks_imgs.append(np.array(m))
    return masks_imgs


def check_inside(a, b):
    """Is bbox b inside of bbox a?"""
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def inside_remove(bboxes):
    """
    Remove a bbox if it is inside another bbox.
    """
    if len(bboxes) <= 1:
        return bboxes

    # sort bboxes by area to simplify function
    bboxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    remaining = []
    while bboxes:
        current_bbox = bboxes.pop(0)
        remaining.append(current_bbox)
        bboxes = [bbox for bbox in bboxes if not check_inside(current_bbox, bbox)]

    return remaining


def cal_mask_per(mask):
    """Calculates the percentage of non-zero pixels in a given mask."""
    mask = mask / 255
    return np.sum(mask) / (mask.shape[0] * mask.shape[1])


def cal_mask_per_v2(mask):
    """Calculates the percentage of non-zero pixels in a given mask."""
    return np.sum(mask) / (mask.shape[0] * mask.shape[1])


def img_crop(img_obj, mask, threshold):
    ### random crop the image and mask
    img_h, img_w, _ = img_obj.shape

    # crop at least 50% of the image
    crop_h = np.random.randint(int(img_h * threshold), img_h)
    crop_w = np.random.randint(int(img_w * threshold), img_w)
    start_x = np.random.randint(0, img_w - crop_w)
    start_y = np.random.randint(0, img_h - crop_h)

    img_crop = img_obj[start_y : start_y + crop_h, start_x : start_x + crop_w]
    mask_crop = mask[start_y : start_y + crop_h, start_x : start_x + crop_w]

    return img_crop, mask_crop


def img_flip(img_obj, mask):
    """
    Flip the image and mask horizontally.
    """
    img_flip = cv2.flip(img_obj, 1)
    mask_flip = cv2.flip(mask, 1)
    return img_flip, mask_flip


def rotation(img_obj, angle_in_degrees):
    """
    Rotate an image by a given angle in degrees.

    Args:
        img_obj (numpy.ndarray): The input image.
        angle_in_degrees (float): The angle of rotation in degrees.

    Returns:
        numpy.ndarray: The rotated image.
    """
    h, w = img_obj.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle_in_degrees, 1)
    rad = math.radians(angle_in_degrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += (b_w / 2) - img_c[0]
    rot[1, 2] += (b_h / 2) - img_c[1]

    outImg = cv2.warpAffine(img_obj, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


def img_rotate(img_obj, mask):
    """
    Rotate the given image and mask by a random angle.
    """

    angle = np.random.randint(0, 360)
    img_rotate = rotation(img_obj, angle)
    mask_rotate = rotation(mask, angle)
    return img_rotate, mask_rotate


def convert_mask_to_bbox(mask):
    """
    Convert a binary mask to bounding box coordinates.

    Args:
        mask (ndarray): Binary mask of shape (h, w) or (h, w, 1).

    Returns:
        list: Bounding box coordinates [x1, y1, x2, y2]
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    binmask = np.where(mask > 127)
    x1 = int(np.min(binmask[1]))
    x2 = int(np.max(binmask[1]))
    y1 = int(np.min(binmask[0]))
    y2 = int(np.max(binmask[0]))
    return [x1, y1, x2 + 1, y2 + 1]


def read_image_opencv(image):
    if isinstance(image, str):
        assert os.path.exists(image), image
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    elif isinstance(image, Image.Image):
        image = pil_to_opencv(image)
    return image


def read_mask_opencv(mask):
    if isinstance(mask, str):
        assert os.path.exists(mask), mask
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask, Image.Image):
        mask = np.asarray(mask)
    return mask


def get_composite_image(foreground_image, foreground_mask, background_image, bbox):
    """
    Generate composite image through copy-and-paste.

    Args:
        foreground_image (str | numpy.ndarray): The path to foreground image or the background image in ndarray form.
        foreground_mask (str | numpy.ndarray): Mask of foreground image which indicates the foreground object region in the foreground image.
        background_image (str | numpy.ndarray): The path to background image or the background image in ndarray form.
        bbox (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2].

    Returns:
        composite_image (numpy.ndarray): Generated composite image with the same resolution as input background image.
        composite_mask (numpy.ndarray): Generated composite mask with the same resolution as composite image.
    """
    fg_img = read_image_opencv(foreground_image)
    fg_mask = read_mask_opencv(foreground_mask)
    bg_img = read_image_opencv(background_image)
    return gaussian_composite_image(bg_img, fg_img, fg_mask, bbox)


def gaussian_composite_image(bg_img, fg_img, fg_mask, bbox, kernel_size=15):
    """
    Composite a foreground image onto a background image using a Gaussian mask.

    Args:
        bg_img (numpy.ndarray): The background image.
        fg_img (numpy.ndarray): The foreground image.
        fg_mask (numpy.ndarray): The foreground mask.
        bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) of the region to composite.
        kernel_size (int, optional): The size of the Gaussian kernel. Defaults to 15.

    Returns:
        numpy.ndarray: The composite image.

    """
    if fg_mask.shape[:2] != fg_img.shape[:2]:
        fg_mask = cv2.resize(fg_mask, (fg_img.shape[1], fg_img.shape[0]))
    fg_mask = cv2.GaussianBlur(255 - fg_mask, (kernel_size, kernel_size), kernel_size / 3.0)
    fg_mask = 255 - fg_mask
    fg_bbox = convert_mask_to_bbox(fg_mask)
    fg_region = fg_img[fg_bbox[1] : fg_bbox[3], fg_bbox[0] : fg_bbox[2]]
    x1, y1, x2, y2 = bbox
    fg_region = cv2.resize(fg_region, (x2 - x1, y2 - y1), cv2.INTER_CUBIC)
    fg_mask = fg_mask[fg_bbox[1] : fg_bbox[3], fg_bbox[0] : fg_bbox[2]]
    fg_mask = cv2.resize(fg_mask, (x2 - x1, y2 - y1))
    norm_mask = (fg_mask.astype(np.float32) / 255)[:, :, np.newaxis]

    comp_img = bg_img.copy()
    comp_img[y1:y2, x1:x2] = (fg_region * norm_mask + comp_img[y1:y2, x1:x2] * (1 - norm_mask)).astype(comp_img.dtype)
    return comp_img


def bbox_gen(sel_bg_img_path, sel_obj_syn_path, para_per_range=[0.05, 0.15]):
    """
    Generate bounding box and select object synthesis.

    Args:
        sel_bg_img_path (str): Path to the selected background image.
        sel_obj_syn_path (str): Path to the selected object synthesis image.
        prs_imgs (tuple, optional): Size of the processed images. Defaults to (512, 512).
        para_per_range (list, optional): Range of the scale down parameter. Defaults to [0.05, 0.15].

    Returns:
        tuple: A tuple containing the normalized bounding box coordinates, the object image, and the mask.
    """

    ### generate bbox and select obj_syn
    sel_bg_img = cv2.imread(sel_bg_img_path)
    bg_h, bg_w, _ = sel_bg_img.shape

    ### bounding box
    obj_img = cv2.imread(sel_obj_syn_path)
    syn_h, syn_w, _ = obj_img.shape

    scale_down_para = np.random.uniform(para_per_range[0], para_per_range[1])
    min_len = min(bg_h, bg_w)
    if syn_h > syn_w:
        scale_obj_h = int(min_len * scale_down_para)
        scale_obj_w = int(syn_w * scale_obj_h / syn_h)
    else:
        scale_obj_w = int(min_len * scale_down_para)
        scale_obj_h = int(syn_h * scale_obj_w / syn_w)

    obj_img = cv2.resize(obj_img, (scale_obj_w, scale_obj_h))
    res_h, res_w, _ = obj_img.shape
    x1 = np.random.randint(0, int(bg_w - res_w))
    y1 = np.random.randint(0, int(bg_h - res_h))

    x2 = x1 + res_w
    y2 = y1 + res_h

    norm_bbox = [int(x1), int(y1), int(x2), int(y2)]
    return norm_bbox


def gradient_synobj(back_img, back_label, fg_img, back_mask, bbox, cls_id):
    """
    Synthesizes a composite image and label by blending a foreground object onto a background image.

    Args:
        back_img (str): Path to the background image.
        back_label (str): Path to the background label.
        fg_img (str): Path to the foreground image.
        back_mask (str): Path to the background mask.
        bbox (tuple): Bounding box of the foreground object in the background image, in the format (X1, Y1, X2, Y2).
        cls_id (int): Class ID of the foreground object.
        option (str): Synthesis option.

    Returns:
        tuple: A tuple containing the synthesized image (comp_img) and the synthesized label (comp_label).
    """

    comp_img = get_composite_image(fg_img, back_mask, back_img, bbox)

    ## merge labels
    ### read labels
    lbs = []
    with open(back_label, "r") as f:
        for line in f:
            lbs.append(line)

    x1, y1, x2, y2 = bbox
    img_h, img_w, _ = comp_img.shape
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    comp_label = np.array([[cls_id, xc / img_w, yc / img_h, w / img_w, h / img_h]])

    if len(lbs) != 0:
        back_label = np.loadtxt(back_label)
        if back_label.ndim == 1:
            back_label = back_label[np.newaxis, :]
        comp_label = np.vstack([back_label, comp_label])

    return comp_img, comp_label


### write a function calcuate the IoU of two boxes
def compute_iou(box1_xcycwh, box2_xcycwh):
    """
    计算两个边界框的IoU。
    边界框格式：[x1, y1, x2, y2]，其中(x1, y1)是左上角的坐标，(x2, y2)是右下角的坐标。
    """
    box1 = [
        box1_xcycwh[0] - box1_xcycwh[2] / 2,
        box1_xcycwh[1] - box1_xcycwh[3] / 2,
        box1_xcycwh[0] + box1_xcycwh[2] / 2,
        box1_xcycwh[1] + box1_xcycwh[3] / 2,
    ]
    box2 = [
        box2_xcycwh[0] - box2_xcycwh[2] / 2,
        box2_xcycwh[1] - box2_xcycwh[3] / 2,
        box2_xcycwh[0] + box2_xcycwh[2] / 2,
        box2_xcycwh[1] + box2_xcycwh[3] / 2,
    ]

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def get_labels(results, img_h, img_w):
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()

    # Iterate through the results
    labels = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        x, y, w, h = x / img_w, y / img_h, w / img_w, h / img_h
        labels.append([cls, x, y, w, h])

    return np.array(labels)


def overlap_filter(overlapping_labels, class_id):
    """
    Used when labels are coming from both the VLM and YOLO model.
    The VLM does not use NMS and therefore can output many bboxes for the same detection.
    Assumption: There is only 1 class in the raw labels with the class_id

    input: overlapping_labels: np.array with shape (n,5)
    output: labels: np.array with shape (n,5)
    """

    labels = []

    # find label with class of interest
    for x in overlapping_labels:
        if int(x[0]) == class_id:
            labels.append(x)
            break

    if not labels:
        raise SystemExit("[overlap_filter] There needs to be at least 1 class of interest.")

    # keep non-overlapping labels
    for x in overlapping_labels:
        if int(x[0]) == class_id:
            continue
        if compute_iou(labels[0][1:], x[1:]) < 0.2:
            labels.append(x[:])

    return np.array(labels)


def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False
    print(f"{num_freeze} layers are freezed.")

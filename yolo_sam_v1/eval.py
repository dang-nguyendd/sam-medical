import os
import time
from datetime import datetime
import random
import argparse
import logging
from collections import defaultdict
import numpy as np
import cv2
import pickle
import torch
import torch.nn.functional as F

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from constants import MODEL
from functions import InferenceSaver
def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.float32) / 255.0

def test(model, bbox_coords_test, test_files):

    image_root = "./data/CVC-ClinicDB/images/test"
    gt_root    = "./data/CVC-ClinicDB/masks/test"

    model.eval()
    predictor_tuned = SamPredictor(model)

    images_path_list = sorted(test_files)

    num_images = len(images_path_list)

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    smooth = 1e-6

    with torch.no_grad():
        for image_name in images_path_list:

            image = cv2.imread(
                os.path.join(image_root, image_name)
            )
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mask = load_mask(os.path.join(gt_root, image_name))

            prompt_box = bbox_coords_test[image_name]

            predictor_tuned.set_image(image)

            pred, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=prompt_box,
                multimask_output=False,
            )

            # Binary masks
            pred_mask = np.where(
                np.array(pred) >= 0.5,
                1,
                0
            )

            gt_mask = np.where(
                mask > 0.1,
                1,
                0
            )

            # Flatten
            pred_flat = pred_mask.reshape(-1)
            gt_flat = gt_mask.reshape(-1)

            # Confusion matrix
            TP = np.sum(
                (pred_flat == 1) &
                (gt_flat == 1)
            )

            FP = np.sum(
                (pred_flat == 1) &
                (gt_flat == 0)
            )

            FN = np.sum(
                (pred_flat == 0) &
                (gt_flat == 1)
            )

            # Metrics
            dice = (
                (2 * TP + smooth) /
                (2 * TP + FP + FN + smooth)
            )

            iou = (
                (TP + smooth) /
                (TP + FP + FN + smooth)
            )

            precision = (
                (TP + smooth) /
                (TP + FP + smooth)
            )

            recall = (
                (TP + smooth) /
                (TP + FN + smooth)
            )

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall

    results = {
        "dice": total_dice / num_images,
        "iou": total_iou / num_images,
        "precision": total_precision / num_images,
        "recall": total_recall / num_images,
        "num_images": num_images
    }

    return results

if __name__ == '__main__':
    # ===========================
    # YOLO inference
    # ===========================
    datasets = ["test"]

    bbox_coords_test = {}

    test_files = []

    # YOLO model init 
    yolo_model = InferenceSaver(MODEL, conf=0.25, iou=0.5)
    yolo_model.load_model()

    #___ YOLO inference Training ___

    for ds in datasets:
        image_root = f"./data/CVC-ClinicDB/images/{ds}"
        gt_root = f"./data/CVC-ClinicDB/masks/{ds}"

        # sort images
        images_path_list = sorted(
            f for f in os.listdir(image_root)
            if f.endswith((".jpg", ".png", ".tif"))
        )

        for img_name in images_path_list:
            image_path = os.path.join(image_root, img_name)

            prompt_box = yolo_model.inference(image_path)

            if prompt_box is None:
                print(f"[WARNING] No YOLO detection for {img_name} ({ds}). Skipping.")
                continue

            bbox_coords_test[img_name] = prompt_box
            test_files.append(img_name)


    print(f"Test images      : {len(test_files)}")


    model_type = 'vit_b'
    checkpoint = './checkpoints/sam_vit_b_01ec64.pth'#sam_vit_b_01ec64.pth' #sam_vit_l_0b3195.pth

    model = sam_model_registry[model_type](checkpoint=checkpoint)

    for i in range (1,4):
        model.load_state_dict(
            torch.load(
                os.path.join(f"./model_pth/YOLOSAM_v1_freeze_image_run{i}_CVC-ClinicDB/YOLOSAM_v1_freeze_image_run{i}_CVC-ClinicDB-best.pth")
            )
        )

        model.eval()

        test_results = test(model, bbox_coords_test, test_files)

        print(f"Final Test Results: ./model_pth/YOLOSAM_v1_freeze_image_run{i}_CVC-ClinicDB/YOLOSAM_v1_freeze_image_run{i}_CVC-ClinicDB-best.pth")
        print(f"Dice:      {test_results['dice']:.4f}")
        print(f"IoU:       {test_results['iou']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall:    {test_results['recall']:.4f}")   

    # for i in range (10, 23):

    #     model.load_state_dict(
    #         torch.load(
    #             os.path.join(f"./model_pth/YOLOSAM_v1_run2_CVC-ClinicDB/YOLOSAM_v1_run2_CVC-ClinicDB{i}-last.pth")
    #         )
    #     )

    #     model.eval()

    #     test_results = test(model, bbox_coords_test, test_files)

    #     print(f"Final Test Results: ./model_pth/YOLOSAM_v1_run2_CVC-ClinicDB/YOLOSAM_v1_run2_CVC-ClinicDB{i}-last.pth")
    #     print(f"Dice:      {test_results['dice']:.4f}")
    #     print(f"IoU:       {test_results['iou']:.4f}")
    #     print(f"Precision: {test_results['precision']:.4f}")
    #     print(f"Recall:    {test_results['recall']:.4f}")   



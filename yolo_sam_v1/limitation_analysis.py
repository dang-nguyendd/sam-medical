import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry

from constants import MODEL
from functions import InferenceSaver


def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(mask_path)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask.astype(np.float32) / 255.0


def get_bbox_from_mask(mask):
    """
    Get bounding box from a binary mask.

    Returns:
        bbox = [x_min, y_min, x_max, y_max]
        or None if mask is empty.
    """

    binary_mask = (mask > 0.1).astype(np.uint8)

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = xs.min()
    y_min = ys.min()
    x_max = xs.max()
    y_max = ys.max()

    return np.array(
        [x_min, y_min, x_max, y_max],
        dtype=np.float32
    )


def draw_bbox(image, bbox, label=None, thickness=3):
    """
    Draw bounding box on image.
    """

    output = image.copy()

    if bbox is None:
        return output

    x1, y1, x2, y2 = bbox.astype(int)

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        thickness
    )

    if label is not None:
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return output


def visualize_result(
    image,
    yolo_bbox,
    gt_bbox,
    pred_mask,
    gt_mask,
    image_name,
    save_dir
):
    """
    Create a 5-panel visualization:

    1. Input image
    2. YOLO bounding box
    3. Ground-truth bounding box
    4. SAM predicted mask
    5. Ground-truth mask
    """

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Input image
    # --------------------------------------------------
    input_image = image.copy()

    # --------------------------------------------------
    # 2. YOLO bounding box
    # --------------------------------------------------
    yolo_image = draw_bbox(
        image,
        yolo_bbox,
        thickness=3
    )

    # --------------------------------------------------
    # 3. Ground-truth bounding box
    # --------------------------------------------------
    gt_bbox_image = draw_bbox(
        image,
        gt_bbox,
        thickness=3
    )

    # --------------------------------------------------
    # 4. SAM predicted mask
    # --------------------------------------------------
    sam_mask_image = image.copy()

    # Create red overlay for predicted mask
    pred_binary = pred_mask.astype(bool)

    overlay = np.zeros_like(sam_mask_image)
    overlay[:, :, 0] = 255  # Red in RGB

    alpha = 0.45

    sam_mask_image[pred_binary] = (
        sam_mask_image[pred_binary] * (1 - alpha)
        + overlay[pred_binary] * alpha
    ).astype(np.uint8)

    # --------------------------------------------------
    # 5. Ground-truth mask
    # --------------------------------------------------
    gt_mask_image = image.copy()

    gt_binary = gt_mask.astype(bool)

    overlay_gt = np.zeros_like(gt_mask_image)
    overlay_gt[:, :, 1] = 255  # Green in RGB

    gt_mask_image[gt_binary] = (
        gt_mask_image[gt_binary] * (1 - alpha)
        + overlay_gt[gt_binary] * alpha
    ).astype(np.uint8)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 5)
    )

    axes[0].imshow(input_image)
    axes[0].set_title("Input Image")

    axes[1].imshow(yolo_image)
    axes[1].set_title("YOLO Bounding Box")

    axes[2].imshow(gt_bbox_image)
    axes[2].set_title("Ground Truth Bounding Box")

    axes[3].imshow(sam_mask_image)
    axes[3].set_title("SAM Predicted Mask")

    axes[4].imshow(gt_mask_image)
    axes[4].set_title("Ground Truth Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # Remove extension safely
    base_name = os.path.splitext(image_name)[0]

    save_path = os.path.join(
        save_dir,
        f"{base_name}_Kvasir_visualization.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[Saved] {save_path}")


def test(
    model,
    bbox_coords_test,
    test_files,
    visualize=True,
    visualization_dir="./visualizations"
):

    image_root = "./data/Kvasir/images/test"
    gt_root = "./data/Kvasir/masks/test"

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

            print(f"Processing: {image_name}")

            # ==================================================
            # Load image
            # ==================================================
            image_path = os.path.join(
                image_root,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                print(
                    f"[WARNING] Could not read image: {image_path}"
                )
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # ==================================================
            # Load ground-truth mask
            # ==================================================
            mask_path = os.path.join(
                gt_root,
                image_name
            )

            mask = load_mask(mask_path)

            # ==================================================
            # YOLO predicted bounding box
            # ==================================================
            prompt_box = bbox_coords_test[image_name]

            # ==================================================
            # Ground-truth bounding box
            # ==================================================
            gt_bbox = get_bbox_from_mask(mask)

            # ==================================================
            # SAM inference using YOLO bounding box
            # ==================================================
            predictor_tuned.set_image(image)

            pred, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=prompt_box,
                multimask_output=False,
            )

            # ==================================================
            # Binary masks
            # ==================================================
            pred_mask = np.where(
                np.array(pred)[0] >= 0.5,
                1,
                0
            ).astype(np.uint8)

            gt_mask = np.where(
                mask > 0.1,
                1,
                0
            ).astype(np.uint8)

            # ==================================================
            # Flatten
            # ==================================================
            pred_flat = pred_mask.reshape(-1)
            gt_flat = gt_mask.reshape(-1)

            # ==================================================
            # Confusion matrix
            # ==================================================
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

            # ==================================================
            # Metrics
            # ==================================================
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

            # ==================================================
            # Visualization
            # ==================================================
            if visualize:

                visualize_result(
                    image=image,
                    yolo_bbox=prompt_box,
                    gt_bbox=gt_bbox,
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    image_name=image_name,
                    save_dir=visualization_dir
                )

    results = {
        "dice": total_dice / num_images,
        "iou": total_iou / num_images,
        "precision": total_precision / num_images,
        "recall": total_recall / num_images,
        "num_images": num_images
    }

    return results


if __name__ == '__main__':

    # ==========================================================
    # YOLO inference
    # ==========================================================

    datasets = ["test"]

    bbox_coords_test = {}
    test_files = []

    # ==========================================================
    # YOLO model initialization
    # ==========================================================

    yolo_model = InferenceSaver(
        MODEL,
        conf=0.25,
        iou=0.5
    )

    yolo_model.load_model()

    # ==========================================================
    # YOLO inference
    # ==========================================================

    for ds in datasets:

        image_root = f"./data/Kvasir/images/{ds}"
        gt_root = f"./data/Kvasir/masks/{ds}"

        images_path_list = sorted(
            f for f in os.listdir(image_root)
            if f.endswith((".jpg", ".png", ".tif"))
        )
        count = 0
        for img_name in images_path_list:
            if count >= 20:
                break
            image_path = os.path.join(
                image_root,
                img_name
            )

            prompt_box = yolo_model.inference(
                image_path
            )

            if prompt_box is None:

                print(
                    f"[WARNING] No YOLO detection for "
                    f"{img_name} ({ds}). Skipping."
                )

                continue

            bbox_coords_test[img_name] = prompt_box

            test_files.append(img_name)
            count += 1

    print(
        f"Test images: {len(test_files)}"
    )

    # ==========================================================
    # SAM model
    # ==========================================================

    model_type = 'vit_b'

    checkpoint = (
        './checkpoints/sam_vit_b_01ec64.pth'
    )

    model = sam_model_registry[model_type](
        checkpoint=checkpoint
    )

    model.load_state_dict(
        torch.load(
            os.path.join(
                "./model_pth/"
                "YOLOSAM_v1_run1_Kvasir/"
                "YOLOSAM_v1_run1_Kvasir-best.pth"
            )
        )
    )

    model.eval()

    # ==========================================================
    # Test + visualization
    # ==========================================================

    test_results = test(
        model=model,
        bbox_coords_test=bbox_coords_test,
        test_files=test_files,
        visualize=True,
        visualization_dir="./visualizations"
    )

    # ==========================================================
    # Final results
    # ==========================================================

    print("\nFinal Test Results:")

    print(
        f"Dice:      {test_results['dice']:.4f}"
    )

    print(
        f"IoU:       {test_results['iou']:.4f}"
    )

    print(
        f"Precision: {test_results['precision']:.4f}"
    )

    print(
        f"Recall:    {test_results['recall']:.4f}"
    )
from pathlib import Path
import cv2
import numpy as np
import json
from utils import generate_prompt, save_mask_prompt
from segment_anything import sam_model_registry, SamPredictor

def eval(
    image_path="./data/CVC-ColonDB/images/1.png", 
    mask_path="./data/CVC-ColonDB/masks/1.png"
):
    # init SAM with VIT-B
    sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)

    # load image
    image_path = image_path
    mask_path = mask_path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # prompt
    # prompt = np.array([100, 150, 400, 500])  # x0, y0, x1, y1
    prompt = generate_prompt(
        image_path=image_path,
        mask_path=mask_path,
        largest_component=True,
        padding=0,          # pixels
        padding_ratio=0     # percentage
    )


    # predict
    masks, scores, logits = predictor.predict(
        box=prompt,
        multimask_output=False,
    )

    # save results
    save_mask_prompt(
        image_path, 
        masks[0], 
        image, 
        prompt, 
        scores[0])



def calculate_seg_metrics(results_dir, mask_dir):
    """
    Calculate Dice, IoU, Precision, Recall for segmentation masks.

    Args:
        results_dir (Path): directory containing predicted masks
        mask_dir (Path): directory containing ground truth masks

    Returns:
        dict: average metrics
    """

    results_dir = Path(results_dir)
    mask_dir = Path(mask_dir)

    dice_scores = []
    iou_scores = []
    precisions = []
    recalls = []

    pred_paths = sorted(
        results_dir.glob("*.png"),
        key=lambda x: int(x.stem)
    )

    for pred_path in pred_paths:
        gt_path = mask_dir / pred_path.name

        if not gt_path.exists():
            print(f"Missing ground truth: {gt_path}")
            continue

        # Load masks
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        # Binary masks
        pred = pred > 0
        gt = gt > 0

        # True positives, false positives, false negatives
        TP = np.logical_and(pred, gt).sum()
        FP = np.logical_and(pred, np.logical_not(gt)).sum()
        FN = np.logical_and(np.logical_not(pred), gt).sum()

        # Metrics
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

        iou = TP / (TP + FP + FN + 1e-8)

        precision = TP / (TP + FP + 1e-8)

        recall = TP / (TP + FN + 1e-8)


        dice_scores.append(dice)
        iou_scores.append(iou)
        precisions.append(precision)
        recalls.append(recall)

    metrics = {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "count": len(dice_scores)
    }

    return metrics


if __name__ == "__main__":
    
    dataset_path = "./data/CVC-ColonDB"
    dataset_path = Path(dataset_path)

    image_dir = dataset_path / "images"
    mask_dir = dataset_path / "masks"
    prompts_dir = dataset_path / "prompts"
    results_dir = dataset_path / "results"

    # sam inference
    image_paths = sorted(image_dir.glob("*.png"))

    for image_path in image_paths:
        filename = image_path.name

        # Skip if already processed
        prompt_path = prompts_dir / filename
        result_path = results_dir / filename

        if prompt_path.exists() and result_path.exists():
            print(f"Skipping {filename}")
            continue

        # Corresponding mask
        mask_path = mask_dir / filename

        if not mask_path.exists():
            print(f"Mask missing: {mask_path}")
            continue

        print(f"Processing: {filename}")

        # batch evaluation
        eval(
            image_path=str(image_path),
            mask_path=str(mask_path)
        )

    # sam metrics cal
    metrics = calculate_seg_metrics(
        results_dir,
        mask_dir
    )

    with open(f"{dataset_path}/zero_shot_eval.json", "w") as f:
        json.dump(metrics, f, indent=4)

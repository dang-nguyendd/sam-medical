from pathlib import Path


import cv2
import numpy as np
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
        padding=50,          # pixels
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


if __name__ == "__main__":
    dataset_path = "./data/CVC-ColonDB"
    dataset_path = Path(dataset_path)

    image_dir = dataset_path / "images"
    mask_dir = dataset_path / "masks"

    # Get all images
    image_paths = sorted(image_dir.glob("*.png"))

    for image_path in image_paths:
        filename = image_path.name

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

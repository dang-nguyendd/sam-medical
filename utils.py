from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_prompt(
    image_path,
    mask_path,
    largest_component=False,
    padding=0,          # pixels
    padding_ratio=0     # percentage
):
    """
    Generate bounding box from segmentation mask and save visualization.

    Args:
        image_path (str): Path to input image.
        mask_path (str): Path to segmentation mask.
        save_path (str): Output path for bbox visualization.
        largest_component (bool): If True, only use largest connected region.

    Returns:
        np.ndarray: SAM bounding box [x0, y0, x1, y1]
    """

    # load path and filename
    path = Path(image_path)

    base_dir = Path(*path.parts[:2])   # data/CVC-ColonDB
    filename = path.name

    # save path
    save_path = Path(base_dir) / "prompts" / filename
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")

    # Binary mask
    mask_binary = mask > 0

    if not np.any(mask_binary):
        raise ValueError("Mask contains no foreground pixels")

    num_labels = None
    # Option 1: use largest connected component
    if largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_binary.astype(np.uint8),
            connectivity=8
        )

        # Ignore background label 0
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        x, y, w, h, _ = stats[largest_label]

        x0, y0 = x, y
        x1, y1 = x + w, y + h

    # Option 2: use all foreground pixels
    else:
        ys, xs = np.where(mask_binary)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()


    # SAM box format
    x0, y0, x1, y1 = add_prompt_padding(
        mask,
        x0,
        y0,
        x1,
        y1,
        padding_ratio=padding_ratio,
        padding=padding
    )
    bbox = np.array([x0, y0, x1, y1])

    # Draw bounding box
    vis = image.copy()

    cv2.rectangle(
        vis,
        (x0, y0),
        (x1, y1),
        color=(255, 0, 0),   # red in RGB
        thickness=3
    )

    # Save visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    plt.title(f"Bounding Box: {bbox.tolist()}")
    plt.axis("off")

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved prompt visualization: {save_path}")
    print(f"SAM prompt: {bbox}")
    print(f"Prompt largest component: {num_labels}")


    return bbox

def add_prompt_padding(
    mask,
    x0,
    y0,
    x1,
    y1,
    padding_ratio=0,
    padding=0
):
    if padding_ratio == 0 and padding == 0:
        return x0, y0, x1, y1

    img_h, img_w = mask.shape

    # Percentage padding
    if padding_ratio > 0:
        box_w = x1 - x0
        box_h = y1 - y0

        padding_x = int(box_w * padding_ratio)
        padding_y = int(box_h * padding_ratio)

    else:
        padding_x = 0
        padding_y = 0

    # Fixed pixel padding
    padding_x += padding
    padding_y += padding

    # Apply padding
    x0 = max(0, x0 - padding_x)
    y0 = max(0, y0 - padding_y)

    x1 = min(img_w - 1, x1 + padding_x)
    y1 = min(img_h - 1, y1 + padding_y)

    return x0, y0, x1, y1

def save_mask_prompt(image_path, mask, image, prompt, score):
    # load path and filename
    path = Path(image_path)

    base_dir = Path(*path.parts[:2])   # data/CVC-ColonDB
    filename = path.name

    # save path
    mask_bbox_path = Path(base_dir) / "results_prompt" / filename
    mask_path = Path(base_dir) / "results" / filename
    mask_bbox_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    mask_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image)
    ax.imshow(mask, alpha=0.5, cmap="jet")

    x0, y0, x1, y1 = prompt
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)

    ax.set_title(f"Score: {score:.3f}")
    ax.axis("off")
    # export results
    plt.savefig(
        mask_bbox_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # export results
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask)
    


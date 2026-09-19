import os

# === CONFIGURATION ===

# MODEL OPTIONS: ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yolo26", "yoloe", "yolow"]
MODEL = "yolo12"
ALL_MODELS = ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yolo26", "yoloe", "yolow", ]

# DATA OPTIONS: ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir", "Kvasir_CVC-ClinicDB"]
DATA = "Kvasir"


data_root = f"./data/{DATA}/"

download_model_root = f"./models/"

results_root = f"./results_data_{DATA}/"
results_all_root = f"./results_data_all/"
results_saved_model_root = f"{results_root}/saved_models/"
results_fig_root = f"{results_root}/figs/"
results_inf_root = f"{results_root}/inferences/"
results_inf_all_root = f"{results_root}/inferences_all/"
os.makedirs(results_root, exist_ok=True)
os.makedirs(results_all_root, exist_ok=True)
os.makedirs(results_saved_model_root, exist_ok=True)
os.makedirs(results_fig_root, exist_ok=True)
os.makedirs(results_inf_root, exist_ok=True)
os.makedirs(results_inf_all_root, exist_ok=True)


image_folder = f"{data_root}/images/test/"  # Folder with images
label_folder = f"{data_root}/labels/test/"  # Folder with ground truth YOLO labels (txt)

# Define your dataset YAML config.
dataset_yaml_path = data_root + 'data.yaml'
dataset_SEG_yaml_path = data_root + 'data_seg.yaml'

# === TEST DATA ===
DATA_TEST = "ETIS-LaribPolypDB-test"
data_test_root = f"./data/{DATA_TEST}/"
dataset_test_yaml_path = data_test_root + 'data.yaml'
dataset_test_SEG_yaml_path = data_test_root + 'data_seg.yaml'

results_root_test = f"./results_data_{DATA}_on_{DATA_TEST}/"
results_fig_root_test = f"{results_root_test}/figs/"
os.makedirs(results_root_test, exist_ok=True)
os.makedirs(results_fig_root_test, exist_ok=True)

data_test_image_folder = f"{data_test_root}/images/test/"  # Folder with images
data_test_label_folder = f"{data_test_root}/labels/test/"  # Folder with ground truth YOLO labels (txt)


# === MODELS ===
yolo8_model_name = "yolov8s"
yolo8_model_config = f"{download_model_root}/{yolo8_model_name}.pt"
yolo8_model_path = f"{results_saved_model_root}/{yolo8_model_name}.pt"

yolo9_model_name = "yolov9s"
yolo9_model_config = f"{download_model_root}/{yolo9_model_name}.pt"
yolo9_model_path = f"{results_saved_model_root}/{yolo9_model_name}.pt"

yolo10_model_name = "yolov10s"
yolo10_model_config = f"{download_model_root}/{yolo10_model_name}.pt"
yolo10_model_path = f"{results_saved_model_root}/{yolo10_model_name}.pt"

yolo11_model_name = "yolo11s"
yolo11_model_config = f"{download_model_root}/{yolo11_model_name}.pt"
yolo11_model_path = f"{results_saved_model_root}/{yolo11_model_name}.pt"

yolo12_model_name = "yolo12s"
yolo12_model_config = f"{download_model_root}/{yolo12_model_name}.pt"
yolo12_model_path = f"{results_saved_model_root}/{yolo12_model_name}.pt"

detr_model_name = "rtdetr-l"
detr_model_config = f"{download_model_root}/{detr_model_name}.pt"
detr_model_path = f"{results_saved_model_root}/{detr_model_name}.pt"

yoloe_model_name = "yoloe-11s-seg"
yoloe_model_config = f"{download_model_root}/{yoloe_model_name}.pt"
yoloe_model_path = f"{results_saved_model_root}/{yoloe_model_name}.pt"

yoloworld_model_name = "yolov8s-worldv2"
yoloworld_model_config = f"{download_model_root}/{yoloworld_model_name}.pt"
yoloworld_model_path = f"{results_saved_model_root}/{yoloworld_model_name}.pt"


yolo26_model_name = "yolo26s"
yolo26_model_config = f"{download_model_root}/{yolo26_model_name}.pt"
yolo26_model_path = f"{results_saved_model_root}/{yolo26_model_name}.pt"




# CONFIGURABLE SIZE CATEGORIES
# Format: 'category_name': (min_normalized_area, max_normalized_area)
SIZE_CATEGORIES = {
    'small': (0.000, 0.005),    # 0.1% - 0.5%
    'medium': (0.005, 0.01),    # 0.5% - 1.0%
    'large': (0.01, 0.05),      # 1% - 5%
    'very large': (0.05, 1.0)   # > 5%
}

map_names = {
    "rtdetr": "RT-DETR",
    "yolo8": "YOLOv8",
    "yolo9": "YOLOv9",
    "yolo10": "YOLOv10",
    "yolo11": "YOLO11",
    "yolo12": "YOLO12",
    "yoloe": "YOLOE",
    "yolow": "YOLOWorld",
    "yolo26": "YOLO26"
}

def get_size_categories():
    """
    Get the current size categories configuration.

    Returns:
        dict: Dictionary mapping category names to (min, max) area tuples
    """
    return SIZE_CATEGORIES.copy()


def set_size_categories(new_categories):
    """
    Update the size categories configuration.

    Args:
        new_categories (dict): New size categories dictionary
    """
    global SIZE_CATEGORIES
    SIZE_CATEGORIES = new_categories.copy()


def validate_size_categories(categories=None):
    """
    Validate that size categories are properly configured.

    Args:
        categories (dict, optional): Categories to validate. If None, uses SIZE_CATEGORIES.

    Returns:
        tuple: (is_valid, error_message)
    """
    if categories is None:
        categories = SIZE_CATEGORIES

    if not categories:
        return False, "Size categories dictionary is empty"

    # Check that ranges are valid
    for name, (min_val, max_val) in categories.items():
        if min_val < 0 or max_val > 1:
            return False, f"Category '{name}' has invalid range: ({min_val}, {max_val}). Must be in [0, 1]"
        if min_val >= max_val:
            return False, f"Category '{name}' has invalid range: min ({min_val}) >= max ({max_val})"

    return True, "Valid"


def print_size_categories():
    """Print the current size categories in a readable format."""
    print("\nCurrent Size Categories:")
    print("=" * 60)
    for name, (min_val, max_val) in SIZE_CATEGORIES.items():
        min_pct = min_val * 100
        max_pct = max_val * 100
        print(f"  {name:10s}: {min_pct:8.4f}% - {max_pct:8.4f}% of image area")
    print("=" * 60)
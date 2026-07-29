from constants import ALL_MODELS, MODEL
from functions import InferenceSaver


if __name__ == "__main__":
    # Configuration
    conf = 0.25  # Confidence threshold
    iou = 0.5  # IoU threshold for NMS

    # Create batch inference saver
    # for model_name in MODEL:
    print(f"{'=' * 60}")
    print(f"Model: {MODEL}")
    print(f"{'=' * 60}")
    saver = InferenceSaver(MODEL, conf=conf, iou=iou)
    saver.load_model()
    saver.inference("./data/CVC-ColonDB/images/test/11.png")

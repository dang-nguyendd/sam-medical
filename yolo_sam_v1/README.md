# Colonic polyp detection using YOLO 

This repository contains the code and configurations for a study submitted as a scientific research article.


[![DOI](https://img.shields.io/badge/DOI-10.3390/computers15040258-blue)](https://doi.org/10.3390/computers15040258) 

This repository contains the code and configurations used in the study titled:
**"Colonic Polyp Detection with Object Detection Models"**
by Raluca Portase, and Eugen-Richard Ardelean, published in *Computers*, 2026.

## 📄 Overview

This project benchmarks modern object detection models (YOLOv8–v12, YOLO26, YOLOE, YOLO-World, RT-DETR) for automatic polyp detection. It evaluates their ability to detect colonic polyps across three publicly available datasets:

* **CVC-ClinicDB**,
* **CVC-ColonDB**,
* **ETIS-LaribPolypDB**,

Our experiments identify **YOLO-World** and **YOLOv11** as the most effective models, offering an excellent trade-off between detection accuracy and computational efficiency.


---

## 📊 Datasets

This project requires a .yaml file to be created for the models to run on each dataset, as shown in this example:
```
train:  ..\..\data\CVC-ClinicDB\images\train
val:    ..\..\data\CVC-ClinicDB\images\val
test:   ..\..\data\CVC-ClinicDB\images\test
nc: 1
names: [Polyp]
```

### 1. CVC-ClinicDB Dataset

* **Name**: CVC-ClinicDB
* **Description**: public colonoscopy polyp segmentation dataset released by the CVC (Computer Vision Center) group in collaboration with Hospital Clínic (Barcelona). It contains 612 frames (with a size of 348 × 288 pixels) extracted from colonoscopy videos (each paired with a pixel-wise polyp mask); it comprises 31 different polyps from 31 sequences from 23 patients. 
* **Access**: Available at 
  🔗 [CVC-ClinicDB Dataset Page](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/)

### 2. CVC-ColonDB Dataset

* **Name**: CVC-ColonDB
* **Description**: public colonoscopy dataset of still frames containing annotated polyps. It contains 300 polyp frames (with a size of 574 × 500 pixels) selected from colonoscopy videos (drawn from from 13 polyp video sequences from 13 patients) to maximize viewpoint variability per polyp and provided with ground-truth polyp masks. 
* **Access**: Available at 
  🔗 [CVC-ColonDB Dataset Page](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/)

### 3. ETIS-LaribPolypDB Dataset

* **Name**: ETIS-LaribPolypDB
* **Description**: used in the MICCAI 2015 Endoscopic Vision Challenge as the challenge test set. It was assembled by ETIS Lab and Lariboisière Hospital and contains 196 high-definition frames (each with a segmentation mask), collected from dozens of video sequences (comprises 44 different polyps from 34 sequences) and representing a diverse, challenging set of polyps from multiple patients, devices and larger image resolutions. 
* **Access**: Available at 
  🔗 [ETIS-LaribPolypDB Dataset Page](https://polyp.grand-challenge.org/ETISLarib/)



---

## 🧠 Models Evaluated

* YOLOv8 to YOLOv12
* YOLO26
* YOLOE: Prompt-guided detection
* YOLO-World: Open-vocabulary object detection
* RT-DETR: Transformer-based object detection



## 📈 Results Summary

We show here the cross-dataset evaluation for models trained on either CVC-ClinicDB or CVC-ColonDB, then tested on ETIS-LaribPolypDB

| Model     | Dataset         | mAP\@50   | GFLOPs |
| --------- | --------------- | --------- | ------ |
| YOLOv8    | CVC-ClinicDB    | 0.669     | 28.4   |
| YOLOv11   | CVC-ClinicDB    | **0.702** | 21.2   |
| YOLO-World| CVC-ClinicDB    | 0.690     | 32.6   |
| YOLOv8    | CVC-ColonDB     | 0.548     | 28.4   |
| YOLOv11   | CVC-ColonDB     | 0.581     | 21.2   |
| YOLO-World| CVC-ColonDB     | **0.607** | 32.6   |

---

## 📜 Citation

If you use this code or reference the models/datasets in your work, please cite:

```bibtex
@article{10.3390/computers15040258,
    author = {Portase, Raluca and Ardelean, Eugen-Richard},
    title = {Colonic Polyp Detection with Object Detection Models},
    journal = {Computers},
    volume = {15},
    year = {2026},
    number = {4},
    article-number = {258},
    url = {https://www.mdpi.com/2073-431X/15/4/258},
    ISSN = {2073-431X},
    DOI = {10.3390/computers15040258}
}
```

---


## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---


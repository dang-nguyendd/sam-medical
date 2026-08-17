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
from .ultralytics import YOLO

def transform_boxes_torch(boxes, original_size, target_size):
    """
    Transform boxes from original image coordinates
    to SAM's transformed image coordinates.

    boxes: [B, 4] in xyxy
    original_size: (H, W)
    target_size: (H', W')
    """

    old_h, old_w = original_size
    new_h, new_w = target_size

    scale = min(
        new_h / old_h,
        new_w / old_w
    )

    boxes = boxes.clone()

    boxes[:, [0, 2]] *= scale
    boxes[:, [1, 3]] *= scale

    return boxes

def mask_to_bbox(mask):
    """
    Convert binary mask [H, W] to xyxy bounding box.
    """
    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        return None

    x_min = xs.min()
    y_min = ys.min()
    x_max = xs.max()
    y_max = ys.max()

    return np.array(
        [x_min, y_min, x_max, y_max],
        dtype=np.float32
    )

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    lr = init_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.float32) / 255.0

def test(model, bbox_coords_test, test_files):

    image_root = "./data/ETIS-LaribPolypDB/images/test"
    gt_root    = "./data/ETIS-LaribPolypDB/masks/test"

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

def validate(model, bbox_coords_val, val_files):
    model.eval()
    predictor_tuned = SamPredictor(model)

    image_root = "./data/ETIS-LaribPolypDB/images/val"
    gt_root    = "./data/ETIS-LaribPolypDB/masks/val"

    images_path_list = sorted(val_files)

    DSC = 0.0
    with torch.no_grad():
        for image_name in sorted(val_files):
            image = cv2.imread(
                os.path.join(image_root, image_name)
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = load_mask(os.path.join(gt_root, image_name))

            gt_box = mask_to_bbox(mask)

            if gt_box is None:
                continue

            gt_box = torch.tensor(
                gt_box,
                dtype=torch.float32,
                device="cuda"
            ).unsqueeze(0)

            box = bbox_coords_val[image_name]
            
            predictor_tuned.set_image(image)
            pred, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=box,
                multimask_output=False,
            )    
            # eval Dice
            input = np.where(np.array(pred) >= 0.5, 1, 0)
            target = np.where(np.array(mask) > 0.1, 1, 0)

            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

    return DSC / len(images_path_list), len(images_path_list)

def train(
        image_list,
        yolo_model, 
        sam_model,
        bbox_coords_val,
        val_files, 
        optimizer, 
        epoch, 
        model_name='SAM'):
    
    sam_model.train()
    global dict_plot
    global best
    global total_train_time
    time_before_epoch_start = time.time()
    size_rates = [1]

    epoch_losses = []
    epoch_dices = []
    i = 0
    
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    for image_name in image_list:
        optimizer.zero_grad()

        image_root = "./data/ETIS-LaribPolypDB/images/train"
        gt_root    = "./data/ETIS-LaribPolypDB/masks/train"
        image_path = os.path.join(image_root, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_rgb = cv2.cvtColor(
            cv2.imread(image_path),
            cv2.COLOR_BGR2RGB
        )

        # YOLO input
        yolo_image = torch.from_numpy(image_rgb).float()
        yolo_image = yolo_image.permute(2, 0, 1).unsqueeze(0)
        yolo_image = yolo_image.cuda() / 255.0

        boxes, scores = yolo_model(yolo_image)

        temperature = 0.1
        weights = torch.softmax(scores / temperature, dim=1)

        pred_box = (
            weights.unsqueeze(-1) * boxes
        ).sum(dim=1)

        box_torch = transform_boxes_torch(
            pred_box,
            original_image_size,
            input_size
        )

        mask = load_mask(os.path.join(gt_root, image_name))

        input_image_np = transform.apply_image(image)
        input_image = torch.as_tensor(
            input_image_np,
            device="cuda"
        )
        del input_image_np

        transformed_image = (
            input_image.permute(2,0,1)
            .contiguous()
            .unsqueeze(0)
        )

        input_image = sam_model.preprocess(transformed_image)

        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        if freeze_image_encoder:
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
        else:
            image_embedding = sam_model.image_encoder(input_image)

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        gt_binary_mask = (
            torch.from_numpy(mask)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .cuda()
        )
        upscaled_masks = sam_model.postprocess_masks(
            low_res_masks, 
            input_size, 
            original_image_size
        ).cuda()

        gt_binary_mask = gt_binary_mask.squeeze(-1)


        lambda_seg = 1.0
        lambda_det = 0.1

        seg_loss = structure_loss(upscaled_masks, gt_binary_mask)

        total_loss = (
            lambda_seg * seg_loss
            + lambda_det * det_loss
        )

        # ===========================
        # Compute Training Dice
        # ===========================
        with torch.no_grad():
            pred = (torch.sigmoid(upscaled_masks) > 0.5).float()

            intersection = (pred * gt_binary_mask).sum()
            dice = (2 * intersection + 1e-6) / (
                pred.sum() + gt_binary_mask.sum() + 1e-6
            )

            epoch_dices.append(dice.item())
        # ===========================

        total_loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # Clear cache
        del image
        del mask
        del input_image
        del transformed_image
        del image_embedding
        del low_res_masks
        del upscaled_masks
        del gt_binary_mask

        i += 1
        if i % 250 == 0 or i == total_step: 
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                ' loss: {:0.4f}]'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                        total_loss.item()))
                        
        epoch_losses.append(total_loss.item())
    avg_train_loss = np.mean(epoch_losses)
    avg_train_dice = np.mean(epoch_dices)

    dict_plot['train_loss'].append(avg_train_loss)
    dict_plot['train_dice'].append(avg_train_dice)
    
    print(
        'EPOCH: {} loss: {} dice: {}'.format(
            epoch,
            avg_train_loss,
            avg_train_dice,
        )
    )

    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: '+ str(total_train_time))
    logging.info('total train time till current epoch: '+ str(total_train_time))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(sam_model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    if (epoch + 1) % 1 == 0:

        dataset_dice, n_images = validate(sam_model, bbox_coords_val, val_files)
        sam_model.train()

        meandice = dataset_dice

        print("Validation Dice:", meandice)
        logging.info(f"epoch: {epoch}, validation dice: {meandice}")

        dict_plot['val_dice'].append(meandice)

        if meandice > best:
            print(f'##################### Dice score improved from {best} to {meandice}')
            logging.info(f'##################### Dice score improved from {best} to {meandice}')
            best = meandice
            torch.save(sam_model.state_dict(),
                    os.path.join(save_path, model_name + '-best.pth'))
    
if __name__ == '__main__':
    dict_plot = {
        'train_loss': [],
        'train_dice': [],
        'val_dice': [],
        'test': {},
        'training_time': None
    }

    freeze_image_encoder = 0
    freeze_decoder = 1
    freeze_prompt = 0
    
    ##################model_name#############################
    model_name = 'YOLOSAM_v1_freeze_mask_run1_ETIS-LaribPolypDB' 
    ###############################################
    print(model_name)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-3, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=1024, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=300, help='every n epochs decay learning rate')

    parser.add_argument('--train_save', type=str,
        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='log_sam/train_log_'+model_name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    model_type = 'vit_b'
    checkpoint = './checkpoints/sam_vit_b_01ec64.pth'#sam_vit_b_01ec64.pth' #sam_vit_l_0b3195.pth

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    if freeze_image_encoder:
        print("Freezing image encoder")
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
    if freeze_decoder:
        print("Freezing mask decoder")
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False
    if freeze_prompt:
        print("Freezing mask decoder")
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
    sam_model.cuda()

    # YOLO model init 
    yolo = YOLO("./checkpoints/yolov12n.pt")

    yolo_model = yolo.model
    yolo_model.cuda()
    yolo_model.train()

    best = 0

    params = [
        p
        for p in list(yolo_model.parameters())
                + list(sam_model.parameters())
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        params,
        lr=opt.lr,
        weight_decay=1e-4
    )

    print(optimizer)

    train_files = sorted([
        f for f in os.listdir("./data/ETIS-LaribPolypDB/images/train")
        if f.endswith((".jpg", ".png", ".tif"))
    ])

    val_files = sorted([
        f for f in os.listdir("./data/ETIS-LaribPolypDB/images/val")
        if f.endswith((".jpg", ".png", ".tif"))
    ])

    test_files = sorted([
        f for f in os.listdir("./data/ETIS-LaribPolypDB/images/test")
        if f.endswith((".jpg", ".png", ".tif"))
    ])

    print(f"Training images  : {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Test images      : {len(test_files)}")

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, opt.epoch):
        import os
        import psutil

        process = psutil.Process(os.getpid())
        print(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(
            train_files,
            yolo_model,
            sam_model,
            val_files,
            optimizer,
            epoch,
            model_name=model_name
        )


    # Training time
    dict_plot['training_time'] = total_train_time
    print('avg train time: '+ str(total_train_time/(opt.epoch-1)))
    logging.info('avg train time: '+ str(total_train_time/(opt.epoch-1)))

    sam_model.load_state_dict(
        torch.load(
            os.path.join(opt.train_save, model_name + "-best.pth")
        )
    )

    sam_model.eval()

    test_results = test(sam_model, bbox_coords_test, test_files)

    dict_plot['test'] = test_results
    print("Final Test Results:")
    print(f"Dice:      {test_results['dice']:.4f}")
    print(f"IoU:       {test_results['iou']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall:    {test_results['recall']:.4f}")


    history_path = os.path.join(
        opt.train_save,
        "training_history.pkl"
    )

    with open(history_path, "wb") as f:
        pickle.dump(dict_plot, f)

    print(f"Saved training history to {history_path}")

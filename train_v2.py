import os
import time
from datetime import datetime
import random
import argparse
import logging
from collections import defaultdict
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from yolov12.ultralytics.nn.tasks import DetectionModel

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


# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
#     decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= decay

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

def evaluate(model, yolo, data_path):
    model.eval()
    yolo.eval()

    image_root = os.path.join(data_path, "images")
    gt_root = os.path.join(data_path, "masks")

    images_path_list = sorted([
        f for f in os.listdir(image_root)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    DSC = 0.0

    with torch.no_grad():

        for image_name in images_path_list:

            # -----------------------------
            # Read image and GT
            # -----------------------------
            image = cv2.imread(os.path.join(image_root, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(
                os.path.join(gt_root, image_name),
                cv2.IMREAD_GRAYSCALE,
            )
            mask = mask.astype(np.float32) / 255.0

            # -----------------------------
            # SAM preprocessing
            # -----------------------------
            transform = ResizeLongestSide(model.image_encoder.img_size)

            input_image = transform.apply_image(image)

            input_image = torch.as_tensor(
                input_image,
                device="cuda"
            )

            transformed_image = (
                input_image.permute(2, 0, 1)
                .contiguous()
                .unsqueeze(0)
            )

            input_image = model.preprocess(transformed_image)

            input_size = tuple(transformed_image.shape[-2:])
            original_image_size = image.shape[:2]

            # -----------------------------
            # YOLO prediction
            # -----------------------------
            preds = yolo(input_image)

            pred_boxes = preds["pred_boxes"]
            pred_scores = preds["pred_scores"]

            scores = pred_scores.max(dim=-1).values
            weights = torch.softmax(scores * 20, dim=1)

            pred_box = (
                pred_boxes *
                weights.unsqueeze(-1)
            ).sum(dim=1)

            # -----------------------------
            # Convert box to SAM coordinates
            # -----------------------------
            H, W = original_image_size

            scale = model.image_encoder.img_size / max(H, W)

            box = pred_box * scale

            # -----------------------------
            # SAM forward
            # -----------------------------
            image_embedding = model.image_encoder(input_image)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box.unsqueeze(1),
                masks=None,
            )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred = model.postprocess_masks(
                low_res_masks,
                input_size,
                original_image_size,
            )

            pred = torch.sigmoid(pred)

            pred = pred.squeeze().cpu().numpy()

            pred = (pred >= 0.5).astype(np.uint8)
            target = (mask > 0.5).astype(np.uint8)

            smooth = 1

            intersection = (pred * target).sum()

            dice = (
                2 * intersection + smooth
            ) / (
                pred.sum() + target.sum() + smooth
            )

            DSC += dice

    return DSC / len(images_path_list), len(images_path_list)

def train(transformed_data, ground_truth_masks, model, optimizer, epoch, test_path, model_name = 'SAM'):
    model.train()
    global best
    global total_train_time
    time_before_epoch_start = time.time()
    size_rates = [1]

    keys = transformed_data.keys()
    epoch_losses = []
    i = 0
    for k in keys:
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            input_image = transformed_data[k]['image'].cuda()
            # yolo 
            # --------------------------------------------------
            # Build YOLO training batch from segmentation mask
            # --------------------------------------------------

            mask_np = ground_truth_masks[k]

            ys, xs = np.where(mask_np > 0)

            if len(xs) == 0:
                continue

            x1 = xs.min()
            x2 = xs.max()
            y1 = ys.min()
            y2 = ys.max()

            H, W = mask_np.shape

            # normalized xywh
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            batch = {
                "img": input_image,
                "batch_idx": torch.tensor([0], device="cuda"),
                "cls": torch.tensor([0], dtype=torch.float32, device="cuda"),
                "bboxes": torch.tensor(
                    [[xc, yc, bw, bh]],
                    dtype=torch.float32,
                    device="cuda",
                ),
            }

            outputs = yolo.loss(batch)

            pred_boxes = outputs["pred_boxes"]
            pred_scores = outputs["pred_scores"]

            scores = pred_scores.max(dim=-1).values
            weights = torch.softmax(scores * 20, dim=1)
            pred_box = (pred_boxes * weights.unsqueeze(-1)).sum(dim=1)

            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']        

            image_embedding = model.image_encoder(input_image)
            
            H, W = original_image_size
            scale = model.image_encoder.img_size / max(H,W)
            box = pred_box * scale

            # box_torch = torch.as_tensor(box, dtype=torch.float, device='cuda')
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :] # (B, 1, 4)
            # box_torch = box_torch[None, :]
      
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box.unsqueeze(1),
                masks=None,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).cuda()         
            # gt_binary_mask = torch.as_tensor(gt_mask_resized > 0.0, dtype=torch.float32)
            gt_binary_mask = (
                torch.from_numpy(ground_truth_masks[k])
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .cuda()
            )
            upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).cuda()

            # Loss Seg
            loss_seg = structure_loss(upscaled_masks, gt_binary_mask)

            # Loss Det
            loss_det = outputs["loss"]
            pred_boxes = outputs["pred_boxes"]
            pred_scores = outputs["pred_scores"]

            # Loss (Seg + Det)
            loss = loss_det + 0.2 * loss_seg

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            i += 1
            if i % 250 == 0 or i == total_step: 
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss.item()))
                           
            epoch_losses.append(loss.item())
    print('EPOCH: '+ str(epoch) + ' loss: ' + str(np.mean(epoch_losses)))
    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: '+ str(total_train_time))
    logging.info('total train time till current epoch: '+ str(total_train_time))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        "sam": model.state_dict(),
        "yolo": yolo.state_dict(),
    }, save_path + '' + model_name + '-last.pth')
    # torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0
                
        val_dice, _ = evaluate(model, yolo, opt.val_path)

        print(f"Validation Dice: {val_dice:.4f}")
        logging.info(f"Validation Dice: {val_dice:.4f}")

        dict_plot['test'].append(val_dice)

        if val_dice > best:
            print(f"Dice improved from {best:.4f} to {val_dice:.4f}")
            logging.info(f"Dice improved from {best:.4f} to {val_dice:.4f}")

            best = val_dice
            torch.save({
                "sam": model.state_dict(),
                "yolo": yolo.state_dict(),
            }, save_path + '' + model_name + '-best.pth')
            # torch.save(
            #     model.state_dict(),
            #     save_path + model_name + '-best.pth'
            # )   
    

if __name__ == '__main__':
    dict_plot = {'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-300':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]} 
    name = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test'] 

    perturb_h_len = 50
    perturb_l_len = 0
    freeze_image_encoder = 0
    freeze_decoder = 1
    
    ##################model_name#############################
    model_name = 'PolypSAM_freeze_mask_decoder_vit_b_train_allimages_e100_Run1' 
    ###############################################
    print(model_name)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

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

    parser.add_argument('--train_path', type=str,
                        default='./data/CVC-ColonDB/',
                        help='path to train dataset')
    
    parser.add_argument('--val_path', type=str,
                        default='./data/CVC-ColonDB/',
                        help='path to validation dataset')

    parser.add_argument('--test_path', type=str,
                        default='./data/CVC-ColonDB/',
                        help='path to testing dataset')

    parser.add_argument('--train_save', type=str,
                        default='./models/model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_'+model_name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    model_type = 'vit_b'
    checkpoint = './checkpoints/sam_vit_b_01ec64.pth'#sam_vit_b_01ec64.pth' #sam_vit_l_0b3195.pth

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    yolo = DetectionModel(cfg = "yolov12.yaml")
    if freeze_image_encoder:
        print("Freezing image encoder")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if freeze_decoder:
        print("Freezing mask decoder")
        for param in model.mask_decoder.parameters():
            param.requires_grad = False
    model.cuda()
    yolo.cuda()

    best = 0
    params = list(model.image_encoder.parameters()) + list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) #+ list(model.out.parameters()) #+ list(model.pvt_cascade.parameters()) #+ list(model.trans2pvt.parameters()) #+ list(model.pvt_stage2.parameters()) + list(model.pvt_norm2.parameters()) + list(model.pvt_stage3.parameters()) + list(model.pvt_norm3.parameters()) + list(model.pvt_stage4.parameters()) + list(model.pvt_norm4.parameters()) + list(model.decoder.parameters()) #.mask_decoder.   
    optimizer = torch.optim.AdamW(
        list(yolo.parameters()) +
        list(model.parameters()),
        lr=opt.lr,
        weight_decay=1e-4,
    )
    # if opt.optimizer == 'AdamW':
    #     optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    #     #optimizer = torch.optim.Adam(params, opt.lr, weight_decay=0)
    # else:
    #     optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)


    # sort images
    images_path_list = sorted([f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
    
    # Use all images in the training dataset
    img_idxs = list(range(len(images_path_list)))
    print(f'Using all {len(img_idxs)} training images.')
    logging.info(f'Using all {len(img_idxs)} training images.')

    # bbox_coords = {}
    # for k in img_idxs:
    #     im = cv2.imread(gt_root+''+images_path_list[k])
    #     gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
    #     y_indices, x_indices = np.where(gray > 0)
    #     # x_min, x_max, y_min, y_max = PromptGenerator.predict(images_path_list[k])
    #     x_min, x_max, y_min, y_max = 0,0,0,0
    #     bbox_coords[images_path_list[k]] = np.array([x_min, y_min, x_max, y_max])
        
    #print(bbox_coords)
    transformed_data = defaultdict(dict)
    masks = defaultdict(dict)
    for k in img_idxs:
        image = cv2.imread(image_root+''+images_path_list[k])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(gt_root+''+images_path_list[k], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0
        transform = ResizeLongestSide(model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device='cuda')
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  
        input_image = model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[images_path_list[k]]['image'] = input_image
        transformed_data[images_path_list[k]]['input_size'] = input_size
        transformed_data[images_path_list[k]]['original_image_size'] = original_image_size
        masks[images_path_list[k]] = mask   

    total_step = len(transformed_data)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train(
            transformed_data,
            masks,
            model,
            optimizer,
            epoch,
            opt.val_path,          # validation set
            model_name
        )

    print('avg train time: ' + str(total_train_time/(opt.epoch-1)))
    logging.info('avg train time: ' + str(total_train_time/(opt.epoch-1)))

    # ======================================================
    # Load the best checkpoint selected using validation
    # ======================================================
    save_path = opt.train_save

    model.load_state_dict(
        torch.load(os.path.join(save_path, model_name + "-best.pth"))
    )

    # ======================================================
    # Final evaluation on the test set
    # ======================================================
    test_dice, n_images = evaluate(model, yolo, opt.test_path)

    print(f"Final Test Dice: {test_dice:.4f}")
    logging.info(f"Final Test Dice: {test_dice:.4f}")

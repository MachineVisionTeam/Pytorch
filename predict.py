import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt




def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def dice_coefficient(pred, gt):
    intersection = np.sum(pred * gt)
    smooth = 1e-5
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    return dice

def jaccard_index(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    smooth = 1e-5
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard

def hausdorff_distance(pred, gt):
    hd = directed_hausdorff(pred, gt)[0]
    return hd

def sensitivity(pred, gt):
    true_positive = np.sum(np.logical_and(pred == 1, gt == 1))
    actual_positive = np.sum(gt == 1)
    smooth = 1e-5
    sensitivity = (true_positive + smooth) / (actual_positive + smooth)
    return sensitivity

def specificity(pred, gt):
    true_negative = np.sum(np.logical_and(pred == 0, gt == 0))
    actual_negative = np.sum(gt == 0)
    smooth = 1e-5
    specificity = (true_negative + smooth) / (actual_negative + smooth)
    return specificity

def ppv(pred, gt):
    true_positive = np.sum(np.logical_and(pred == 1, gt == 1))
    predicted_positive = np.sum(pred == 1)
    smooth = 1e-5
    ppv = (true_positive + smooth) / (predicted_positive + smooth)
    return ppv

def npv(pred, gt):
    true_negative = np.sum(np.logical_and(pred == 0, gt == 0))
    predicted_negative = np.sum(pred == 0)
    smooth = 1e-5
    npv = (true_negative + smooth) / (predicted_negative + smooth)
    return npv

def f1_score(pred, gt):
    true_positive = np.sum(np.logical_and(pred == 1, gt == 1))
    actual_positive = np.sum(gt == 1)
    predicted_positive = np.sum(pred == 1)
    smooth = 1e-5
    precision = true_positive / (predicted_positive + smooth)
    recall = true_positive / (actual_positive + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def get_ground_truth_filename(input_filename, gt_folder):
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    return os.path.join(gt_folder, f'{base_name}_gt.gif')


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        gt_folder = rf'C:\Users\ADMIN\Desktop\Pytorch\Pytorch-UNet-master\data\masks'  
        gt_filename = get_ground_truth_filename(filename, gt_folder)
        if os.path.exists(gt_filename):
            gt_img = Image.open(gt_filename)
            gt_segmentation = np.array(gt_img)
            
            dice = dice_coefficient(mask, gt_segmentation)
            jaccard = jaccard_index(mask, gt_segmentation)
            hausdorff = hausdorff_distance(mask, gt_segmentation)
            sensitivity_val = sensitivity(mask, gt_segmentation)
            specificity_val = specificity(mask, gt_segmentation)
            ppv_val = ppv(mask, gt_segmentation)
            npv_val = npv(mask, gt_segmentation)
            f1_score_val = f1_score(mask, gt_segmentation)
            
            print(f"Image: {filename}")
            print("Dice Coefficient:", dice)
            print("Jaccard Index:", jaccard)
            print("Hausdorff Distance:", hausdorff)
            print("Sensitivity:", sensitivity_val)
            print("Specificity:", specificity_val)
            print("Positive Predictive Value:", ppv_val)
            print("Negative Predictive Value:", npv_val)
            print("F1-Score:", f1_score_val)

            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

            # Save the predicted mask alongside the ground truth mask
            predicted_mask_path = os.path.join(gt_folder, f'predicted_mask_{i}.png')
            predicted_mask = mask_to_image(mask, mask_values)
            predicted_mask.save(predicted_mask_path)
            logging.info(f'Predicted mask saved to {predicted_mask_path}')

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            axes[1].imshow(gt_segmentation, cmap='gray')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('on')
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('on')
            plt.show()
        else:
            logging.warning(f'Ground truth image not found for {os.path.basename(filename)}')


    

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            
            

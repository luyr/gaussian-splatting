import cv2
import argparse
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--dataroot', type=str, default='/data/datasets/object_gs_dataset/')
    parser.add_argument('--dataset', type=str, default='7')
    
    args = parser.parse_args()
    
    image_dir = os.path.join(args.dataroot, args.dataset, 'images_4')
    mask_dir = os.path.join(args.dataroot, args.dataset, 'masks')
    
    image_list = sorted(os.listdir(image_dir))
    mask_list = sorted(os.listdir(mask_dir))
    
    if not os.path.exists(os.path.join(args.dataroot, args.dataset, 'object')):
        os.makedirs(os.path.join(args.dataroot, args.dataset, 'object'))
    if not os.path.exists(os.path.join(args.dataroot, args.dataset, 'background')):
        os.makedirs(os.path.join(args.dataroot, args.dataset, 'background'))
    
    for i in range(len(image_list)):
        image = cv2.imread(os.path.join(image_dir, image_list[i]))
        mask = cv2.imread(os.path.join(mask_dir, mask_list[i]))
        mask[mask < 125] = 0
        mask[mask >= 125] = 255
        # dilate the mask once
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        background = image.copy()
        object = image.copy()
        # remove the masked area in the image
        background[mask != 0] = 0
        # only keep the masked area in the image
        object[mask == 0] = 0
        
        cv2.imwrite(os.path.join(args.dataroot, args.dataset, 'object', image_list[i]), object)
        cv2.imwrite(os.path.join(args.dataroot, args.dataset, 'background', image_list[i]), background)
        
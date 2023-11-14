import argparse
import json
import os

import imgviz
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils.about_log import logger
from onekey_algo.utils.segmentation_utils import get_color_map_list
from onekey_core.core import create_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def transform_img(image_path, resize_img):
    img = Image.open(image_path).convert('RGB')
    # fname, ext = os.path.splitext(os.path.basename(image_path))
    # img_shape = img.size
    img = resize_img(img)
    image = F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # image = transform(img)
    image = image.unsqueeze(0)
    return image


def inference(image_list, model, class_names, device, save_dir='./', post_process=None, base_size=480,
              with_prob: bool = False):
    results = []
    model.eval()
    if not isinstance(image_list, (list, tuple)):
        image_list = [image_list]
    # transform = transforms.Compose([
    #         transforms.Resize(520),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    create_dir_if_not_exists(os.path.join(save_dir, 'masks'))
    if with_prob:
        create_dir_if_not_exists(os.path.join(save_dir, 'prob'))
    if 'unet' in str(model.__class__).lower():
        resize_img = transforms.Resize((base_size, base_size))
    else:
        resize_img = transforms.Resize(base_size)
    with torch.no_grad():
        for image_path in image_list:
            logger.info(f'Predicting {image_path}')
            img = Image.open(image_path).convert('RGB')
            fname, ext = os.path.splitext(os.path.basename(image_path))
            img_shape = img.size
            img = resize_img(img)
            image = F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # image = transform(img)
            image = image.unsqueeze(0)
            output = model(image.to(device))
            if isinstance(output, dict):
                output = output['out']
            predict_arr = np.uint8(output.argmax(1).cpu().numpy()).squeeze()

            if post_process is not None and callable(post_process):
                predict_arr = post_process(predict_arr)
            # print(predict_arr.shape, img.size)
            pred_viz = imgviz.label2rgb(label=predict_arr, image=np.array(img), font_size=15,
                                        loc='rb', label_names=class_names)

            combine_viz = np.concatenate([img, pred_viz], axis=1)
            imgviz.io.imsave(os.path.join(save_dir, os.path.basename(image_path)), combine_viz)
            # 保存Masks文件
            colors = np.array(get_color_map_list(len(class_names))).reshape(-1).tolist()
            pred_img = Image.fromarray(predict_arr).convert('P')
            pred_img = pred_img.resize(img_shape)
            pred_img.putpalette(colors)
            pred_img.save(os.path.join(save_dir, 'masks', f"{fname}.png"))
            if with_prob:
                prob_arr = output.cpu().numpy().squeeze()
                np.save(os.path.join(save_dir, 'prob', fname), prob_arr)
            results.append(os.path.join(save_dir, os.path.basename(image_path)))
    return results


def init(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    class_names = [l.strip() for l in open(os.path.join(config_path, 'labels.txt')).readlines()]
    model_path = os.path.join(config_path, 'BEST-training-params.pth')
    assert 'model_name' in config and 'num_classes' in config
    num_classes = len(class_names)
    model = create_model(f"segmentation.{config['model_name']}", num_classes=len(class_names),
                         aux_loss=config['aux_loss'], pretrained=False)
    # device_info = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, class_names, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Evaluating.')

    parser.add_argument('--data', default=None, nargs='*', required=True, help='dataset')
    parser.add_argument('--model_root', default=None, required=True, help='Model root')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--save_dir', default='.', help='path where to save')

    args = parser.parse_args()
    model, class_names, device = init(args.model_root)
    inference(args.data, model, device=device, class_names=class_names, save_dir=args.save_dir)

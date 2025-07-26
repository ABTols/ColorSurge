import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.colorsurge_arch import ColorSurge
from basicsr.utils import images_to_video, video2grayscale_frames_and_txt
import torch.nn.functional as F
from basicsr.data.colorsurge_dataset import ColorSurge_Dataset
from torch.utils.data import DataLoader


class VideoColorizationPipeline(object):
    def __init__(self, model_path, input_size=[256,256], model_type='Tiny'):
        
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_type=='Tiny':
            self.decoder_type = "TinyColorDecoder"
        else:
            self.decoder_type = "LargeColorDecoder"

        self.model = ColorSurge(
            encoder_name='convnextv2-l',
            decoder_name=self.decoder_type,
            input_size=input_size,
            last_norm='Spectral',
            num_queries=100,
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img_list, frame_len, input_size=[256,256]):
        orig_ls = []
        tensor_gray_rgbs = []
        output_imgs = []
        for i in range(len(img_list)):
            img = cv2.imread(img_list[i][0])
            self.height, self.width = img.shape[:2]
            if self.width * self.height < 100000:
                self.input_size = 512

            img = (img / 255.0).astype(np.float32)
            orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)
            orig_ls.append(orig_l)

            # resize rgb image -> lab -> get grey -> rgb
            img = cv2.resize(img, (input_size[1], input_size[0]))
            img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
            img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
            img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

            tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
            tensor_gray_rgbs.append(tensor_gray_rgb)
            
        tensor_gray_rgbs = torch.cat(tensor_gray_rgbs, dim=0)
        
        output_ab = self.model(tensor_gray_rgbs, frame_len).cpu()  # (1, 2, self.height, self.width)
        
        for i in range(output_ab.shape[0]):
            output_ab_resize = F.interpolate(output_ab[i:i+1], size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
            output_lab = np.concatenate((orig_ls[i], output_ab_resize), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
            output_img = (output_bgr * 255.0).round().astype(np.uint8)    
            output_imgs.append(output_img)
            
        return output_imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/net_g_60000.pth')
    parser.add_argument('--input_video', type=str, default='./inputs/raw_segment_14_gray.mp4', help='input test image folder or video path')
    parser.add_argument('--output_video', type=str, default='./results/color_video', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=[256,256], help='input size for model')
    parser.add_argument('--model_type', type=str, choices=['Tiny', 'Large'], default='Tiny', help='Choose model type: Tiny or Large')
    args = parser.parse_args()

    colorizer = VideoColorizationPipeline(model_path=args.model_path, input_size=args.input_size)
    
    gray_frames_dir = os.path.join("./results/gray_frames",os.path.splitext(os.path.basename(args.input_video))[0])
    gray_frame_txt_path = os.path.join("./results/input_frames_txt",os.path.splitext(os.path.basename(args.input_video))[0]+".txt")
    os.makedirs(gray_frames_dir, exist_ok=True)
    os.makedirs("./results/input_frames_txt", exist_ok=True)
    
    fps = video2grayscale_frames_and_txt(
        args.input_video,
        gray_frames_dir,
        gray_frame_txt_path
    )
    dataset = ColorSurge_Dataset(meta_info_file=gray_frame_txt_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(dataloader):
        img_list = batch['gt_path']
        frame_len = len(img_list)
        image_out = colorizer.process(img_list, frame_len, input_size=args.input_size)
        for i in range(len(image_out)):
            save_frames_dir = os.path.join("./results/color_imgs", img_list[i][0].split("/")[-2])
            os.makedirs(save_frames_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_frames_dir,img_list[i][0].split("/")[-1]), image_out[i])
    
    images_to_video(img_folder=save_frames_dir, output_path=args.output_video, img_ext='jpg', fps=fps)
    print("down!")
   

if __name__ == '__main__':
    main()

import os
import numpy as np
import encode_images_fk as ei
import config
import generate_figures_fk as gf
import align_images_fk
from dnnlib import tflib



def mixing_pre():
    args, other_args = config.parser.parse_known_args()
    # 路径设置
    # args.raw_dir = 'raw_images'
    args.raw_dir = 'results_f'
    args.aligned_dir = 'aligned_images_f'
    args.src_dir = 'aligned_images_f'
    args.generated_images_dir = 'generated_images'
    args.dlatent_dir = "latent_representations"

    # 执行图像中人脸的检测与图像的剪裁
    align_images_fk.align(args, other_args)
    #     执行图像的StyleEncoder的编码过程
    ei.styleGAN_encoder(args)

# def mixing_image(image_name, pic_A, pic_B):
#     args, other_args = config.parser.parse_known_args()
#     src_dlatents =  np.load(args.dlatent_dir+'/'+pic_A+'.npy').reshape(1,18,512) # [seed, layer, component]
#     dst_dlatents =  np.load(args.dlatent_dir+'/'+pic_B+'.npy').reshape(1,18,512)# [seed, layer, component]
#
#     gf.draw_style_mixing_figure(
#         os.path.join(config.result_dir, image_name+'.png'), gf.load_Gs(gf.url_ffhq), w=1024,
#         h=1024, style_ranges=[range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, 18)], src_dlatents=src_dlatents, dst_dlatents=dst_dlatents)

def mixing_image(path_A, path_B, ):
    args, other_args = config.parser.parse_known_args()
    # 执行图像中人脸的检测与图像的剪裁
    align_images_fk.align(args, other_args, path_A, path_B)
    #     执行图像的StyleEncoder的编码过程
    ei.styleGAN_encoder(args)

# result_path

if __name__ == '__main__':
    tflib.init_tf()

    # 希望运行多次mingxing_image 在图像未增多之前运行一次 mixing_pre 即可
    # mixing_pre()
    # pic_A 与 pic_B 是希望合成的两图的名称
    mixing_image(path_A= './raw_images/wr.jpg', path_B='./raw_images/wx.jpg' )


    # 因环境问题，不建议将generate_image 和 minxing部分联合起来运行。可分为两次运行或者单独使用
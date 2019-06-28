import os
from mmdet.apis import init_detector, inference_detector, show_result

config_file = 'configs_zhym/cascade_mask_rcnn_r101_fpn_1x_four_points.py'
checkpoint_file = 'work_dirs/cascade_mask_rcnn_r101_fpn_1x/epoch_9.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img_root_dir = '/home/zhaoyanmei/mmdetection/data/CoCoFourPoint/test/'
dst_dir = '/home/zhaoyanmei/mmdetection/TestJpgs/'
for img_file in os.listdir(img_root_dir):
    img = os.path.join(img_root_dir, img_file)
    result = inference_detector(model, img)
    show_result(img, result,[model.CLASSES], out_file=os.path.join(dst_dir, img_file))

# test a list of images and write the results to image files
#imgs = ['000000000060.jpg']
#for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

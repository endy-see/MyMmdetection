import os
import cv2
import mmcv
import numpy as np
from mmcv.image import imread, imwrite
from mmcv import color_val
from mmdet.apis import init_detector, inference_detector

config_file = 'configs_zhym/faster_rcnn_r50_fpn_1x_voc_handeonlytable.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_handeonlytable/epoch_10.pth'

#config_file = 'configs_zhym/cascade_mask_rcnn_r101_fpn_1x_four_points.py'
#checkpoint_file = 'work_dirs/cascade_mask_rcnn_r101_fpn_1x/epoch_12.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img_root_dir = '/home/zhaoyanmei/data/HANDE/HandeOnlyTable/PDF2_new_JPEGs/'
#img_root_dir = '/home/zhaoyanmei/mmdetection/data/CoCoFourPoint/test/'
dst_dir = '/home/zhaoyanmei/data/HANDE/HandeOnlyTable/visualize_PDF2/'
dst_pred_txt = dst_dir + 'pred_result.txt'
pred_txt_file = open(dst_pred_txt, 'w')

def show_result(img, result, class_names, score_thr=0.5, out_file=None):
    assert isinstance(class_names, (tuple, list))
    img_name = os.path.basename(img)
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    print('---->>> bbox_result = ', bbox_result)
    print('---->>> segm_result = ', segm_result)
    bboxes = np.vstack(bbox_result)
    print('---->>> bboxes = ', bboxes)
    # draw bounding boxes
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    print('---->>> labels1 = ', labels)
    labels = np.concatenate(labels)
    print('---->>> labels2 = ', labels)
    imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        img_name,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)

def imshow_det_bboxes(img, bboxes, labels, img_name, class_names=None, score_thr=0.7, bbox_color='green', text_color='green', thickness=1, font_scale=0.5,show=False,win_name='',wait_time=0, out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxeses = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1]-2), cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        
        bbox_str = [str(bbox[i]) for i in range(len(bbox))]
        bbox_str.insert(0, img_name)
        bbox_str.append(label_text)
        pred_str = ','.join(bbox_str)
        pred_txt_file.write(pred_str+'\n')

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


for i, img_file in enumerate(os.listdir(img_root_dir)):
    print(i)
    img = os.path.join(img_root_dir, img_file)
    if img is None:
        continue
    result = inference_detector(model, img)
    if i > 2:
        break
    show_result(img, result, model.CLASSES, out_file=os.path.join(dst_dir, img_file))

# test a list of images and write the results to image files
#imgs = ['000000000060.jpg']
#for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

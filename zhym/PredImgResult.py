import os, sys
import mmcv
import cv2
from mmcv.image import imread, imwrite
from mmcv import color_val
import argparse
import numpy as np
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Set prediction parameters:')
    parser.add_argument('--config', dest='config', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', type=str)
    parser.add_argument('--imgdir', dest='imgdir', type=str)
    parser.add_argument('--dstdir', dest='dstdir', type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def do_pred_imgs(config, checkpoint, imgdir, dstdir, score_thresh=0, is_save_pred_img=False):
    model = init_detector(config, checkpoint, device='cuda:0')
    pred_txt_file = open(os.path.join(dstdir, 'pred_result.txt'), 'w')

    for i, img_name in enumerate(os.listdir(imgdir)):
        print(i)
        img = os.path.join(imgdir, img_name)
        if img is None:
            continue
        try:
            result = inference_detector(model, img)
        except:
            pass
        
        class_names = model.CLASSES
        assert isinstance(class_names, (tuple, list))
        bboxes = np.vstack(result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
        labels = np.concatenate(labels)

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        
        if score_thresh > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thresh
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        for bbox, label in zip(bboxes, labels):
            # save pred result info to txt
            label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
            bbox_res_str = [str(bbox[i]) for i in range(len(bbox)-1)]
            bbox_res_str.insert(0, str(bbox[-1]))
            bbox_res_str.insert(0, img_name)
            bbox_res_str.append(label_text)
            pred_txt_file.write(','.join(bbox_res_str)+'\n')

            if is_save_pred_img:
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                
                img = imread(img)
                bbox_color = color_val('green')
                text_color = color_val('red')
                cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)
                label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
                if len(bbox) > 4:
                    label_text += '|{:.02f}'.format(bbox[-1])
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1]-2), cv2.FONT_HERSHEY_COMPLEX, font_scale=0.5, text_color=text_color)

                imwrite(img, os.path.join(dst_dir, img_file))



if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    do_pred_imgs(args.config, args.checkpoint, args.imgdir, args.dstdir, is_save_pred_img=False)

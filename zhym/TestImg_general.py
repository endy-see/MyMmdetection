import os
import argparse
from mmdet.apis import init_detector, inference_detector, show_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize you test img')
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-m', '--checkpoint', required=True, type=str)
    parser.add_argument('-i', '--input', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)
    parser.add_argument('-u', '--cuda', required=True, type=int)

    args = vars((parser.parse_args()))
    config_file = args['config']
    checkpoint_file = args['checkpoint']
    img_root_dir = args['input']
    dst_dir = args['output']
    cuda = args['cuda']

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:' + str(cuda))

    for i, img_file in enumerate(os.listdir(img_root_dir)):
        print(i)
        img = os.path.join(img_root_dir, img_file)
        result = inference_detector(model, img)
        show_result(img, result, [model.CLASSES], out_file=os.path.join(dst_dir, img_file))

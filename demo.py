from os.path import join as pjoin
from os import makedirs
from argparse import ArgumentParser

import cv2


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def get_image_height(img_path):
    org = cv2.imread(img_path)
    return org.shape[0]


if __name__ == '__main__':
    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    key_params = {'min-grad': 3, 'ffl-block': 5, 'min-ele-area': 8, 'merge-contained-ele': True,
                  'max-word-inline-gap': 2, 'max-line-gap': 1}

    # set input paths
    ls_imgs_dir = 'data/mask/lside'
    rs_imgs_dir = 'data/mask/rside'
    output_root = 'data/output'
    url_box_path = pjoin(output_root, 'detected_url_box.png')

    # parse the arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="input",
                        help="Input the image file to detect")
    parser.add_argument("-r", "--result", dest="output",
                        help="Result file to output")
    args = parser.parse_args()
    input_img_path = args.input
    result_text_path = args.output
    if input_img_path is None:
        print("No supported the image filename to detect.")
        exit(-1)

    is_ip = True
    is_ocr = True
    is_detected = False

    # detect the URL box
    if is_ip:
        import detect_compo.ip_region_proposal as ip

        makedirs(output_root, exist_ok=True)
        is_detected = ip.detect_url_box(input_img_path, ls_imgs_dir, rs_imgs_dir, output_root, key_params,
                                        url_box_path=url_box_path, show=True)

    # recognize the URL text by OCR
    if is_ocr and is_detected:
        import detect_text.ocr_core as ocr

        makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        ocr_text = ocr.ocr_core(url_box_path, result_text_path=result_text_path, show=True)
        print(ocr_text)

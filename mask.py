import cv2

import detect_compo.lib_ip.ip_preprocessing as pre_proc

if __name__ == '__main__':

    key_params = {'min-grad': 3, 'ffl-block': 5, 'min-ele-area': 8, 'merge-contained-ele': True,
                  'max-word-inline-gap': 2, 'max-line-gap': 2}

    # set input/output image path
    input_img_path = 'data/mask/chrome_prefix.png'
    output_img_path = 'data/mask/chrome_prefix_binary.png'

    # *** Step 1 *** pre-processing: read image and get binary map from it
    org, grey = pre_proc.read_img(input_img_path)
    binary = pre_proc.get_binary_map(org, grad_min=int(key_params['min-grad']), show=True, wait_key=0)
    cv2.imwrite(output_img_path, binary)

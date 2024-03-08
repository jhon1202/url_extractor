from os import listdir
from os.path import isfile
from os.path import join as pjoin

import cv2
import numpy as np
from config.CONFIG_UIED import Config

C = Config()


def read_img(path, resize_height=None, kernel_size=None):
    def resize_by_height(org):
        w_h_ratio = org.shape[1] / org.shape[0]
        resize_w = resize_height * w_h_ratio
        re = cv2.resize(org, (int(resize_w), int(resize_height)))
        return re

    try:
        img = cv2.imread(path)
        if kernel_size is not None:
            img = cv2.medianBlur(img, kernel_size)
        if img is None:
            print("*** Image does not exist ***")
            return None, None
        if resize_height is not None:
            img = resize_by_height(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    except Exception as e:
        print(e)
        print("*** Img Reading Failed ***\n")
        return None, None


def gray_to_gradient(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f = np.copy(img)
    img_f = img_f.astype("float")

    kernel_h = np.array([[0, 0, 0], [0, -1., 1.], [0, 0, 0]])
    kernel_v = np.array([[0, 0, 0], [0, -1., 0], [0, 1., 0]])
    dst1 = abs(cv2.filter2D(img_f, -1, kernel_h))
    dst2 = abs(cv2.filter2D(img_f, -1, kernel_v))
    gradient = (dst1 + dst2).astype('uint8')
    return gradient


def grad_to_binary(grad, min_val):
    rec, binary = cv2.threshold(grad, min_val, 255, cv2.THRESH_BINARY)
    return binary


def reverse_binary(binary, show=False):
    """
    Reverse the input binary image
    """
    r, binary = cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY_INV)
    if show:
        cv2.imshow('binary_rev', binary)
        cv2.waitKey()
    return binary


def get_binary_map(org, grad_min, show=False, write_path=None, wait_key=0):
    grey = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    grad = gray_to_gradient(grey)  # get RoI with high gradient
    binary = grad_to_binary(grad, grad_min)  # enhance the RoI
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (3, 3))  # remove noises
    if write_path is not None:
        cv2.imwrite(write_path, morph)
    if show:
        cv2.imshow('binary', morph)
        if wait_key is not None:
            cv2.waitKey(wait_key)
    return morph


def find_best_matched_region(binary, mask_dir, grad_min):
    # get the list of origin/gray/binary maps
    mask_files = [f for f in listdir(mask_dir) if isfile(pjoin(mask_dir, f))]
    mask_orgs = []
    mask_grays = []
    mask_binaries = []
    for f in mask_files:
        mask_path = pjoin(mask_dir, f)
        mask_org, mask_gray = read_img(mask_path)
        mask_binary = get_binary_map(mask_org, grad_min)
        mask_orgs.append(mask_org)
        mask_grays.append(mask_gray)
        mask_binaries.append(mask_binary)

    # find the best matched region
    best_val = 0.0
    best_loc = None
    width = height = 0
    for mask_binary in mask_binaries:
        w, h = mask_binary.shape[::-1]
        res = cv2.matchTemplate(binary, mask_binary, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            width = w
            height = h

    return best_val, best_loc, width, height


def clip_fair_region(img_path, ls_dir, rs_dir, grad_min, show=False, write_path=None, wait_key=0):
    # get the origin/gray/binary map for image
    org, grey = read_img(img_path)
    if show:
        cv2.imshow('binary', org)
        if wait_key is not None:
            cv2.waitKey(wait_key)

    binary = get_binary_map(org, grad_min, show=False, wait_key=wait_key)

    # find the best fair region
    fair_binary = None
    fair_grey = None
    fair_org = None
    # get the fair region for left-side
    best_val, best_loc, width, height = find_best_matched_region(binary, ls_dir, grad_min)
    if best_val >= C.THRESHOLD_MASK_MATCHING:
        fair_binary = binary[best_loc[1]-16:best_loc[1]+height+16, 0::]
        fair_grey = grey[best_loc[1]-16:best_loc[1]+height+16, 0::]
        fair_org = org[best_loc[1]-16:best_loc[1]+height+16, 0::]
    else:
        # get the fair region for right-side
        best_val, best_loc, width, height = find_best_matched_region(binary, rs_dir, grad_min)
        if best_val >= C.THRESHOLD_MASK_MATCHING:
            fair_binary = binary[best_loc[1]-16:best_loc[1]+height+16, 0::]
            fair_grey = grey[best_loc[1]-16:best_loc[1]+height+16, 0::]
            fair_org = org[best_loc[1]-16:best_loc[1]+height+16, 0::]

    if fair_binary is not None:
        if show:
            cv2.imshow('Clipped region', fair_org)
            if wait_key is not None:
                cv2.waitKey(wait_key)
        if write_path is not None:
            cv2.imwrite(write_path, fair_org)
    else:
        if show:
            cv2.imshow('Not detected region', binary)
            if wait_key is not None:
                cv2.waitKey(wait_key)

    return fair_org, fair_grey, fair_binary

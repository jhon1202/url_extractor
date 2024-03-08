import time

import cv2

import detect_compo.lib_ip.Component as Compo
import detect_compo.lib_ip.block_division as blk
import detect_compo.lib_ip.ip_detection as detector
import detect_compo.lib_ip.ip_preprocessing as pre_proc
from config.CONFIG_UIED import Config

C = Config()


# def processing_block(org, binary, blocks, block_pad):
#     image_shape = org.shape
#     uicompos_all = []
#     for block in blocks:
#         # *** Step 2.1 *** check: examine if the block is valid layout block
#         if block.block_is_top_or_bottom_bar(image_shape, C.THRESHOLD_TOP_BOTTOM_BAR):
#             continue
#         if block.block_is_uicompo(image_shape, C.THRESHOLD_COMPO_MAX_SCALE):
#             uicompos_all.append(block)
#
#         # *** Step 2.2 *** binary map processing: erase children block -> clipping -> remove lines(opt)
#         binary_copy = binary.copy()
#         for i in block.children:
#             blocks[i].block_erase_from_bin(binary_copy, block_pad)
#         block_clip_bin = block.compo_clipping(binary_copy)
#         # det.line_removal(block_clip_bin, show=True)
#
#         # *** Step 2.3 *** component extraction: detect components in block binmap -> convert position to relative
#         uicompos = det.component_detection(block_clip_bin)
#         Compo.cvt_compos_relative_pos(uicompos, block.bbox.col_min, block.bbox.row_min)
#         uicompos_all += uicompos
#     return uicompos_all


def nesting_inspection(org, grey, compos, ffl_block):
    """
    Inspect all big compos through block division by flood-fill
    :param org:
    :param grey:
    :param compos:
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    """
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:
            replace = False
            compo.compo_clipping(org)
            clip_grey = compo.compo_clipping(grey)
            n_compos = blk.block_division(clip_grey, org, grad_thresh=ffl_block, show=False)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def detect_url_box(img_path, ls_dir, rs_dir, output_root, uied_params, url_box_path=None, show=False, wai_key=0):
    start = time.process_time()

    # *** Step 1 *** pre-processing: find the fair region and clip it
    org, grey, binary = pre_proc.clip_fair_region(img_path, ls_dir, rs_dir, grad_min=int(uied_params['min-grad']),
                                                  show=show, wait_key=wai_key)
    if org is None:
        print("*** Not detected address bar region ***\n")
        return False

    # *** Step 2 *** detect the UI elements
    detector.rm_lines(binary, show=False, wait_key=wai_key)
    ui_compos = detector.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']))
    # draw.draw_bounding_box(org, ui_compos, show=show, name='components', wait_key=wai_key)

    # *** Step 3 *** refine the results
    ui_compos = detector.merge_intersected_corner(ui_compos, org,
                                                  is_merge_contained_ele=uied_params['merge-contained-ele'],
                                                  max_gap=(3, 1), max_ele_height=25)
    Compo.compos_update(ui_compos, org.shape)
    Compo.compos_containment(ui_compos)

    # *** Step 4 *** nesting inspection: treat the big compos as block and check if they have nesting element
    ui_compos += nesting_inspection(org, grey, ui_compos, ffl_block=uied_params['ffl-block'])
    ui_compos = detector.compo_filter(ui_compos, min_area=int(uied_params['min-ele-area']))
    Compo.compos_update(ui_compos, org.shape)

    # *** Step 5 *** find the URL component
    min_height = C.THRESHOLD_MIN_URL_BOX_HEIGHT
    ui_compos = detector.filter_url_box(ui_compos, min_height)
    if len(ui_compos) == 0:
        print("*** Not detected URL box ***\n")
        return False

    url_compo = ui_compos[0]
    url_box = org[url_compo.bbox.row_min:url_compo.bbox.row_max, url_compo.bbox.col_min:url_compo.bbox.col_max]
    cv2.imwrite(url_box_path, url_box)

    if show:
        cv2.imshow('Detected URL box', url_box)
        if wai_key is not None:
            cv2.waitKey(wai_key)

    print("[Compo Detection Completed in %.3f s] %s" % (time.process_time() - start, img_path))

    return True

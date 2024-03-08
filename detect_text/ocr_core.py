import time

import cv2
# from PIL import Image
import pytesseract


def ocr_core(image_path, result_text_path=None, show=False, wai_key=0):
    """
    This function will handle the core OCR processing of images.
    """
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\KimCJ\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    start = time.process_time()

    org = cv2.imread(image_path)
    gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    custom_config = r'--oem 3 --psm 7'
    details = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng')
    # print(details.keys())

    total_boxes = len(details['text'])
    for num in range(total_boxes):
        if float(details['conf'][num]) > 30.0:
            (x, y, w, h) = (details['left'][num], details['top'][num], details['width'][num], details['height'][num])
            binary = cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show:
        cv2.imshow('Captured text', binary)
        if wai_key is not None:
            cv2.waitKey(wai_key)

    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        if word != '':
            word_list.append(word)
            last_word = word

        if (last_word != '' and word == '') or (word==details['text'][-1]):
            parse_text.append(word_list)
            word_list = []

    if result_text_path is not None:
        import csv
        with open(result_text_path, 'w', newline="") as file:
            csv.writer(file, delimiter=" ").writerows(parse_text)

    print("[OCR Completed in %.3f s]" % (time.process_time() - start))

    return parse_text  # Then we will print the text in the image

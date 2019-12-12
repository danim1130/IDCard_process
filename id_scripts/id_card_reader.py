import re
from typing import Dict

import cv2
import numpy as np
from pylibdmtx import pylibdmtx
from pyzbar import pyzbar

import id_scripts.pytesseract as pytesseract
from id_scripts.id_card_models import *


name_regex = re.compile('[^a-zA-ZáÁéÉíÍóÓöÖőŐüÜűŰúÚ.\- ]')
city_regex = re.compile('[^a-zA-Z0-9áÁéÉíÍóÓöÖőŐüÜúÚ\- ]')
date_regex = re.compile('[^0-9.]')




def __run_tesseract_multiple_images(images,
                                    extension_configs,
                                    lang,
                                    run_otsu=False,
                                    blur_image=False,
                                    cluster_image_num=0):
    for i in range(0, len(images)):
        image = images[i]
        if blur_image:
            image = cv2.GaussianBlur(image, (5, 5), 0.7)
        if run_otsu:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
            image = cv2.copyMakeBorder(image, 25, 25, 15, 15, cv2.BORDER_CONSTANT, value=255)
        elif cluster_image_num != 0:
            z = image.reshape((-1, 3))
            z = np.float32(z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = cluster_image_num
            ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            min_index = 0
            for j in range(1, k):
                if sum(center[j]) < sum(center[min_index]):
                    min_index = j

            for j in range(0, k):
                if min_index == j:
                    center[j][:] = 0
                else:
                    center[j][:] = 255

            res = center[label.flatten()]
            image = res.reshape(image.shape)
            image = cv2.copyMakeBorder(image, 25, 25, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        images[i] = image

    read_str = pytesseract.run_multiple_and_get_output(images,
                                                       extension='tsv',
                                                       extension_configs=extension_configs,
                                                       config="--psm 7", lang=lang)
    #    for idx, image in enumerate(images):
    #        cv2.imshow("TEST", image)
    #        cv2.waitKey(0)
    #        cv2.imwrite("test%d.png" % idx, image, (cv2.IMWRITE_PNG_COMPRESSION, 0))

    read_values = []
    lines = read_str.split("\n")
    for line in lines:
        read_values.append(line.split("\t"))

    page_index = read_values[0].index("page_num")
    base_row = read_values.pop(0)
    ret_list = [[]]
    ret_list[0].append(base_row)
    prev_page = '1'
    while len(read_values) != 0:
        next_row = read_values.pop(0)
        if next_row[page_index] != prev_page:
            ret_list.append([])
            ret_list[-1].append(base_row)
            prev_page = next_row[page_index]

        ret_list[-1].append(next_row)

    return ret_list


def __image_digits(read_values):
    confidence_level = 0
    text_index = read_values[0].index("text")
    confidence_index = read_values[0].index("conf")
    date_parts = []
    for i in range(1, len(read_values)):
        if len(read_values[i]) > text_index:
            date_parts_candidate = re.findall(r'\d+', read_values[i][text_index])
            #            name_part_candidate = date_regex.sub('', read_values[i][text_index]).lstrip().rstrip()
            #            date_parts_candidate = name_part_candidate.split(".")
            if 3 <= len(date_parts_candidate) <= 4 \
                    and len(date_parts_candidate[0]) == 4 and date_parts_candidate[0].isdigit() \
                    and len(date_parts_candidate[1]) <= 2 and date_parts_candidate[1].isdigit() \
                    and len(date_parts_candidate[2]) <= 2 and date_parts_candidate[2].isdigit():
                date_parts = date_parts_candidate
                confidence_level = int(read_values[i][confidence_index])
                break

    if len(date_parts) == 0:
        return ConfidenceValue(value="", confidence=0)

    year = int(date_parts[0])
    month = int(date_parts[1])
    day = int(date_parts[2])
    date = str(year) + "." + str(month).zfill(2) + "." + str(day).zfill(2)
    return ConfidenceValue(value=date, confidence=confidence_level)


def __filter_read_text(read_values):
    name = ""
    confidence_levels = []
    text_index = read_values[0].index("text")
    confidence_index = read_values[0].index("conf")
    for i in range(1, len(read_values)):
        if len(read_values[i]) > text_index:
            name_part_candidate = name_regex.sub('', read_values[i][text_index]).lstrip().rstrip()
            if len(name_part_candidate) != 0:
                confidence_levels.append(read_values[i][confidence_index])
                name = name + name_part_candidate + " "

    return name, confidence_levels


def __image_name(read_values) -> (ConfidenceValue, bool):
    name, confidence_levels = __filter_read_text(read_values)

    if len(name) == 0:
        return ConfidenceValue(value="", confidence=0), False

    name = name.upper().rstrip().lstrip()

    name_parts = name.split(' ')
    for part in name_parts.copy()[::-1]:
        if len(part) == 0 or part[0] == '-' or part[0] == '.':
            confidence_levels.pop(name_parts.index(part))
            name_parts.remove(part)

    if len(name_parts) > 1:
        return ConfidenceValue(
            value=' '.join(name_parts).title(),
            confidence=min(confidence_levels[0:len(name_parts) - 1])), True
    elif len(name_parts) == 0:
        return ConfidenceValue(value="", confidence=0), False
    else:
        return ConfidenceValue(value=' '.join(name_parts).title(), confidence=min(confidence_levels)), False


with open('data/names.txt', 'r', encoding="utf-8") as f:
    names = frozenset(f.read().splitlines())

with open('data/cities.txt', 'r', encoding="utf-8") as f:
    cities = frozenset(f.read().splitlines())


def __image_city(read_values):
    name, confidence_levels = __filter_read_text(read_values)

    if len(name) == 0:
        return ConfidenceValue(value="", confidence=0)

    city = name.upper().rstrip().lstrip()

    city_parts = city.split(' ')
    for part in city_parts.copy()[::-1]:
        if len(part) == 0 or part[0] == '-':
            confidence_levels.pop(city_parts.index(part))
            city_parts.remove(part)

    best_candidate = ''
    numbers_found = ''
    confidence_level = 100

    for part in city_parts[::-1]:
        if part in cities:
            best_candidate = part
            confidence_level = confidence_levels[city_parts.index(best_candidate)]
            break
        elif len(part) >= 2 and part[0:2].isdigit() and 0 < int(part[0:2]) <= 23:
            numbers_found = part[0:2]
        else:
            confidence_level = min(confidence_level, int(confidence_levels[city_parts.index(part)]))
            best_candidate = part + " " + best_candidate

    if len(numbers_found) != 0:
        best_candidate = best_candidate + " " + numbers_found

    return ConfidenceValue(value=best_candidate.title(), confidence=confidence_level)


def __check_field_match(input_field_value, field):
    return ConfidenceValue(value=input_field_value == field.value, confidence=int(field.confidence))


def __read_barcode_image(img):
    for i in range(1, 5):
        barcode_x_scale = 1 / i
        barcode_image = cv2.resize(img, (0, 0), fx=barcode_x_scale, fy=1)

        info = pyzbar.decode(barcode_image)
        if len(info) == 0:
            continue

        for barcode in info:
            if barcode.type == "CODE128" or barcode.type == "I25":
                return barcode.data.decode("UTF-8")


def __get_datamatrix_data(img):
    info = []
    for i in range(1, 5):
        barcode_x_scale = 1 / i
        barcode_image = cv2.resize(img, (0, 0), fx=barcode_x_scale, fy=1)

        info = pylibdmtx.decode(barcode_image)
        if len(info) == 0:
            continue
        else:
            break

    for barcode in info:
        return barcode.data.decode("UTF-8")


def __get_image_part(img, part: Rectangle):
    return img[part.top:(part.bottom + 1), part.left:(part.right + 1)]


def __run_image_field_check(img,
                            validating_fields: List[ValidationField],
                            field_type_map: Dict[str, IDCardFieldTypeEnum],
                            real_field_coordinates: Dict[str, Rectangle],
                            run_otsu,
                            use_blur,
                            cluster_image_num,
                            lang) -> List[ValidationResult]:
    success_list = []
    image_parts = []

    for field in validating_fields:
        field_type = field_type_map[field.key]
        if field_type != IDCardFieldTypeEnum.BARCODE and field_type != IDCardFieldTypeEnum.DATAGRAM:
            image_parts.append(__get_image_part(img, real_field_coordinates[field.key]))

    tesseract_output = []
    if len(image_parts) != 0:
        tesseract_output = __run_tesseract_multiple_images(image_parts,
                                                           extension_configs=[],
                                                           lang=lang,
                                                           run_otsu=run_otsu,
                                                           blur_image=use_blur,
                                                           cluster_image_num=cluster_image_num)

    i = 0
    for field in validating_fields:
        field_type = field_type_map[field.key]
        read_value = ""
        if field_type == IDCardFieldTypeEnum.TEXT:
            read_value = __filter_read_text(tesseract_output[i])[0]
            i += 1
        elif field_type == IDCardFieldTypeEnum.TEXT_CITY:
            read_value = __image_city(tesseract_output[i]).value
            i += 1
        elif field_type == IDCardFieldTypeEnum.TEXT_NAME:
            read_value = __image_name(tesseract_output[i])[0].value
            i += 1
        elif field_type == IDCardFieldTypeEnum.TEXT_DATE:
            read_value = __image_digits(tesseract_output[i]).value
            i += 1
        elif field_type == IDCardFieldTypeEnum.BARCODE:
            read_value = __read_barcode_image(__get_image_part(img, real_field_coordinates[field.key]))
        elif field_type == IDCardFieldTypeEnum.DATAGRAM:
            read_value = __get_datamatrix_data(__get_image_part(img, real_field_coordinates[field.key]))

        if read_value == field.value:
            success_list.append(ValidationResult(key=field.key, is_correct=True, confidence=100))

    return success_list


def validate_fields(img,
                    fields: List[ValidationField],
                    id_card_config: IDCardConfiguration) -> List[ValidationResult]:
    available_fields = [x.key for x in id_card_config.fields]
    remaining_validating_fields = [x for x in fields if x.key in available_fields]

    field_type_map = {x.key: x.type for x in id_card_config.fields}
    field_coordinates = {x.key: x.rectangle for x in id_card_config.fields}

    field_results = []

    if len(remaining_validating_fields) != 0:  # 79
        field_results += __run_image_field_check(img,
                                                 validating_fields=remaining_validating_fields,
                                                 field_type_map=field_type_map,
                                                 real_field_coordinates=field_coordinates,
                                                 run_otsu=False, use_blur=True, cluster_image_num=5, lang=id_card_config.language)
        remaining_validating_fields = [x
                                       for x
                                       in remaining_validating_fields
                                       if x.key not in [y.key for y in field_results]]

    if len(remaining_validating_fields) != 0:  # 79
        field_results += __run_image_field_check(img,
                                                 validating_fields=remaining_validating_fields,
                                                 field_type_map=field_type_map,
                                                 real_field_coordinates=field_coordinates,
                                                 run_otsu=True, use_blur=True, cluster_image_num=0, lang=id_card_config.language)
        remaining_validating_fields = [x for x in remaining_validating_fields if
                                       x.key not in [y.key for y in field_results]]
    if len(remaining_validating_fields) != 0:  # 77
        field_results += __run_image_field_check(img,
                                                 validating_fields=remaining_validating_fields,
                                                 field_type_map=field_type_map,
                                                 real_field_coordinates=field_coordinates,
                                                 run_otsu=False, use_blur=True, cluster_image_num=0, lang=id_card_config.language)
        remaining_validating_fields = [x for x in remaining_validating_fields if
                                       x.key not in [y.key for y in field_results]]
    if len(remaining_validating_fields) != 0:
        field_results += __run_image_field_check(img,
                                                 validating_fields=remaining_validating_fields,
                                                 field_type_map=field_type_map,
                                                 real_field_coordinates=field_coordinates,
                                                 run_otsu=False, use_blur=False, cluster_image_num=0, lang=id_card_config.language)
        remaining_validating_fields = [x for x in remaining_validating_fields if
                                       x.key not in [y.key for y in field_results]]

    if len(remaining_validating_fields) != 0:  # 75
        field_results += __run_image_field_check(img,
                                                 validating_fields=remaining_validating_fields,
                                                 field_type_map=field_type_map,
                                                 real_field_coordinates=field_coordinates,
                                                 run_otsu=True, use_blur=False, cluster_image_num=0, lang=id_card_config.language)
        remaining_validating_fields = [x for x in remaining_validating_fields if
                                       x.key not in [y.key for y in field_results]]

    field_results += [ValidationResult(x.key, False, 100) for x in remaining_validating_fields]

    return field_results

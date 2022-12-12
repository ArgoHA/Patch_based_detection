from typing import List

def xywh_to_xyxy(
    lines: List[str],
    img_height: int,
    img_width: int) -> List[List[int]]:

    '''
    This function gets list with YOLO labels in a format:
    label, x-center, y-center, bbox width, bbox height
    coordinates are in relative scale (0-1).
    Returns list of lists with xyxy format and absolute scale.
    '''
    labels = []
    for _, cur_line in enumerate(lines):
        cur_line = cur_line.split(' ')
        cur_line[-1] = cur_line[-1].split('\n')[0]

        # convert from relative to absolute scale (0-1 to real pixel numbers)
        x, y, w, h = list(map(float, cur_line[1:]))
        x = int(x * img_width)
        y = int(y * img_height)
        w = int(w * img_width)
        h = int(h * img_height)

        # convert to xyxy
        left, top, right, bottom = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        labels.append([int(cur_line[0]), left, top, right, bottom])

    return labels


def xyxy_to_xywh(
    label: List[int],
    img_width: int,
    img_height: int) -> List[float]:

    '''
    This function gets list with label and coordinates in a format:
    label, x1, y1, x2, y2
    coordinates are in absolute scale.
    Returns list with xywh format and relative scale
    '''

    x1, y1, x2, y2 = list(map(float, label[1:]))
    w = x2 - x1
    h = y2 - y1

    x_cen = round((x1 + w / 2) / img_width, 6)
    y_cen = round((y1 + h / 2) / img_height, 6)
    w = round(w / img_width, 6)
    h = round(h / img_height, 6)

    return [label[0], x_cen, y_cen, w, h]

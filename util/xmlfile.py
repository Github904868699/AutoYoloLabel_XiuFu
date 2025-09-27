import os
from pathlib import Path
from typing import Dict, Iterable, List

import xml.etree.ElementTree as ET


# 排版
def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


YOLO_CLASS_FILE = "classes.txt"


def _normalise_label_name(name: str) -> str:
    """Sanitise label text before writing it to the class list."""

    return name.strip()


def _load_existing_classes(path: Path) -> List[str]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _ensure_class_ids(save_dir: Path, label_names: Iterable[str]) -> Dict[str, int]:
    """Ensure ``classes.txt`` exists and returns the mapping from name to index."""

    classes_path = save_dir / YOLO_CLASS_FILE
    classes = _load_existing_classes(classes_path)
    updated = False

    for raw_name in label_names:
        normalised = _normalise_label_name(raw_name)
        if not normalised:
            continue
        if normalised not in classes:
            classes.append(normalised)
            updated = True

    if updated or not classes_path.exists():
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        with classes_path.open("w", encoding="utf-8") as handle:
            if classes:
                handle.write("\n".join(classes) + "\n")

    return {name: idx for idx, name in enumerate(classes)}


def _write_yolo_annotation(base_path: Path, size, labels):
    """Write YOLOv8 compatible annotations next to the annotation stem."""

    image_w, image_h = size[0], size[1]
    if not image_w or not image_h:
        return

    save_dir = base_path.parent
    class_map = _ensure_class_ids(save_dir, (label["name"] for label in labels))
    txt_path = base_path.with_suffix(".txt")
    yolo_lines: List[str] = []

    for label in labels:
        class_name = _normalise_label_name(label["name"])
        if not class_name:
            continue
        class_id = class_map.get(class_name)
        if class_id is None:
            # The class list was empty (e.g. stripped names). Skip to avoid invalid files.
            continue

        x_min, y_min, width_or_xmax, height_or_ymax = label["bndbox"][:4]
        width = width_or_xmax
        height = height_or_ymax

        # Historical annotations stored width/height instead of xmax/ymax.
        # Guard against negative or zero dimensions by falling back to difference.
        if width <= 0 and width_or_xmax > x_min:
            width = width_or_xmax - x_min
        if height <= 0 and height_or_ymax > y_min:
            height = height_or_ymax - y_min

        if width <= 0 or height <= 0:
            continue

        x_center = (x_min + width / 2) / image_w
        y_center = (y_min + height / 2) / image_h
        norm_width = width / image_w
        norm_height = height / image_h

        # Clamp to [0, 1] to avoid out-of-range issues during training.
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        norm_width = min(max(norm_width, 0.0), 1.0)
        norm_height = min(max(norm_height, 0.0), 1.0)

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        )

    if yolo_lines:
        with txt_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(yolo_lines) + "\n")
    else:
        if txt_path.exists():
            txt_path.unlink()


def write_yolo_labels(base_path: Path, size, labels):
    """Persist annotations in YOLO format using the provided base path."""

    _write_yolo_annotation(base_path, size, labels)


def load_yolo_labels(txt_path: Path, image_w: int, image_h: int):
    """Load YOLO annotations and convert them to the internal label structure."""

    if not txt_path.exists() or not image_w or not image_h:
        return [], [], []

    class_list = _load_existing_classes(txt_path.parent / YOLO_CLASS_FILE)
    labels = []
    boxes = []
    names = []

    with txt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_w
                y_center = float(parts[2]) * image_h
                box_width = float(parts[3]) * image_w
                box_height = float(parts[4]) * image_h
            except ValueError:
                continue

            class_name = class_list[class_id] if 0 <= class_id < len(class_list) else str(class_id)

            x_min = max(int(round(x_center - box_width / 2)), 0)
            y_min = max(int(round(y_center - box_height / 2)), 0)
            width = max(int(round(box_width)), 0)
            height = max(int(round(box_height)), 0)

            label = {
                "name": class_name,
                "pose": "Unspecified",
                "truncated": 0,
                "difficult": 0,
                "bndbox": [x_min, y_min, width, height],
            }

            labels.append(label)
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            names.append(class_name)

    return labels, boxes, names


def xml(image_path, save_path, size, labels):
    root = ET.Element('annotation')
    folder_name = os.path.dirname(image_path)
    folder = ET.SubElement(root, 'folder')
    folder.text = folder_name

    file_name = os.path.basename(image_path)
    filename = ET.SubElement(root, 'filename')
    filename.text = file_name

    filepath = ET.SubElement(root, 'path')
    filepath.text = image_path

    img_size = ET.SubElement(root, 'size')
    width = ET.SubElement(img_size, 'width')
    width.text = str(size[0])
    height = ET.SubElement(img_size, 'height')
    height.text = str(size[1])
    depth = ET.SubElement(img_size, 'depth')
    depth.text = str(size[2])

    for dic in labels:
        object = ET.SubElement(root, 'object')

        lab_name = ET.SubElement(object, 'name')
        lab_name.text = dic['name']

        pose = ET.SubElement(object, 'pose')
        pose.text = dic['pose']

        truncated = ET.SubElement(object, 'truncated')
        truncated.text = str(dic['truncated'])

        difficult = ET.SubElement(object, 'difficult')
        difficult.text = str(dic['difficult'])

        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(dic['bndbox'][0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(dic['bndbox'][1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(dic['bndbox'][2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(dic['bndbox'][3])

    indent(root)  # 格式化xml
    tree = ET.ElementTree(root)
    tree.write(save_path)  # 写入文件

    return tree

def xml_message(save_path,image_name,img_width,img_height,text,x,y,w,h):
    file_path = os.path.join(save_path, f"{image_name}.xml")
    size = [img_width, img_height, 3]
    result = {
        'name': text,
        'pose': 'Unspecified',
        'truncated': 0,
        'difficult': 0,
        'bndbox': [x, y, w, h]
    }
    return result,file_path,size


if __name__ == "__main__":
    path = r'dog.4019.jpg'
    save_path = "111.xml"
    size = [640, 640, 3]
    labels = [{'name': 'body',
               'pose': 'Unspecified',
               'truncated': 0,
               'difficult': 0,
               'bndbox': [9, 89, 297, 305]},
              {'name': 'body',
               'pose': 'Unspecified',
               'truncated': 0,
               'difficult': 1,
               'bndbox': [20, 89, 297, 350]}]

    print(xml(path,save_path, size, labels))

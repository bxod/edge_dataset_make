# pip install ultralytics pillow

import os, math, xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from PIL import Image, ImageOps
from ultralytics import YOLO


MODEL_WEIGHTS   = "yolov8x.pt"
DEVICE          = "cuda"              # or "cpu"
TARGET_SIZE     = 640
BORDER_COLOR    = (0, 0, 0)           # (255,255,255) for white
DELETE_ORIGINAL = True


model = YOLO(MODEL_WEIGHTS).to(DEVICE)

def get_person_boxes(img: Image.Image) -> List[List[float]]:
    res = model.predict(source=img, classes=[0], verbose=False, conf=0.25)[0] #We are only detecting humans
    return [box.xyxy[0].tolist() for box in res.boxes]

def preprocess_image(img_path: Path) -> Path:
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)                

    w, h = img.size

    if h > w:
        diff = h - w
        pad_lr = diff // 4                    
        img = ImageOps.expand(img,            
                              border=(pad_lr, 0, pad_lr, 0),
                              fill=BORDER_COLOR)
        w += 2 * pad_lr                      
        boxes = get_person_boxes(img)
        cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes))/2 if boxes else h/2
        top = int(max(0, cy - w/2))
        top = min(top, h - w)                  
        img = img.crop((0, top, w, top + w))   
        h = w                                   

    if w != h:
        side = max(w, h)
        img = ImageOps.expand(img,
                              ((side - w)//2, (side - h)//2,
                               (side - w + 1)//2, (side - h + 1)//2),
                              fill=BORDER_COLOR)
        w = h = side

    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    out_path = img_path.with_name(f"{img_path.stem}_sq{TARGET_SIZE}{img_path.suffix}")
    img.save(out_path, quality=95)
    return out_path


def create_pascal_voc_xml(img_path, class_name, detections):
    img = Image.open(img_path)
    w, h = img.size
    d = len(img.getbands())

    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text, ET.SubElement(ann, "filename").text = class_name, os.path.basename(img_path)
    ET.SubElement(ann, "path").text = os.path.abspath(img_path)

    src = ET.SubElement(ann, "source"); ET.SubElement(src, "database").text = "Unknown"
    sz  = ET.SubElement(ann, "size")
    for t, v in (("width", w), ("height", h), ("depth", d)):
        ET.SubElement(sz, t).text = str(v)
    ET.SubElement(ann, "segmented").text = "0"

    for x1, y1, x2, y2 in detections:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text, ET.SubElement(obj, "pose").text = class_name, "Unspecified"
        ET.SubElement(obj, "truncated").text, ET.SubElement(obj, "difficult").text = "0", "0"
        bb = ET.SubElement(obj, "bndbox")
        for tag, val in zip(("xmin", "ymin", "xmax", "ymax"),
                            map(lambda v: max(0, int(v)), (x1, y1, x2, y2))):
            ET.SubElement(bb, tag).text = str(val)

    return ET.ElementTree(ann)


def process_dataset(root_dir: str | Path):
    root_dir = Path(root_dir)
    for class_dir in root_dir.iterdir():
        if not class_dir.is_dir(): continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}: continue

            processed = preprocess_image(img_path)
            results   = model.predict(source=str(processed), classes=[0], verbose=False)
            boxes     = [b.xyxy[0].tolist() for r in results for b in r.boxes]
            if not boxes:
                print(f"{processed.name}: no person → removed.")
                processed.unlink(missing_ok=True)
                if DELETE_ORIGINAL: img_path.unlink(missing_ok=True)
                continue

            xml = create_pascal_voc_xml(processed, class_dir.name, boxes)
            xml.write(processed.with_suffix(".xml"))
            print(f"{processed.name} annotated.")

            if DELETE_ORIGINAL:
                img_path.unlink(missing_ok=True)


if __name__ == "__main__":
    process_dataset("./images")

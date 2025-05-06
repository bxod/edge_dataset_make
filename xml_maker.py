# pip install ultralytics
# pip install pillow

import os
import math
import xml.etree.ElementTree as ET
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8x.pt")  # Nano (yolov8n.pt) Small (yolov8s.pt) Medium (yolov8m.pt) Large (yolov8l.pt) Extra-large (yolov8x.pt) Bigger model - better labeling
model.to('cuda') # You may comment this line out if you want to run the detector using CPU

def create_pascal_voc_xml(img_path, class_name, detections):
    img = Image.open(img_path)
    width, height = img.size
    depth = len(img.getbands())

    # XML structure
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = class_name
    ET.SubElement(annotation, "filename").text = os.path.basename(img_path)
    ET.SubElement(annotation, "path").text = os.path.abspath(img_path)
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = "0"

    for det in detections:
        x1, y1, x2, y2 = det
        x1 = max(0, math.floor(x1))
        y1 = max(0, math.floor(y1))
        x2 = min(width, math.floor(x2))
        y2 = min(height, math.floor(y2))
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)
    return ET.ElementTree(annotation)

def process_dataset(root_dir):
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                continue
            results = model.predict(source=file_path, classes=[0], verbose=False)
            person_boxes = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    person_boxes.append(xyxy)
            if not person_boxes:
                print(f"Warning: No person detected in {file_path}, skipping annotation.")
                continue
            xml_tree = create_pascal_voc_xml(file_path, class_name, person_boxes)
            xml_path = os.path.splitext(file_path)[0] + ".xml"
            xml_tree.write(xml_path)
            print(f"Saved annotation for {file_path} -> {xml_path}")

if __name__ == "__main__":
    dataset_path = "./dataset_root"
    process_dataset(dataset_path)

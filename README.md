## Project Overview

This repository guides you through preparing a custom dataset in PASCAL VOC format for the EDGE Computing course project using an auto-labeling technique to avoid manually labeling every image.

Our goal is to build a dataset that detects:

* **no\_helmet**: a kick scooter rider without a helmet
* **with\_helmet**: a kick scooter rider wearing a helmet
* **two\_person**: two persons riding the same kick scooter

The workflow covers:

1. **Collecting and formatting images**
2. **Shuffling and renaming**
3. **Auto-labeling**

---

## 1. Dataset Preparation

### 1.1 Image Collection

* Use the **Open Camera** Android app to capture images at **720×960** resolution in JPG format.
* Captured **3,000 images** across the three classes, ensuring varied backgrounds and clothing for model robustness.
* Downloaded an additional **400 images** from the internet using `img_downloader.py`.

### 1.2 Folder Organization

* Placed all images in three separate folders under the dataset root:

  ```
  dataset_root/
  ├── no_helmet/    (1,034 images)
  ├── with_helmet/  (1,615 images)
  └── two_person/   (762 images)
  ```
* Ensure all images are in **.jpg** format and within the **512×512 to 1080×1080** pixel range to match the SSDLite320–640 input requirements.

---

## 2. Shuffling and Renaming

Use `shuffler.py` to randomly shuffle images within each class folder and rename them as `{class_name}_{order}.jpg`.

Run:

```bash
python shuffler.py
# Will rename images inside the given folder. E.g. with_helmet_1.jpg, with_helmet_2.jpg, ...
```

---

## 3. Auto-Labeling

## 3.1 Install the following packages: 
```bash
pip install ultralytics
pip install pillow
```

Make sure you have CUDA available if you are going to use GPU for auto detecting objects
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3.1 Create XML files
Now run `xml_maker.py` to create XML labels for each class folder. Make sure you pass correct dataset root that contains the class folders with images e.g. `./dataset_root`
```bash
python xml_maker.py
```

You will now have XML label files created next to each image file. You can now split the dataset into train, test, validation subfolders according to PASCAL VOC format.

---

## Note 

It is important to collect dataset of at least 4 classes to avoid issues with the project code. If you are having 3 or fewer classes, add a dummy class of one image along with its XML file to the final dataset. If you are planning to have 4 or more classes, you can ignore this note. Good luck.

---

## Scripts

* **img\_downloader.py**: Bulk download images from the internet.
* **shuffler.py**: Shuffle and rename images.
* **xml\_maker.py**: Run the auto-labeling model and generate VOC XML files.

---

## Important: Leave a star! ⭐

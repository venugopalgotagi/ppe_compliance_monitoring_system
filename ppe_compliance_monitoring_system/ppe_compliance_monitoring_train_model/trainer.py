import os
from dotenv import load_dotenv
from ultralytics import YOLO
import shutil

load_dotenv(verbose=True)


data_dir = os.getenv("DATA_DIR")
image_dir = os.path.join(data_dir, 'images')
label_dir = os.path.join(data_dir, 'labels')
yolo_dir = 'datasets/yolo'
meta_data_dir = os.path.join(data_dir, 'meta-data')
class_file = os.path.join(meta_data_dir, 'classes.txt') if 'meta-data' in os.listdir(data_dir) else None
if class_file and os.path.exists(class_file):
    with open(class_file, 'r') as f:
        class_names = {i: line.strip() for i, line in enumerate(f.readlines())}
else:
    class_names = {
        0: 'Helmet', 1: 'Mask', 2: 'Safety Vest', 3: 'Gloves', 4: 'Safety Glasses',
        5: 'Boots', 6: 'Ear Protection', 7: 'Harness', 8: 'Coveralls', 9: 'Respirator',
        10: 'Hard Hat', 11: 'Face Shield', 12: 'Safety Belt', 13: 'Knee Pads',
        14: 'Reflective Tape', 15: 'Goggles', 16: 'Other PPE'
    }
print("Class names:", class_names)
'''
print("Dataset contents:", os.listdir(data_dir))

images = [os.path.join(image_dir,img) for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]


labels = [os.path.join(image_dir,lbl) for lbl in os.listdir(label_dir) if lbl.endswith('.txt')]

images.sort()
labels.sort()
image_names = [os.path.splitext(os.path.basename(img))[0] for img in images]
label_names = [os.path.splitext(os.path.basename(img))[0] for img in labels]

paired_images = [img for img, img_name in zip(images, image_names) if img_name in label_names]
paired_labels = [os.path.join(label_dir, f"{img_name}.txt") for img_name in image_names if img_name in label_names]
print(f"Paired: {len(paired_images)} images, {len(paired_labels)} labels")

# Filter valid PPE annotations
ppe_class_ids = set(range(17))
filtered_images = []
filtered_labels = []
for img_path, lbl_path in zip(paired_images, paired_labels):
    with open(lbl_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        bboxes = [line.split() for line in lines]
        valid_bboxes = [box for box in bboxes if len(box) >= 5]
        if valid_bboxes and any(int(box[0]) in ppe_class_ids for box in valid_bboxes):
            filtered_images.append(img_path)
            filtered_labels.append(lbl_path)

print(f"Filtered: {len(filtered_images)} images, {len(filtered_labels)} labels")


try:
    with open(os.path.join(data_dir, 'train_files.txt'), 'r') as f:
        train_files = set(line.strip() for line in f.readlines() if line.strip())
    with open(os.path.join(data_dir, 'val_files.txt'), 'r') as f:
        val_files = set(line.strip() for line in f.readlines() if line.strip())
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure train_files.txt and val_files.txt exist.")
    raise


# Ensure 1:1 image-label correspondence
train_images = [img for img in filtered_images if os.path.basename(img) in train_files]
train_labels = [lbl for lbl in filtered_labels if os.path.basename(lbl) in [os.path.splitext(os.path.basename(img))[0] + '.txt' for img in train_images]]
val_images = [img for img in filtered_images if os.path.basename(img) in val_files]
val_labels = [lbl for lbl in filtered_labels if os.path.basename(lbl) in [os.path.splitext(os.path.basename(img))[0] + '.txt' for img in val_images]]

# Verify counts
if len(train_images) != len(train_labels) or len(val_images) != len(val_labels):
    print("Warning: Image-label mismatch detected!")
    print(f"Train: {len(train_images)} images, {len(train_labels)} labels")
    print(f"Val: {len(val_images)} images, {len(val_labels)} labels")
else:
    print(f"Train: {len(train_images)} images, {len(train_labels)} labels")
    print(f"Val: {len(val_images)} images, {len(val_labels)} labels")

# Create YOLO structure

for split in ['train', 'val']:
    os.makedirs(os.path.join(yolo_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, split, 'labels'), exist_ok=True)

# Copy files
for img, lbl in zip(train_images, train_labels):
    shutil.copy(img, os.path.join(yolo_dir, 'train', 'images', os.path.basename(img)))
    shutil.copy(lbl, os.path.join(yolo_dir, 'train', 'labels', os.path.basename(lbl)))
for img, lbl in zip(val_images, val_labels):
    shutil.copy(img, os.path.join(yolo_dir, 'val', 'images', os.path.basename(img)))
    shutil.copy(lbl, os.path.join(yolo_dir, 'val', 'labels', os.path.basename(lbl)))


# Generate data.yaml

data_yaml = f"""
train: {os.path.join(yolo_dir, 'train')}
val: {os.path.join(yolo_dir, 'val')}
nc: {len(class_names)}
names: {list(class_names.values())}

with open(os.path.join(yolo_dir, 'data.yaml'), 'w') as f:
    f.write(data_yaml)
print("YOLO dataset prepared at:", yolo_dir)


'''


print("Starting training ...............")
model = YOLO('yolo11n.pt')
model.train(
    data=os.path.join("/Users/venugopalgotagi/PycharmProjects/repos/ppe_compliance_monitoring_system/ppe_compliance_monitoring_train_model/datasets/yolo", 'data.yaml'),
    epochs=30,
    imgsz=640,
    batch=16
)


print("Training complete!")

print("Training complete! Results saved in /kaggle/working/runs/safescan_yolo11n_final")


import os
import shutil
import random

input_dir = r"C:\Users\Asus\OneDrive\Desktop\Interenship Projects\Project 2"
output_dir = r"split_data"

train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

for class_name in os.listdir(input_dir):

    if class_name == "split_data":
        continue

    class_path = os.path.join(input_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if os.path.isfile(os.path.join(class_path, f))]

    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split, split_imgs in zip(['train', 'val', 'test'],
                                [train_imgs, val_imgs, test_imgs]):

        split_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

print("✅ Dataset split completed!")
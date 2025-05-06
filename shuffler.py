import os
import random
import uuid

def shuffle_and_rename_images(folder_path='./with_helmet'):
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        print("No files found in", folder_path)
        return

    random.shuffle(files)

    temp_names = []
    for fname in files:
        old_path = os.path.join(folder_path, fname)
        ext = os.path.splitext(fname)[1]  # keeps .jpg, .png, etc.
        temp_fname = f"temp_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(folder_path, temp_fname)
        os.rename(old_path, temp_path)
        temp_names.append((temp_path, ext))

    for idx, (temp_path, ext) in enumerate(temp_names, start=1):
        new_fname = f"with_helmet_{idx}{ext}"
        new_path = os.path.join(folder_path, new_fname)
        os.rename(temp_path, new_path)

    print(f"Renamed {len(temp_names)} files in '{folder_path}'.")

if __name__ == "__main__":
    shuffle_and_rename_images()

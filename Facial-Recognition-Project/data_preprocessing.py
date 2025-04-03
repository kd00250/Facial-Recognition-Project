import os
import zipfile
import shutil
import re

def extract_and_arrange(zip_file="Faces.zip", extracted_folder="Faces_unorganized", structured_folder="Faces"):
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
    print("Extraction completed. Files extracted into:", extracted_folder)

    print("Arranging files into structured folders...")
    os.makedirs(structured_folder, exist_ok=True)

    # Recursively walk through the extracted folder
    for root, dirs, files in os.walk(extracted_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract label: everything before the underscore followed by digits.
                match = re.match(r'^(.*?)_\d+', filename)
                if match:
                    label = match.group(1)
                else:
                    label = os.path.splitext(filename)[0]
                
                # Create a subfolder for this label if it doesn't exist.
                label_dir = os.path.join(structured_folder, label)
                os.makedirs(label_dir, exist_ok=True)
                
                # Define source and destination paths.
                src = os.path.join(root, filename)
                dst = os.path.join(label_dir, filename)
                shutil.move(src, dst)
    print("Dataset arranged in folder:", structured_folder)

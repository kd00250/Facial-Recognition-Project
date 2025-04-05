import os
import zipfile
import shutil
import re

def extract_and_arrange(zip_file="Faces.zip", extracted_folder="Faces_unorganized", structured_folder="Faces"):
    """
    Extracts a ZIP archive containing face images and arranges the images into a structured directory.
    
    The ZIP file is assumed to be in the same directory as this script. The archive is extracted into
    a temporary folder (extracted_folder), and then the images are moved into a new folder (structured_folder)
    organized by class. Each image file is moved into a subfolder named after the label extracted from its filename.
    
    The function assumes that image filenames follow a pattern where the label appears before an underscore
    followed by digits (e.g., "Akshay Kumar_0.jpg"). If no underscore is found, the entire filename (without extension)
    is used as the label.
    
    Args:
        zip_file (str): The name of the ZIP file containing the images.
        extracted_folder (str): The directory where the ZIP file will be extracted.
        structured_folder (str): The directory where the images will be arranged by class.
    
    Returns:
        None
    """
    # Get the directory where this file is located.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute paths for the ZIP file, the extraction folder, and the final structured folder.
    zip_path = os.path.join(base_dir, zip_file)
    extracted_path = os.path.join(base_dir, extracted_folder)
    structured_path = os.path.join(base_dir, structured_folder)
    
    # Print the location of the ZIP file being extracted.
    print("Extracting zip file from:", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    print("Extraction completed. Files extracted into:", extracted_path)

    print("Arranging files into structured folders...")
    # Ensure the final structured folder exists.
    os.makedirs(structured_path, exist_ok=True)

    # Walk through the extracted folder recursively to process all image files.
    for root, dirs, files in os.walk(extracted_path):
        for filename in files:
            # Process only image files (png, jpg, jpeg).
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract the label from the filename:
                # Everything before the first underscore followed by digits is considered the label.
                match = re.match(r'^(.*?)_\d+', filename)
                if match:
                    label = match.group(1)
                else:
                    # If the expected pattern is not found, use the filename without extension.
                    label = os.path.splitext(filename)[0]
                
                # Create a subfolder for this label inside the structured folder.
                label_dir = os.path.join(structured_path, label)
                os.makedirs(label_dir, exist_ok=True)
                
                # Construct full source and destination paths.
                src = os.path.join(root, filename)
                dst = os.path.join(label_dir, filename)
                # Move the file from the extracted folder to the appropriate class folder.
                shutil.move(src, dst)
    
    print("Dataset arranged in folder:", structured_path)

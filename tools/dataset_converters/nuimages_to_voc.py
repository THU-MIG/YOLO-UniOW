import json
import xml.etree.ElementTree as ET
import os

# Function to create or update Pascal VOC XML with multiple objects
def create_or_update_voc_xml(image_name, width, height, objects, output_dir):
    # Path to the XML file
    xml_path = os.path.join(output_dir, f"{image_name.split('.')[0]}.xml")

    if os.path.exists(xml_path):
        # If the XML already exists, load it and append new objects
        tree = ET.parse(xml_path)
        root = tree.getroot()
    else:
        # If the XML does not exist, create a new one
        root = ET.Element("annotation")

        folder = ET.SubElement(root, "folder").text = "VOC2007"
        filename = ET.SubElement(root, "filename").text = image_name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

    # Add the objects to the XML (for both new and existing XML files)
    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj['class_name']
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(obj['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(obj['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(obj['ymax'])

    # Write the updated XML back to the file
    tree = ET.ElementTree(root)
    tree.write(xml_path)

# Function to load category mapping from JSON
def load_category_mapping(json_file):
    with open(json_file) as f:
        categories = json.load(f)
    category_mapping = {category["token"]: category["name"] for category in categories}
    return category_mapping

# Function to load image metadata mapping from JSON
def load_image_metadata(json_file):
    with open(json_file) as f:
        image_metadata = json.load(f)
    # Map the image token to filename, width, and height
    image_mapping = {image["token"]: {
        "filename": image["filename"],
        "width": image["width"],
        "height": image["height"]
    } for image in image_metadata}
    return image_mapping

# Mapping NuImages annotations to Pascal VOC
def convert_nuimages_to_voc(nuimages_data, category_mapping, image_mapping, output_dir):
    for annotation in nuimages_data:
        image_token = annotation['sample_data_token']  # Token in the annotation
        image_info = image_mapping.get(image_token, None)  # Get the corresponding image info
        
        if not image_info:
            print(f"Image token {image_token} not found in metadata.")
            continue
        
        image_name = os.path.basename(image_info['filename'])  # Get the filename from the metadata
        bbox = annotation['bbox']
        category_token = annotation['category_token']
        class_name = category_mapping.get(category_token, "Unknown")  # Map category_token to class name

        objects = [{
            'class_name': class_name,
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3]
        }]

        image_width, image_height = image_info['width'], image_info['height']

        # Generate the Pascal VOC XML for this image
        create_or_update_voc_xml(image_name, image_width, image_height, objects, output_dir)

# Example usage
# Load the category mapping from the category.json file
category_mapping = load_category_mapping("v1.0-val/category.json")

# Load the image metadata from the corresponding JSON file
image_mapping = load_image_metadata("v1.0-val/sample_data.json")

# Load the annotations from nuimages_annotations.json
with open("v1.0-val/object_ann.json") as f:
    nuimages_data = json.load(f)

# Set output directory for Pascal VOC XML files
output_dir = "Annotations"
os.makedirs(output_dir, exist_ok=True)

# Convert all annotations to Pascal VOC format
convert_nuimages_to_voc(nuimages_data, category_mapping, image_mapping, output_dir)

print(f"Converted {len(nuimages_data)} annotations to Pascal VOC format.")
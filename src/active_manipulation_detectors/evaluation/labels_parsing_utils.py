import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_xml(xml_path: Path) -> dict:
    # Parse the XML file
    tree = ET.parse(xml_path.as_posix())
    root = tree.getroot()

    # Initialize a dictionary to store the parsed data
    data = {"filename": None, "size": {}, "objects": []}

    # Extract filename
    data["filename"] = root.find("filename").text

    # Extract image size
    size = root.find("size")
    data["size"] = {
        "width": int(size.find("width").text),
        "height": int(size.find("height").text),
        "depth": int(size.find("depth").text),
    }

    # Extract objects
    for obj in root.findall("object"):
        obj_data = {
            "name": obj.find("name").text,
            "bndbox": {
                "xmin": int(obj.find("bndbox/xmin").text),
                "ymin": int(obj.find("bndbox/ymin").text),
                "xmax": int(obj.find("bndbox/xmax").text),
                "ymax": int(obj.find("bndbox/ymax").text),
            },
        }
        data["objects"].append(obj_data)

    return data

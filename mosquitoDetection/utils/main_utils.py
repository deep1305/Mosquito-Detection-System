import os.path
import sys
import yaml
import base64
from pathlib import Path

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    output_path = Path(fileName)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = Path("./data") / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
        
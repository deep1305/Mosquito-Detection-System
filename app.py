import os
import pathlib
import shutil
import sys

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from mosquitoDetection.utils.main_utils import decodeImage, encodeImageIntoBase64

app = Flask(__name__)
CORS(app)

BASE_DIR = pathlib.Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "yolov5"
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = YOLO_DIR / "runs"
DEFAULT_IMAGE_NAME = "inputImage.jpg"

# Use env vars to configure these values for local/EC2.
WEIGHTS_PATH = pathlib.Path(os.getenv("YOLO_WEIGHTS", str(YOLO_DIR / "best.pt")))
CONF_THRES = float(os.getenv("YOLO_CONF", "0.5"))
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "416"))
WINDOWS_PATH_COMPAT = os.getenv("YOLO_WINDOWS_PATH_COMPAT", "1") == "1"

_DETECT_RUNNER = None


def _maybe_enable_windows_path_compat():
    """Enable Windows-only path compatibility for legacy Linux-trained checkpoints."""
    if os.name == "nt" and WINDOWS_PATH_COMPAT:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]


def _get_detect_runner():
    global _DETECT_RUNNER

    if _DETECT_RUNNER is not None:
        return _DETECT_RUNNER

    if not YOLO_DIR.exists():
        raise FileNotFoundError(f"YOLO directory not found: {YOLO_DIR}")

    _maybe_enable_windows_path_compat()

    yolo_dir_str = str(YOLO_DIR)
    if yolo_dir_str not in sys.path:
        sys.path.insert(0, yolo_dir_str)

    from detect import run as detect_run  # pylint: disable=import-error

    _DETECT_RUNNER = detect_run
    return _DETECT_RUNNER


def _cleanup_runs():
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR, ignore_errors=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_route():
    try:
        payload = request.get_json(silent=True) or {}
        image_base64 = payload["image"]

        input_image_path = DATA_DIR / DEFAULT_IMAGE_NAME
        decodeImage(image_base64, str(input_image_path))

        if not WEIGHTS_PATH.exists():
            return Response(
                f"Model weights not found at: {WEIGHTS_PATH}. Set YOLO_WEIGHTS correctly.",
                status=500,
            )

        detect_run = _get_detect_runner()
        source_path = str(input_image_path.resolve())
        output_path = YOLO_DIR / "runs" / "detect" / "exp" / DEFAULT_IMAGE_NAME

        detect_run(
            weights=str(WEIGHTS_PATH),
            source=source_path,
            imgsz=(IMG_SIZE, IMG_SIZE),
            conf_thres=CONF_THRES,
            project=str(YOLO_DIR / "runs" / "detect"),
            name="exp",
            exist_ok=True,
        )

        if not output_path.exists():
            return Response("Prediction image was not generated.", status=500)

        encoded_base64 = encodeImageIntoBase64(str(output_path))
        result = {"image": encoded_base64.decode("utf-8")}
        _cleanup_runs()
        return jsonify(result)

    except KeyError:
        return Response("Missing required key: 'image'", status=400)
    except ValueError:
        return Response("Invalid JSON payload.", status=400)
    except Exception as exc:  # pylint: disable=broad-except
        return Response(f"Prediction failed: {exc}", status=500)


@app.route("/live", methods=["GET"])
@cross_origin()
def predict_live():
    try:
        if not WEIGHTS_PATH.exists():
            return Response(
                f"Model weights not found at: {WEIGHTS_PATH}. Set YOLO_WEIGHTS correctly.",
                status=500,
            )

        detect_run = _get_detect_runner()
        detect_run(
            weights=str(WEIGHTS_PATH),
            source="0",
            imgsz=(IMG_SIZE, IMG_SIZE),
            conf_thres=CONF_THRES,
            project=str(YOLO_DIR / "runs" / "detect"),
            name="exp",
            exist_ok=True,
        )
        _cleanup_runs()
        return Response("Camera prediction finished.", status=200)

    except Exception as exc:  # pylint: disable=broad-except
        return Response(f"Live prediction failed: {exc}", status=500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

import os


UPLOAD_FOLDER = "./app/uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


BASE_DIR = os.path.abspath(UPLOAD_FOLDER)


RETINAFACE_MODEL_PATH = "./app/weights/det_10g.onnx"
ARCFACE_MODEL_PATH = "./app/weights/w600k_r50.onnx"
FACE_SWAPPER_MODEL_PATH = "./app/weights/inswapper_128.onnx"
FACE_ENHANCER_MODEL_PATH = './app/weights/gfpgan_1.4.onnx'


PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

FACE_SEG_MODEL_PATH = "./app/weights/79999_iter.pth"

DEVICE = 'cuda'
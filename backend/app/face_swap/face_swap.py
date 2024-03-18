
import cv2
import onnxruntime

import os


from app.face_swap.utils.common import Face
from app.face_swap.retinaface import RetinaFace 
from app.face_swap.arcface_onnx import ArcFaceONNX
from app.face_swap.inswapper import INSwapper
from app.face_swap.face_enhancer import enhance_face
from app.config import UPLOAD_FOLDER
from app.config import RETINAFACE_MODEL_PATH, ARCFACE_MODEL_PATH, FACE_SWAPPER_MODEL_PATH, FACE_ENHANCER_MODEL_PATH
from app.config import PROVIDERS


retinaface_det_model = RetinaFace(RETINAFACE_MODEL_PATH, providers=PROVIDERS)
retinaface_det_model.prepare(ctx_id=1, input_size=(640, 640), det_thresh=0.5)
arcface_emedding_model = ArcFaceONNX(ARCFACE_MODEL_PATH, providers=PROVIDERS)
face_swapper_model = INSwapper(FACE_SWAPPER_MODEL_PATH, providers=PROVIDERS)
face_enhancer_model = onnxruntime.InferenceSession(FACE_ENHANCER_MODEL_PATH,providers=PROVIDERS)






def get_processed_face(img_path):
    image = cv2.imread(img_path)
    bboxes, kpss = retinaface_det_model.detect(image,max_num=1,metric='default')
    print(bboxes)
    bbox = bboxes[0, 0:4]
    det_score = bboxes[0, 4]
    kps = kpss[0]
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    face['embedding'] = arcface_emedding_model.get(image, face)
    return face


def single_face_swap(uid):
    image_location = os.path.join(UPLOAD_FOLDER, uid, "image.jpg")
    result_location = os.path.join(UPLOAD_FOLDER, uid, "result.jpg")

    new_face = get_processed_face(image_location)
    old_face = get_processed_face(result_location)

    frame = cv2.imread(result_location)

    frame = face_swapper_model.get(frame, old_face, new_face, paste_back=True)

    frame = enhance_face(old_face, frame, face_enhancer_model)

    cv2.imwrite(result_location, frame)

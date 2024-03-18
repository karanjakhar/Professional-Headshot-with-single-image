from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse, FileResponse

import uuid
import os
import numpy as np

import PIL
import cv2
import torch
import torchvision.transforms as transforms

from diffusers import StableDiffusionInpaintPipeline
from app.face_swap.face_swap import single_face_swap


from app.face_seg.model import BiSeNet
from app.config import FACE_SEG_MODEL_PATH
from app.config import UPLOAD_FOLDER
from app.config import DEVICE

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net = net.to(DEVICE)

net.load_state_dict(torch.load(FACE_SEG_MODEL_PATH, map_location=torch.device(DEVICE)) )
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",  # React frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_mask(uid):
    image_location = os.path.join(UPLOAD_FOLDER, uid, "image.jpg")
    mask_location = os.path.join(UPLOAD_FOLDER, uid, "mask.jpg")
    with torch.no_grad():
        
        img = PIL.Image.open(image_location)
        image = img.resize((512, 512), PIL.Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(DEVICE)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        mask_image = np.zeros((512,512))
        mask_image[parsing==1] = 1.0
        mask_image[parsing==17] = 1.0
        cv2.imwrite(mask_location, 255 * (1 - mask_image))
        

def make_square(image_path, output_path):
    with PIL.Image.open(image_path) as img:
        longer_side = max(img.size)
        horizontal_padding = (longer_side - img.size[0]) / 2
        vertical_padding = (longer_side - img.size[1]) / 2
        padded_image = PIL.ImageOps.expand(img, border=(int(horizontal_padding), int(vertical_padding)), fill='white')
        padded_image.save(output_path)


@app.post('/upload')
def upload_image(file: UploadFile = File(...)):

   

    uid = str(uuid.uuid4())

    if not os.path.exists(os.path.join(UPLOAD_FOLDER, uid)):
        os.mkdir(os.path.join(UPLOAD_FOLDER, uid))

    image_location = os.path.join(UPLOAD_FOLDER, uid, "image.jpg")
    mask_location = os.path.join(UPLOAD_FOLDER, uid, "mask.jpg")
    result_location = os.path.join(UPLOAD_FOLDER, uid, "result.jpg")

    with open(image_location, "wb+") as image_object:
        image_object.write(file.file.read()) 
    

    make_square(image_location, image_location)
    
    create_mask(uid)


    init_image = PIL.Image.open(image_location)
    mask_image = PIL.Image.open(mask_location)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to(DEVICE)

    prompt = "a person in suit, high resolution, looking towards camera"
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    image.save(result_location)
    single_face_swap(uid)

    return {"uid":uid}


@app.get('/get_result_image/{uid}')
def serve_image(uid: str):
    result_location = os.path.join(UPLOAD_FOLDER, uid, "result.jpg")

    if not os.path.isfile(result_location):
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(result_location)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
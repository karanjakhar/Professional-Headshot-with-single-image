from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse

import uuid
import os
import numpy as np
import io

import PIL
import cv2
import torch
import torchvision.transforms as transforms

from diffusers import StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler


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

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(DEVICE)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



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


def create_mask(img):
    with torch.no_grad():
        image = img.resize((512, 512), PIL.Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(DEVICE)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        mask_image = np.zeros((512,512))
        mask_image[parsing==1] = 1.0
        mask_image[parsing==17] = 1.0
        
    

        print(mask_image.shape)

        mask_image = 255 * (1 - mask_image)
        mask_image = mask_image.astype(np.uint8)
        
        return PIL.Image.fromarray(mask_image)
        

def make_square(img):
    
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    padded_image = PIL.ImageOps.expand(img, border=(int(horizontal_padding), int(vertical_padding)), fill='white')
    return padded_image


@app.post('/upload')
def upload_image(file: UploadFile = File(...)):

    # Read the uploaded image file
    init_image = PIL.Image.open(io.BytesIO(file.file.read()))

    # Convert the image to RGB format if it's RGBA
    if init_image.mode == 'RGBA':
        init_image = init_image.convert('RGB')
    
    init_image = make_square(init_image)
    
    mask_image = create_mask(init_image)

    prompt = "a person in suit, high resolution, looking towards camera"
    
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=20).images[0]

    result_image = single_face_swap(  image, init_image)

    # Convert the numpy array to a PIL Image
    image = PIL.Image.fromarray(result_image)

    # Save the image to a byte buffer
    output = io.BytesIO()
    image.save(output, format="JPEG")
    output.seek(0)

    # Return the image as a StreamingResponse
    return StreamingResponse(output, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
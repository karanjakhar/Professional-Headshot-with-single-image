from io import BytesIO
from pathlib import Path

from modal import (
    Image,
    Stub,
    asgi_app,
    build,
    enter,
    gpu,
    method,
)

image = Image.debian_slim().apt_install("git").apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1", "wget"
    ).pip_install('onnxruntime-gpu', extra_index_url="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/").pip_install(
    "opencv-python",
    "facexlib>=0.2.5",
    "realesrgan",
    "gfpgan==1.3.8",
    "insightface==0.7.3",
    "Pillow~=10.1.0",
    "diffusers~=0.24.0",
    "transformers~=4.35.2",  # This is needed for `import torch`
    "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
    "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
).run_commands('pip uninstall basicsr -y').pip_install("new-basicsr").run_commands('wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P /root/checkpoints/',
 'wget  https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P /root/gfpgan/weights/',
 'wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -P /root/gfpgan/weights/'
 )

stub = Stub("professional-headshot-single-image", image=image)

with image.imports():
    import torch
    from torch import nn
    import numpy as np
    import cv2
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers import DPMSolverMultistepScheduler
    from diffusers.utils import load_image
    from huggingface_hub import snapshot_download, hf_hub_download
    import PIL
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from insightface.app import FaceAnalysis
    import insightface
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer

def load_models_concurrently(load_functions_map: dict) -> dict:
    model_id_to_model = {}
    with ThreadPoolExecutor(max_workers=len(load_functions_map)) as executor:
        future_to_model_id = {
            executor.submit(load_fn): model_id
            for model_id, load_fn in load_functions_map.items()
        }
        for future in as_completed(future_to_model_id.keys()):
            model_id_to_model[future_to_model_id[future]] = future.result()
    return model_id_to_model

@stub.cls(gpu=gpu.T4(), container_idle_timeout=1200)
class Model:
    @build()
    def download_models(self):
        # Ignore files that we don't need to speed up download time.
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]

        snapshot_download("runwayml/stable-diffusion-inpainting", ignore_patterns=ignore)
        snapshot_download("jonathandinu/face-parsing", ignore_patterns=ignore)

        hf_hub_download(repo_id="ashleykleynhans/inswapper", filename="inswapper_128.onnx", repo_type="model", local_dir="/root/checkpoints")
        hf_hub_download(repo_id="dtarnow/UPscaler", filename="RealESRGAN_x2plus.pth", repo_type="model", local_dir="/root/checkpoints")
        snapshot_download(repo_id="karanjakhar/insightface_weights", local_dir="/root/.insightface/models/buffalo_l")

    @enter()
    def enter(self):

        self.PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.face_inswapper_path = "/root/checkpoints/inswapper_128.onnx"
        self.face_enhancer_path = '/root/checkpoints/GFPGANv1.4.pth'
        arch = 'clean'
        channel_multiplier = 2

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.bg_upsampler = RealESRGANer(
                scale=2,
                model_path='/root/checkpoints/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
                )  

        self.components = load_models_concurrently({
            "pipe": lambda: StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16),
            "image_processor": lambda: SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing"),
            "seg_model": lambda: SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing"),
            "face_app": lambda: FaceAnalysis(name="buffalo_l", providers=self.PROVIDERS),
            "face_swapper": lambda: insightface.model_zoo.get_model(self.face_inswapper_path,providers=self.PROVIDERS),
            "face_enhancer_model": lambda: GFPGANer(model_path=self.face_enhancer_path, upscale=2, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=self.bg_upsampler, device="cuda"),
        })

        self.pipe = self.components["pipe"]
        self.seg_model = self.components["seg_model"]
        self.face_app = self.components["face_app"]
        self.face_swapper = self.components["face_swapper"]

        self.face_enhancer_model = self.components["face_enhancer_model"]
        self.image_processor = self.components["image_processor"]

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        self.seg_model = self.seg_model.to("cuda")
        self.seg_model.eval()

        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    @method()
    def inference(self, image_bytes, prompt):
        temp_init_image = load_image(PIL.Image.open(BytesIO(image_bytes)))
        longer_side = max(temp_init_image.size)
        horizontal_padding = (longer_side - temp_init_image.size[0]) / 2
        vertical_padding = (longer_side - temp_init_image.size[1]) / 2
        init_image = PIL.ImageOps.expand(temp_init_image, border=(int(horizontal_padding), int(vertical_padding)), fill='white')
        init_image = init_image.resize((512,512))

        # run face seg inference on image
        inputs = self.image_processor(images=init_image, return_tensors="pt").to('cuda')
        outputs = self.seg_model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

        # resize output to match input image dimensions
        upsampled_logits = nn.functional.interpolate(logits,
                        size=init_image.size[::-1], # H x W
                        mode='bilinear',
                        align_corners=False)

        # get label masks
        labels = upsampled_logits.argmax(dim=1)[0]

        parsing = labels.cpu().numpy()

        mask_image = np.zeros(parsing.shape)
        mask_image[parsing==0] = 1.0
        mask_image[parsing==18] = 1.0
        mask_image[parsing==17] = 1.0

        mask_image = mask_image

        mask_area = np.sum(1-mask_image)
        
        # mask_area_ratio > 0.4 then add padding around the image and resize to 512.
        total_area = np.prod(mask_image.shape)

        mask_area_ratio = mask_area / total_area

        if mask_area_ratio > 0.4:
            # Add padding around the image
            pad = int(((mask_area_ratio * 400) - 100) * 5.12)//2
            mask_image = np.pad(mask_image, ((pad, pad), (pad, pad)), mode='constant',constant_values=(1))
            mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_AREA)
            init_image = PIL.ImageOps.expand(init_image, border=(pad, pad), fill='white')
            init_image = init_image.resize((512, 512))

        # Define the kernel size for erosion
        kernel_size = 5  # You can adjust this value to control the amount of erosion

        # Define the kernel       
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Apply erosion to the mask_image
        eroded_mask = cv2.erode(1-mask_image, kernel, iterations=1)

        eroded_mask = 255 *  (1-eroded_mask)
        eroded_mask = eroded_mask.astype(np.uint8)

        mask = PIL.Image.fromarray(eroded_mask)

        num_inference_steps = 20
      
        image = self.pipe(
            prompt,
            image=init_image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
        ).images[0]

        # face swap
        output = np.array(image)
        source_face = self.face_app.get(np.array(temp_init_image))[0]  # try temp_init_image
        output_face = self.face_app.get(output)[0]
        face_result = self.face_swapper.get(output, output_face, source_face, paste_back=True)
        
        # face enhancer
        _, _, face_result = self.face_enhancer_model.enhance(
                    face_result,
                    paste_back=True
                )
        face_result = face_result.astype(np.uint8)
        face_result = PIL.Image.fromarray(face_result)

        byte_stream = BytesIO()
        face_result.save(byte_stream, format="JPEG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


DEFAULT_IMAGE_PATH = "/home/karan/kj_workspace/kj_ai/Professional-Headshot-with-single-image/backend/modal_inference/akhil.png"


@stub.local_entrypoint()
def main(
    image_path=DEFAULT_IMAGE_PATH,
    prompt="a person in suit, high resolution, looking towards camera, white wall background",
):
    with open(image_path, "rb") as image_file:
        input_image_bytes = image_file.read()
        output_image_bytes = Model().inference.remote(input_image_bytes, prompt)

    dir = Path("/tmp/stable-diffusion-xl-turbo")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(output_image_bytes)




web_image = Image.debian_slim().pip_install("jinja2")


@stub.function(
    image=web_image,
    allow_concurrent_inputs=2,
)
@asgi_app()
def app():

    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    

    web_app = FastAPI()

    # Allow requests from your frontend origin
    origins = [
        "https://professional-headshot.netlify.app",
        "http://localhost:3000",
    ]

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @web_app.post('/upload')
    async def upload_image(file: UploadFile = File(...)):

        # Read the uploaded image file
        input_image_bytes = file.file.read()
        prompt = "a person in suit, high resolution, looking towards camera, white wall background"

        output_image_bytes = Model().inference.remote(input_image_bytes, prompt)


        return StreamingResponse(BytesIO(output_image_bytes), media_type="image/png")

    return web_app
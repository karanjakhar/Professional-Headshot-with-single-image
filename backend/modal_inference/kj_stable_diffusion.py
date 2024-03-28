
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
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    ).pip_install('onnxruntime-gpu', extra_index_url="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/").pip_install(
    "opencv-python",
    "basicsr>=1.4.2",
    "facexlib>=0.2.5",
    "realesrgan",
    "gfpgan==1.3.8",
    "insightface==0.7.3",
    "Pillow~=10.1.0",
    "diffusers~=0.24.0",
    "transformers~=4.35.2",  # This is needed for `import torch`
    "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
    "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
)

stub = Stub("stable-diffusion-xl-turbo", image=image)

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
    #from super_image import EdsrModel, ImageLoader
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

        #snapshot_download("eugenesiow/edsr-base", ignore_patterns=ignore)

        hf_hub_download(repo_id="ashleykleynhans/inswapper", filename="inswapper_128.onnx", repo_type="model", local_dir="/root/checkpoints")
        hf_hub_download(repo_id='Neus/GFPGANv1.4', filename='GFPGANv1.4.onnx', repo_type='model', local_dir="/root/checkpoints")
        hf_hub_download(repo_id="dtarnow/UPscaler", filename="RealESRGAN_x2plus.pth", repo_type="model", local_dir="/root/checkpoints")
        #hf_hub_download(repo_id="karanjakhar/codeformer", filename="codeformer.onnx", repo_type="model", local_dir="/root/checkpoints")
        snapshot_download(repo_id="karanjakhar/insightface_weights", local_dir="/root/.insightface/models/buffalo_l")

    

    @enter()
    def enter(self):

        self.PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.face_inswapper_path = "/root/checkpoints/inswapper_128.onnx"
        self.face_enhancer_path = '/root/checkpoints/GFPGANv1.4.onnx'
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'

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
            #"upscale_model": lambda: EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2),
            "face_app": lambda: FaceAnalysis(name="buffalo_l", providers=self.PROVIDERS),
            "face_swapper": lambda: insightface.model_zoo.get_model(self.face_inswapper_path,providers=self.PROVIDERS),
            #"face_enhancer_model": lambda: onnxruntime.InferenceSession(self.face_enhancer_path,providers=self.PROVIDERS),
            "face_enhancer_model": lambda: GFPGANer(model_path=self.face_enhancer_path, upscale=2, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=self.bg_upsampler, device="cuda"),
        })


        self.pipe = self.components["pipe"]
        self.seg_model = self.components["seg_model"]
        self.face_app = self.components["face_app"]
        self.face_swapper = self.components["face_swapper"]
        #self.upscale_model = self.components["upscale_model"]
        self.face_enhancer_model = self.components["face_enhancer_model"]
        self.image_processor = self.components["image_processor"]

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        self.seg_model = self.seg_model.to("cuda")
        self.seg_model.eval()

        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        


    def blend_frame(self,temp_frame, paste_frame):
        face_enhancer_blend = 0.4
        temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
        return temp_frame
    

    def paste_back(self,temp_frame, crop_frame, affine_matrix ):
        inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_frame_height, temp_frame_width = temp_frame.shape[0:2]
        crop_frame_height, crop_frame_width = crop_frame.shape[0:2]
        inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
        inverse_mask = np.ones((crop_frame_height, crop_frame_width, 3), dtype = np.float32)
        inverse_mask_frame = cv2.warpAffine(inverse_mask, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
        inverse_mask_frame = cv2.erode(inverse_mask_frame, np.ones((2, 2)))
        inverse_mask_border = inverse_mask_frame * inverse_crop_frame
        inverse_mask_area = np.sum(inverse_mask_frame) // 3
        inverse_mask_edge = int(inverse_mask_area ** 0.5) // 20
        inverse_mask_radius = inverse_mask_edge * 2
        inverse_mask_center = cv2.erode(inverse_mask_frame, np.ones((inverse_mask_radius, inverse_mask_radius)))
        inverse_mask_blur_size = inverse_mask_edge * 2 + 1
        inverse_mask_blur_area = cv2.GaussianBlur(inverse_mask_center, (inverse_mask_blur_size, inverse_mask_blur_size), 0)
        temp_frame = inverse_mask_blur_area * inverse_mask_border + (1 - inverse_mask_blur_area) * temp_frame
        temp_frame = temp_frame.clip(0, 255).astype(np.uint8)
        return temp_frame
    

    def normalize_crop_frame(self,crop_frame):
        crop_frame = np.clip(crop_frame, -1, 1)
        crop_frame = (crop_frame + 1) / 2
        crop_frame = crop_frame.transpose(1, 2, 0)
        crop_frame = (crop_frame * 255.0).round()
        crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
        return crop_frame
    
    def prepare_crop_frame(self,crop_frame):
        crop_frame = crop_frame[:, :, ::-1] / 255.0
        crop_frame = (crop_frame - 0.5) / 0.5
        crop_frame = np.expand_dims(crop_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
        return crop_frame
    
    def warp_face(self,target_face,temp_frame):
        template = np.array(
        [
            [ 192.98138, 239.94708 ],
            [ 318.90277, 240.1936 ],
            [ 256.63416, 314.01935 ],
            [ 201.26117, 371.41043 ],
            [ 313.08905, 371.15118 ]
        ])
        affine_matrix = cv2.estimateAffinePartial2D(target_face['kps'], template, method = cv2.LMEDS)[0]
        crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (512, 512))
        return crop_frame, affine_matrix
    

    # def enhance_face(self, target_face, temp_frame, face_enhancer_model):
    #     frame_processor = face_enhancer_model
    #     # crop_frame, affine_matrix = self.warp_face(target_face, temp_frame)
    #     input_frame = self.prepare_crop_frame(temp_frame)
    #     frame_processor_inputs = {}
    #     for frame_processor_input in frame_processor.get_inputs():
    #         if frame_processor_input.name == 'input':
    #             frame_processor_inputs[frame_processor_input.name] = input_frame
    #         if frame_processor_input.name == 'weight':
    #             frame_processor_inputs[frame_processor_input.name] = np.array([ 1 ], dtype = np.double)
        
    #     result_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    #     result_frame = self.normalize_crop_frame(result_frame)
    #     #temp_frame = self.normalize_crop_frame(temp_frame)
    #     #paste_frame = self.paste_back(temp_frame, crop_frame, affine_matrix)
    #     temp_frame = self.blend_frame(temp_frame, result_frame)
    #     return temp_frame

    def enhance_face(self,target_face, temp_frame):
        start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
        padding_x = int((end_x - start_x) * 0.5)
        padding_y = int((end_y - start_y) * 0.5)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        _, _, temp_face = self.face_enhancer_model.enhance(
                    temp_face,
                    paste_back=True
                )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
        return temp_frame


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

        #mask_image = 255 * mask_image
        #mask_image = mask_image.astype(np.uint8)

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

        target_face = self.face_app.get(face_result)[0]
        face_result = self.enhance_face(target_face, face_result, self.face_enhancer_model)
        face_result = face_result.astype(np.uint8)
        face_result = PIL.Image.fromarray(face_result)


        # upscale
        # inputs = ImageLoader.load_image(face_result)
        # preds = self.upscale_model(inputs)
        # pred = preds.data.cpu().numpy()
        # pred[0] = np.clip(pred[0], 0, 255)
        # pred = pred[0].transpose((1, 2, 0)) * 255.0

        # pred = pred.astype(np.uint8)

        # result = PIL.Image.fromarray(pred)

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
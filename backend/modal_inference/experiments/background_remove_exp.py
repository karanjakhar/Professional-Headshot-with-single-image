
from io import BytesIO
from pathlib import Path

from modal import (
    Image,
    Mount,
    Stub,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
)

image = Image.debian_slim().apt_install("git").apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    ).pip_install('onnxruntime-gpu', extra_index_url="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/").pip_install(
    "super-image",
    "rembg",
    "requests",
    "opencv-python",
    "insightface==0.7.3",
    "Pillow~=10.1.0",
    "diffusers~=0.25.0",
    "transformers~=4.35.2",  
    "accelerate~=0.25.0",  
    "safetensors~=0.4.1",
    
)

stub = Stub("stable-diffusion-xl-turbo", image=image)

with image.imports():
    import torch
    from torch import nn
    import numpy as np
    import cv2
    import rembg
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers import DPMSolverMultistepScheduler
    from diffusers.utils import load_image
    from huggingface_hub import snapshot_download, hf_hub_download
    from insightface.app import FaceAnalysis
    import insightface
    import onnxruntime
    import PIL
    import requests
    from io import BytesIO
    from PIL import ImageFilter
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from super_image import EdsrModel, ImageLoader

@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
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
        snapshot_download("eugenesiow/edsr-base", ignore_patterns=ignore)

        hf_hub_download(repo_id="ashleykleynhans/inswapper", filename="inswapper_128.onnx", repo_type="model", local_dir="/root/checkpoints")
        hf_hub_download(repo_id='Neus/GFPGANv1.4', filename='GFPGANv1.4.onnx', repo_type='model', local_dir="/root/checkpoints")
        #hf_hub_download(repo_id="karanjakhar/office_background", filename="office_background.jpg", repo_type="model", local_dir="/root/checkpoints")
        snapshot_download(repo_id="karanjakhar/insightface_weights", local_dir="/root/.insightface/models/buffalo_l")



    @enter()
    def enter(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        # load models
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.seg_model.to("cuda")
        self.seg_model.eval()

        self.PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.face_app = FaceAnalysis(name="buffalo_l", providers=self.PROVIDERS)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.face_inswapper_path = "/root/checkpoints/inswapper_128.onnx"

        self.face_swapper = insightface.model_zoo.get_model(self.face_inswapper_path,
                                                            providers=self.PROVIDERS)

        self.upscale_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

        self.face_enhancer_path = '/root/checkpoints/GFPGANv1.4.onnx'
        self.face_enhancer_model = onnxruntime.InferenceSession(self.face_enhancer_path,providers=self.PROVIDERS)

        # URL of the image
        url = "https://huggingface.co/karanjakhar/office_background/resolve/main/office_background_v4.jpg"

        # Send a GET request to the URL
        response = requests.get(url)

        # Read the image from the response content
        self.dst = np.array(PIL.Image.open(BytesIO(response.content)))
        
    







    def blend_frame(self,temp_frame, paste_frame):
        face_enhancer_blend = 0
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
    

    def enhance_face(self, target_face, temp_frame, face_enhancer_model):
        frame_processor = face_enhancer_model
        crop_frame, affine_matrix = self.warp_face(target_face, temp_frame)
        crop_frame = self.prepare_crop_frame(crop_frame)
        frame_processor_inputs = {}
        for frame_processor_input in frame_processor.get_inputs():
            if frame_processor_input.name == 'input':
                frame_processor_inputs[frame_processor_input.name] = crop_frame
            if frame_processor_input.name == 'weight':
                frame_processor_inputs[frame_processor_input.name] = np.array([ 1 ], dtype = np.double)
        
        crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
        crop_frame = self.normalize_crop_frame(crop_frame)
        paste_frame = self.paste_back(temp_frame, crop_frame, affine_matrix)
        temp_frame = self.blend_frame(temp_frame, paste_frame)
        return temp_frame


    @method()
    def inference(self, image_bytes, prompt):
        temp_init_image = load_image(PIL.Image.open(BytesIO(image_bytes)))
        longer_side = max(temp_init_image.size)
        horizontal_padding = (longer_side - temp_init_image.size[0]) / 2
        vertical_padding = (longer_side - temp_init_image.size[1]) / 2
        init_image = PIL.ImageOps.expand(temp_init_image, border=(int(horizontal_padding), int(vertical_padding)), fill='white')

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

        mask_image = 255 *  mask_image

        mask = PIL.Image.fromarray(mask_image)

        mask = mask.convert('L')

        # Smooth the outline
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

        # Convert to binary
        binary_mask = blurred_mask.point(lambda x: 0 if x < 127 else 255, 'L')

        
        image = self.pipe(
            prompt,
            image=init_image,
            mask_image=binary_mask,
            num_inference_steps=20,
        ).images[0]

        

        # background removal
        # Apply background removal using rembg
        output_array = rembg.remove(np.array(image))
        output_array = output_array[:,:,:3]
        
        # Create a mask for the source image
        mask = 255 * np.ones(output_array.shape, output_array.dtype)
        mask[output_array == 0] = 0


        # Specify the center point in the destination image
        center = (self.dst.shape[1] // 2, self.dst.shape[0] // 2)

        # Clone the source image onto the destination image
        output = cv2.seamlessClone(np.array(image), self.dst, mask, center, cv2.NORMAL_CLONE)

        #face_result = PIL.Image.fromarray(output)


        # face swap
        source_face = self.face_app.get(np.array(init_image))[0]
        output_face = self.face_app.get(output)[0]
        face_result = self.face_swapper.get(output, output_face, source_face, paste_back=True)
        


        # face enhancer
        target_face = self.face_app.get(face_result)[0]
        face_result = self.enhance_face(target_face, face_result, self.face_enhancer_model)
        face_result = face_result.astype(np.uint8)
        face_result = PIL.Image.fromarray(face_result)

        # upscale
        inputs = ImageLoader.load_image(face_result)
        preds = self.upscale_model(inputs)
        pred = preds.data.cpu().numpy()
        pred[0] = np.clip(pred[0], 0, 255)
        pred = pred[0].transpose((1, 2, 0)) * 255.0

        pred = pred.astype(np.uint8)

        result = PIL.Image.fromarray(pred)


        byte_stream = BytesIO()
        result.save(byte_stream, format="JPEG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


DEFAULT_IMAGE_PATH = "/home/karan/kj_workspace/kj_ai/Professional-Headshot-with-single-image/backend/modal_inference/akhil.png"


@stub.local_entrypoint()
def main(
    image_path=DEFAULT_IMAGE_PATH,
    prompt="a person in suit, high resolution, looking towards camera",
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
    allow_concurrent_inputs=20,
)
@asgi_app()
def app():

    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import StreamingResponse
    

    web_app = FastAPI()

    @web_app.post('/upload')
    async def upload_image(file: UploadFile = File(...)):

        # Read the uploaded image file
        input_image_bytes = file.file.read()
        prompt = "a person in suit, high resolution, looking towards camera"

        output_image_bytes = Model().inference.remote(input_image_bytes, prompt)


        return StreamingResponse(BytesIO(output_image_bytes), media_type="image/png")

    return web_app
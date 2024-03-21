
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
    "Pillow~=10.1.0",
    "diffusers==0.25.0",
    "transformers~=4.35.2",  # This is needed for `import torch`
    "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
    "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
    "insightface==0.7.3",
    "opencv-python",
    "einops",
    "ip-adapterv==0.1.0"
)

stub = Stub("stable-diffusion-kj-ip", image=image)

with image.imports():
    import torch
 
    import numpy as np
    import os

    from diffusers.utils import load_image
    from huggingface_hub import snapshot_download,hf_hub_download
    import PIL

    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
    
    from insightface.app import FaceAnalysis
    import insightface
    import onnxruntime
    from insightface.utils import face_align
    from ip_adapter.ip_adapter_faceid import  IPAdapterFaceIDPlus


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
        
        # hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin", repo_type="model")
        #hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")

        #snapshot_download("runwayml/stable-diffusion-inpainting", ignore_patterns=ignore)
        #snapshot_download("h94/IP-Adapter-FaceID", ignore_patterns=ignore)
        snapshot_download("SG161222/Realistic_Vision_V4.0_noVAE", ignore_patterns=ignore)
        snapshot_download("stabilityai/sd-vae-ft-mse", ignore_patterns=ignore)
        snapshot_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ignore_patterns=ignore)
        

    @enter()
    def enter(self):

        self.base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        self.vae_model_path = "stabilityai/sd-vae-ft-mse"
        self.image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # self.ip_ckpt = "ip-adapter-faceid_sd15.bin"
        #self.ip_plus_ckpt = "ip-adapter-faceid-plusv2_sd15.bin"

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
        ).to('cuda')
        from ip_adapter.ip_adapter_faceid import  IPAdapterFaceIDPlus
        self.ip_plus_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")
        self.face_inswapper_path = hf_hub_download(repo_id="ashleykleynhans/inswapper", filename="inswapper_128.onnx", repo_type="model")

        # self.ip_model = IPAdapterFaceID(self.pipe, self.ip_ckpt, 'cuda')
        self.ip_model_plus = IPAdapterFaceIDPlus(self.pipe, self.image_encoder_path, self.ip_plus_ckpt, 'cuda')



        self.face_app = FaceAnalysis(name="buffalo_l", providers=onnxruntime.get_available_providers())
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # os.system(
        #         'wget https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx -P /root/.insightface/models/buffalo_l'
        #     )

        self.face_swapper = insightface.model_zoo.get_model(self.face_inswapper_path,
                                                            providers=onnxruntime.get_available_providers())

        
                
        
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe = self.pipe.to("cuda")

        # load models
        # self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        # self.seg_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        # self.seg_model.to("cuda")
        # self.seg_model.eval()



    @method()
    def inference(self, image_bytes, prompt):
        temp_init_image = load_image(PIL.Image.open(BytesIO(image_bytes)))
        face = np.array(temp_init_image)
        faces = self.face_app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224)

        face_strength = 1.3
        likeness_strength = 1
        total_negative_prompt = 'naked, bikini, skimpy, scanty, bare skin, lingerie, swimsuit, exposed, see-through'

        image = self.ip_model_plus.generate(
            prompt=prompt, negative_prompt=total_negative_prompt, faceid_embeds=faceid_embed,
            scale=likeness_strength, face_image=face_image, shortcut=True, s_scale=face_strength, width=512, height=512, 
            num_inference_steps=30, num_samples=1
        )[0]
        
    
        byte_stream = BytesIO()
        output_face = self.face_app.get(np.array(image))[0]
        result = self.face_swapper.get(np.array(image), output_face, faces[0], paste_back=True)
        result = result.astype(np.uint8)
        result = PIL.Image.fromarray(result)
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
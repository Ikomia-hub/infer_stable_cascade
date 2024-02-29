import copy
from ikomia import core, dataprocess, utils
import torch
import numpy as np
import random
import os
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferStableCascadeParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.prompt = "Anthropomorphic cat dressed as a pilot"
        self.cuda = torch.cuda.is_available()
        self.guidance_scale = 0.0
        self.prior_guidance_scale = 4.0
        self.negative_prompt = ""
        self.height = 1024
        self.width = 1024
        self.num_inference_steps = 30
        self.prior_num_inference_steps = 20
        self.num_inference_steps = 20
        self.num_images_per_prompt = 1
        self.seed = -1
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.prompt = str(param_map["prompt"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.prior_guidance_scale = float(param_map["prior_guidance_scale"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.negative_prompt = str(param_map["negative_prompt"])
        self.seed = int(param_map["seed"])
        self.height = int(param_map["height"])
        self.width = int(param_map["width"])
        if self.width % 128 != 0:
            self.width = self.width // 128 * 128
            print("Updating width to {} to be a multiple of 128".format(self.width))
        if self.height % 128 != 0:
            self.height = self.height // 128 * 128
            print("Updating height to {} to be a multiple of 128".format(self.height))
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.prior_num_inference_steps = int(param_map["prior_num_inference_steps"])
        self.num_images_per_prompt = int(param_map["num_images_per_prompt"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["prompt"] = str(self.prompt)
        param_map["cuda"] = str(self.cuda)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["prior_guidance_scale"] = str(self.prior_guidance_scale)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["height"] = str(self.height)
        param_map["width"] = str(self.width)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["prior_num_inference_steps"] = str(self.prior_num_inference_steps)
        param_map["num_images_per_prompt"] = str(self.num_images_per_prompt)
        param_map["seed"] = str(self.seed)

        return param_map

# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferStableCascade(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.CImageIO())
        # Create parameters object
        if param is None:
            self.set_param_object(InferStableCascadeParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.decoder = None
        self.prior = None
        self.generator = None
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")


    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def load_pipelines(self):

        # Load prior pipeline
        try:
            self.prior = StableCascadePriorPipeline.from_pretrained(
                "stabilityai/stable-cascade-prior",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                cache_dir=self.model_folder,
                local_files_only=True
                ).to(self.device)
        except Exception as e:
            print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
            self.prior = StableCascadePriorPipeline.from_pretrained(
                "stabilityai/stable-cascade-prior",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                cache_dir=self.model_folder
            ).to(self.device)

        # Load decoder pipeline
        try:
            self.decoder = StableCascadeDecoderPipeline.from_pretrained(
                "stabilityai/stable-cascade",
                torch_dtype=torch.float16,
                # use_safetensors=True,
                cache_dir=self.model_folder,
                local_files_only=True,
                ).to(self.device)
        except Exception as e:
            print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
            self.decoder = StableCascadeDecoderPipeline.from_pretrained(
                "stabilityai/stable-cascade",
                torch_dtype=torch.float16,
                # use_safetensors=True,
                cache_dir=self.model_folder,
            ).to(self.device)

    def run(self):
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Load pipelines
        if self.prior is None or self.decoder is None:
        # if param.update or self.prior is None or self.decoder is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.load_pipelines()

        # Generate seed
        if param.seed == -1:
            self.seed = random.randint(0, 191965535)
        else:
            self.seed = param.seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

        with torch.no_grad():
            prior_output = self.prior(
                prompt=param.prompt,
                height=param.height,
                width=param.width,
                generator=self.generator,
                negative_prompt=param.negative_prompt,
                guidance_scale=param.prior_guidance_scale,
                num_images_per_prompt=param.num_images_per_prompt,
                num_inference_steps=param.prior_num_inference_steps
            )
            results = self.decoder(
                image_embeddings=prior_output.image_embeddings.half(),
                height=param.height,
                width=param.width,
                prompt=param.prompt,
                generator=self.generator,
                negative_prompt=param.negative_prompt,
                guidance_scale=param.guidance_scale,
                output_type="pil",
                num_inference_steps=param.num_inference_steps
            ).images

        print(f"Prompt:\t{param.prompt}\nSeed:\t{self.seed}")

        # Set image output
        if len(results) > 1:
            for i, image in enumerate(results):
                self.add_output(dataprocess.CImageIO())
                img = np.array(image)
                output = self.get_output(i)
                output.set_image(img)
        else:
            image = np.array(results[0])
            output_img = self.get_output(0)
            output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferStableCascadeFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_stable_cascade"
        self.info.short_description = "Stable Cascade is a diffusion model trained to generate images given a text prompt."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Pablo Pernias and Dominic Rampas and Mats L. Richter and Christopher J. Pal and Marc Aubreville."
        self.info.article = "Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models"
        self.info.article_url = "https://arxiv.org/pdf/2112.10752.pdf"
        self.info.journal = "arXiv"
        self.info.year = 2023
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2306.00637"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_stable_cascade"
        self.info.original_repository = "https://github.com/Stability-AI/StableCascade"
        # Keywords used for search
        self.info.keywords = "Stable Diffusion, Hugging Face, Stability-AI,text-to-image, Generative"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"
    def create(self, param=None):
        # Create algorithm object
        return InferStableCascade(self.info.name, param)

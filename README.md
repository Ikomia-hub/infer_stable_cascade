<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_stable_cascade</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_stable_cascade">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_stable_cascade">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_stable_cascade/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_stable_cascade.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Stable Cascade is a diffusion model trained to generate images given a text prompt.

![SD cascade](https://huggingface.co/stabilityai/stable-cascade/resolve/main/figures/collage_1.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_stable_cascade", auto_connect=False)

# Run directly on your image
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **prompt** (str) - default 'Anthropomorphic cat dressed as a pilot' : Text prompt to guide the image generation .
- **negative_prompt** (str, *optional*) - default '': The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **prior_num_inference_steps** (int) - default '20': Stage B timesteps.
- **prior_guidance_scale** (float) - default '4.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **num_inference_steps** (int) - default '30': Stage C timesteps
- **guidance_scale** (float) - default '0.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **height** (int) - default '1024': The height in pixels of the generated image.
- **width** (int) - default '1024': The width in pixels of the generated image.
- **num_images_per_prompt** (int) - default '1': Number of generated image(s).
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 191965535.



```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_stable_cascade", auto_connect=False)

algo.set_parameters({
    'prompt': 'Anthropomorphic cat dressed as a pilot',
    'negative_prompt': '',
    'prior_num_inference_steps': '25',
    'prior_guidance_scale': '4.0',
    'num_inference_steps': '100',
    'guidance_scale': '1.0',
    'seed': '-1',
    'width': '1024',
    'height': '1024',
    'num_images_per_prompt':'1',
    })

# Generate your image
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_stable_cascade", auto_connect=False)

# Run  
wf.run()

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

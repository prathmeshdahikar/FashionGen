import streamlit as st
import torch
import PIL
import numpy as np
import ipywidgets as widgets
from PIL import Image
import imageio
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config
from skimage import img_as_ubyte

# Speed up computation
torch.autograd.set_grad_enabled(False)
#torch.backends.cudnn.benchmark = True

# Specify model to use
config = Config(
  model='StyleGAN2',
  layer='style',
  output_class= 'lookbook',
  components=80,
  use_w=True,
  batch_size=5_000, # style layer quite small
)

inst = get_instrumented_model(config.model, config.output_class,
                              config.layer, torch.device('cpu'), use_w=config.use_w)

path_to_components = get_or_compute(config, inst)

model = inst.model

comps = np.load(path_to_components)
lst = comps.files
latent_dirs = []
latent_stdevs = []

load_activations = False

for item in lst:
    if load_activations:
      if item == 'act_comp':
        for i in range(comps[item].shape[0]):
          latent_dirs.append(comps[item][i])
      if item == 'act_stdev':
        for i in range(comps[item].shape[0]):
          latent_stdevs.append(comps[item][i])
    else:
      if item == 'lat_comp':
        for i in range(comps[item].shape[0]):
          latent_dirs.append(comps[item][i])
      if item == 'lat_stdev':
        for i in range(comps[item].shape[0]):
          latent_stdevs.append(comps[item][i])
          
def mix_w(w1, w2, content, style):
    for i in range(0,5):
        w2[i] = w1[i] * (1 - content) + w2[i] * content

    for i in range(5, 16):
        w2[i] = w1[i] * (1 - style) + w2[i] * style
    
    return w2
    
def display_sample_pytorch(seed, truncation, directions, distances, scale, start, end, w=None, disp=True, save=None, noise_spec=None):
    # blockPrint()
    model.truncation = truncation
    if w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy()
        w = [w]*model.get_max_latents() # one per layer
    else:
        w = [np.expand_dims(x, 0) for x in w]
    
    for l in range(start, end):
      for i in range(len(directions)):
        w[l] = w[l] + directions[i] * distances[i] * scale
    
    torch.cuda.empty_cache()
    #save image and display
    out = model.sample_np(w)
    final_im = Image.fromarray((out * 255).astype(np.uint8)).resize((500,500),Image.LANCZOS)
    
    
    if save is not None:
      if disp == False:
        print(save)
      final_im.save(f'out/{seed}_{save:05}.png')
    if disp:
      display(final_im)
    
    return final_im

## Generate image for app
def generate_image(seed1, seed2, content, style, truncation, c0, c1, c2, c3, c4, c5, c6, start_layer, end_layer):
    seed1 = int(seed1)
    seed2 = int(seed2)

    scale = 1
    params = {'c0': c0,
          'c1': c1,
          'c2': c2,
          'c3': c3,
          'c4': c4,
          'c5': c5,
          'c6': c6}

    param_indexes = {'c0': 0,
              'c1': 1,
              'c2': 2,
              'c3': 3,
              'c4': 4,
              'c5': 5,
              'c6': 6}

    directions = []
    distances = []
    for k, v in params.items():
        directions.append(latent_dirs[param_indexes[k]])
        distances.append(v)

    w1 = model.sample_latent(1, seed=seed1).detach().cpu().numpy()
    w1 = [w1]*model.get_max_latents() # one per layer
    im1 = model.sample_np(w1)

    w2 = model.sample_latent(1, seed=seed2).detach().cpu().numpy()
    w2 = [w2]*model.get_max_latents() # one per layer
    im2 = model.sample_np(w2)
    combined_im = np.concatenate([im1, im2], axis=1)
    input_im = Image.fromarray((combined_im * 255).astype(np.uint8))
    

    mixed_w = mix_w(w1, w2, content, style)
    return input_im, display_sample_pytorch(seed1, truncation, directions, distances, scale, int(start_layer), int(end_layer), w=mixed_w, disp=False)
    
    
# Streamlit app title
st.title("FashionGen Demo - AI assisted fashion design")
"""Leverages the StyleGAN architecture and GANSpace exploration to generate realistic garment images. Trained on the extensive LookBook dataset, it streamlines the fashion design process by offering creative and diverse visual inspiration."""

## Side bar texts
st.sidebar.title('Tuning Parameters')
st.sidebar.subheader('(Based on latent space PCA)')

# Create UI widgets
seed1 = st.sidebar.number_input("Seed 1", value=0)
seed2 = st.sidebar.number_input("Seed 2", value=0)
content = st.sidebar.slider("Structural Composition", min_value=0.0, max_value=1.0, value=0.5)
style = st.sidebar.slider("Style", min_value=0.0, max_value=1.0, value=0.5)
truncation = st.sidebar.slider("Dimensional Scaling", min_value=0.0, max_value=1.0, value=0.5)

slider_min_val = -20
slider_max_val = 20
slider_step = 1

c0 = st.sidebar.slider("Sleeve Size Scaling", min_value=slider_min_val, max_value=slider_max_val, value=0)
c1 = st.sidebar.slider("Jacket Features", min_value=slider_min_val, max_value=slider_max_val, value=0)
c2 = st.sidebar.slider("Women's Overcoat", min_value=slider_min_val, max_value=slider_max_val, value=0)
c3 = st.sidebar.slider("Coat", min_value=slider_min_val, max_value=slider_max_val, value=0)
c4 = st.sidebar.slider("Graphic Elements", min_value=slider_min_val, max_value=slider_max_val, value=0)
c5 = st.sidebar.slider("Darker Color", min_value=slider_min_val, max_value=slider_max_val, value=0)
c6 = st.sidebar.slider("Modest Neckline", min_value=slider_min_val, max_value=slider_max_val, value=0)
start_layer = st.sidebar.number_input("Start Layer", value=0)
end_layer = st.sidebar.number_input("End Layer", value=14)

# Call the function with the UI input values and display the output images
input_im, output_im = generate_image(seed1, seed2, content, style, truncation, c0, c1, c2, c3, c4, c5, c6, start_layer, end_layer)

st.image(input_im, caption="Input Image")
st.image(output_im, caption="Output Image")

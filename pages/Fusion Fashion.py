import random
import streamlit as st
import torch
import PIL
import numpy as np
from PIL import Image
import imageio
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config
from skimage import img_as_ubyte
import clip
from torchvision.transforms import Resize, Normalize, Compose, CenterCrop
from torch.optim import Adam
from stqdm import stqdm


st.set_page_config(
    page_title="Fusion Fashion",
    page_icon="👗",
)

#torch.set_num_threads(8)

# Speed up computation
torch.autograd.set_grad_enabled(True)
torch.backends.cudnn.benchmark = True

# Specify model to use
config = Config(
  model='StyleGAN2',
  layer='style',
  output_class= 'lookbook',
  components=80,
  use_w=True,
  batch_size=5_000, # style layer quite small
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = Compose([
    Resize(224),
    CenterCrop(224),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

@st.cache_data
def clip_optimized_latent(text, seed, iterations=25, lr=1e-2):
    seed = int(seed)
    text_input = clip.tokenize([text]).to(device)

    # Initialize a random latent vector
    latent_vector = model.sample_latent(1,seed=seed).detach().to(device)
    latent_vector.requires_grad = True
    latent_vector = [latent_vector]*model.get_max_latents()
    params = [torch.nn.Parameter(latent_vector[i], requires_grad=True) for i in range(len(latent_vector))]
    optimizer = Adam(params, lr=lr, betas=(0.9, 0.999))

    #with torch.no_grad():
    #    text_features = clip_model.encode_text(text_input)
        
    #pbar = tqdm(range(iterations), dynamic_ncols=True)
    
    for iteration in stqdm(range(iterations)):
        optimizer.zero_grad()

        # Generate an image from the latent vector
        image = model.sample(params)
        image = image.to(device)
        
        # Preprocess the image for the CLIP model
        image = preprocess(image)
        #image = clip_preprocess(Image.fromarray((image_np * 255).astype(np.uint8))).unsqueeze(0).to(device)
        
        # Extract features from the image
        #image_features = clip_model.encode_image(image)

        # Calculate the loss and backpropagate
        loss = 1 - clip_model(image, text_input)[0] / 100
        #loss = -torch.cosine_similarity(text_features, image_features).mean()
        loss.backward()
        optimizer.step()
        
        #pbar.set_description(f"Loss: {loss.item()}")  # Update the progress bar to show the current loss
        w = [param.detach().cpu().numpy() for param in params]
    
    return w
          
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
        w_numpy = [x.cpu().detach().numpy() for x in w]
        w = [np.expand_dims(x, 0) for x in w_numpy]
        #w = [x.unsqueeze(0) for x in w]

    
    for l in range(start, end):
      for i in range(len(directions)):
        w[l] = w[l] + directions[i] * distances[i] * scale

    w = [torch.from_numpy(x).to(device) for x in w]
    torch.cuda.empty_cache()
    #save image and display
    out = model.sample(w)
    out = out.permute(0, 2, 3, 1).cpu().detach().numpy()
    out = np.clip(out, 0.0, 1.0).squeeze()
    
    final_im = Image.fromarray((out * 255).astype(np.uint8)).resize((500,500),Image.LANCZOS)
    
    
    if save is not None:
      if disp == False:
        print(save)
      final_im.save(f'out/{seed}_{save:05}.png')
    if disp:
      display(final_im)
    
    return final_im

## Generate image for app
def generate_image(content, style, truncation, c0, c1, c2, c3, c4, c5, c6, start_layer, end_layer,w1,w2):

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
            
    if w1 is not None and w2 is not None:
        w1 = [torch.from_numpy(x).to(device) for x in w1]
        w2 = [torch.from_numpy(x).to(device) for x in w2]


    #w1 = clip_optimized_latent(text1, seed1, iters)
    im1 = model.sample(w1)
    im1_np = im1.permute(0, 2, 3, 1).cpu().detach().numpy()
    im1_np = np.clip(im1_np, 0.0, 1.0).squeeze()
    
    #w2 = clip_optimized_latent(text2, seed2, iters)
    im2 = model.sample(w2)
    im2_np = im2.permute(0, 2, 3, 1).cpu().detach().numpy()
    im2_np = np.clip(im2_np, 0.0, 1.0).squeeze()

    combined_im = np.concatenate([im1_np, im2_np], axis=1)
    input_im = Image.fromarray((combined_im * 255).astype(np.uint8))
    

    mixed_w = mix_w(w1, w2, content, style)
    return input_im, display_sample_pytorch(seed1, truncation, directions, distances, scale, int(start_layer), int(end_layer), w=mixed_w, disp=False)
    
    
# Streamlit app title
st.image('./pics/logo.jpeg')
'''## Fusion Fashion'''

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    inst = get_instrumented_model(config.model, config.output_class,
                              config.layer, device, use_w=config.use_w)
    return clip_model, inst

# Then, to load your models, call this function:
clip_model, inst = load_model()
model = inst.model


path_to_components = get_or_compute(config, inst)
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

## Side bar texts
st.sidebar.title('Customization Options')


# Create UI widgets
text1 = st.sidebar.text_input("Style Specs 1", help = "Provide a clear and concise description of the design you wish to generate. This helps the app understand your preferences and create a customized design that matches your vision.")
text2 = st.sidebar.text_input("Style Specs 2", help = "Provide a clear and concise description of the design you wish to generate. This helps the app understand your preferences and create a customized design that matches your vision.")
if 'seed1' not in st.session_state and 'seed2' not in st.session_state:
    #st.session_state['seed1'] = random.randint(1, 1000)
    st.session_state['seed1'] = 3
    #st.session_state['seed2'] = random.randint(1, 1000)
    st.session_state['seed2'] = 200
    
with st.sidebar.expander("Advanced"):
    seed1 = st.number_input("ID 1", value= st.session_state['seed1'], help = "Capture this unique id to reproduce the exact same result later.")
    seed2 = st.number_input("ID 2", value= st.session_state['seed2'], help = "Capture this unique id to reproduce the exact same result later.")

    st.session_state['seed1'] = seed1
    st.session_state['seed2'] = seed2
    iters = st.number_input("Cycles", value = 25, help = "Increase the sensitivity of the algorithm to find the design matching the style description. Higher values might enhance the accuracy but may lead to slower loading times")

submit_button = st.sidebar.button("Discover")
content = st.sidebar.slider("Design Frame", min_value=0.0, max_value=1.0, value=0.5, help = "Increasing it makes the structure similar to the image on the right and decreasing it will make the structure similar to the image on the left")
style = st.sidebar.slider("Style Composition", min_value=0.0, max_value=1.0, value=0.5, help = "Increasing it makes the style/pattern similar to the image on the right and decreasing it will make the style/pattern similar to the image on the left")
truncation = 0.5
#truncation = st.sidebar.slider("Dimensional Scaling", min_value=0.0, max_value=1.0, value=0.5)

slider_min_val = -20
slider_max_val = 20
slider_step = 1

c0 = st.sidebar.slider("Sleeve Size Scaling", min_value=slider_min_val, max_value=slider_max_val, value=0, help="Adjust the scaling of sleeve sizes. Increase to make sleeve sizes appear larger, and decrease to make them appear smaller.")
c1 = st.sidebar.slider("Jacket Features", min_value=slider_min_val, max_value=slider_max_val, value=0, help = "Control the prominence of jacket features. Increasing this value will make the features more pronounced, while decreasing it will make them less noticeable")
c2 = st.sidebar.slider("Women's Overcoat", min_value=slider_min_val, max_value=slider_max_val, value=0, help = "Modify the dominance of the women's overcoat style. Increase the value to enhance its prominence, and decrease it to reduce its impact.")
c3 = st.sidebar.slider("Coat", min_value=slider_min_val, max_value=slider_max_val, value=0, help = "Control the prominence of coat features. Increasing this value will make the features more pronounced, while decreasing it will make them less noticeable")
c4 = st.sidebar.slider("Graphic Elements", min_value=slider_min_val, max_value=slider_max_val, value=0, help = "Fine-tune the visibility of graphic elements. Increasing this value will make the graphics more prominent, while decreasing it will make them less visible.")
c5 = st.sidebar.slider("Darker Color", min_value=slider_min_val, max_value=slider_max_val, value=0, help = "Adjust the intensity of the color tones towards darker shades. Increasing this value will make the colors appear deeper, while decreasing it will lighten the overall color palette.")
c6 = st.sidebar.slider("Neckline", min_value=slider_min_val, max_value=slider_max_val, value=0,help = "Control the emphasis on the neckline of the garment. Increase to highlight the neckline, and decrease to downplay its prominence.")
start_layer = 0
end_layer = 14
#start_layer = st.sidebar.number_input("Start Layer", value=0)
#end_layer = st.sidebar.number_input("End Layer", value=14)

# if 'w1-np' not in st.session_state:
    # st.session_state['w1-np'] = None

# if 'w2-np' not in st.session_state:
    # st.session_state['w2-np'] = None


if submit_button:  # Execute when the submit button is pressed
    w1 = clip_optimized_latent(text1, seed1, iters)
    st.session_state['w1-np'] = w1
    w2 = clip_optimized_latent(text2, seed2, iters)
    st.session_state['w2-np'] = w2

try:
    input_im, output_im = generate_image(content, style, truncation, c0, c1, c2, c3, c4, c5, c6, start_layer, end_layer,st.session_state['w1-np'],st.session_state['w2-np'])
    st.image(input_im, caption="Input Image")
    st.image(output_im, caption="Output Image")
except:
    pass
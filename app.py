import streamlit as st
#from streamlit_extras.app_logo import add_logo


st.set_page_config(
    page_title="FashionGen",
    page_icon="👗",
)

#st.title("FashionGen")

#add_logo("./pics/logo_small.jpeg", height=20)

st.image('./pics/logo.jpeg')
'''#### About:'''
'''This application employs the StyleGAN framework, CLIP and GANSpace exploration techniques to synthesize images of garments from textual inputs. With training based on the comprehensive LookBook dataset, it supports an efficient fashion design process by transforming text into visual concepts, showcasing the practical application of Generative Adversarial Networks (GANs) in the realm of creative design.'''


''' There are two modes of image generatation: \n
**Fashion Fusion:** Takes two descriptions and generates two designs whose styles features can be combined and edited. \n
**Style One:** Takes a single description and generates one design whose style features can be edited.'''
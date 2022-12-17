import tkinter as tk
import customtkinter as ctk

from PIL import Image, ImageTk
from auth_token import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Creating app interface
app = tk.Tk()
app.geometry('500x840')
app.title('Text to Image Stable')
app['background'] = '#121212'

# Adding prompt entry
prompt = ctk.CTkEntry(height=40, width=460,
                      text_font=('Helvetica', 12),
                      text_color='white',
                      fg_color='#212123')

prompt.place(x=20, y=620)
prompttext = ctk.CTkLabel(height=22,
                          width=50,
                          text='Enter your prompt',
                          text_font=('Helvetica', 8),
                          text_color='white',
                          fg_color='#121212')

prompttext.place(x=22, y=595)

# Title
titlebox = ctk.CTkLabel(height=22,
                        width=50,
                        text='AI-Art-Generator',
                        text_font=('Helvetica', 20),
                        text_color='#1E88E5',
                        fg_color='#121212')
titlebox.place(x=162, y=20)


# Subtitle
subtext = ctk.CTkLabel(height=10,
                       width=50,
                       text='Using Stable Diffusion',
                       text_font=('Helvetica', 8),
                       text_color='white',
                       fg_color='#121212')

subtext.place(x=202, y=50)


# Image grid

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = (230, 230)
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img.resize((224, 224)), box=(i % cols*w, i//cols*h))
    return grid


placeholder = Image.open('ai.png')
placeholder_img = ImageTk.PhotoImage(placeholder)
lmain = ctk.CTkLabel(height=200, width=200, text='')
lmain.place(x=20, y=100)
lmain.configure(image=placeholder_img)


# Importing model and setting up the Stable Diffusion Pipeline

model_id = 'CompVis/stable-diffusion-v1-4'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision='fp16', torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

global inference_steps
inference_steps = 50


# Art Generation function

def generate():
    image_aray = []
    with autocast(device):
        for i in range(4):
            image_aray.append(
                pipe(prompt.get(), num_inference_steps=inference_steps, guidance_scale=8.5).images[0])

    grid = image_grid(image_aray, rows=2, cols=2)
    img = ImageTk.PhotoImage(grid)
    label = ctk.CTkLabel(height=200, width=320)
    label.image = img
    lmain.configure(image=img)


# Function to change the number of denoising steps (inference steps)
# in order to change the quality level
def set_quality(choice):
    global inference_steps
    if choice == 'Low: Abstract and faster results. (Processing time: <1 min)':
        inference_steps = 15
    elif choice == 'High: Higher quality but slower results. (Processing time: 3-4 mins)':
        inference_steps = 100
    else:
        inference_steps = 50


# Dropdown menu to select the quality level

comboboxtext = ctk.CTkLabel(height=22,
                            width=50,
                            text='Quality level',
                            text_font=('Helvetica', 8),
                            text_color='white',
                            fg_color='#121212')
comboboxtext.place(x=22, y=675)


combobox = ctk.CTkOptionMenu(width=460,
                             height=40,
                             master=app,
                             values=[
                                 'Low: Abstract and faster results. (Processing time: <1 min)',
                                 'Medium (Default): Better quality results (Processing time: 1-2 mins)',
                                 'High: Higher quality but slower results. (Processing time: 3-4 mins)'],
                             fg_color='#212123',
                             button_color='#212123',
                             command=set_quality)
combobox.place(y=700, x=20)
combobox.set("Medium")

# Button to start the Image generation process

trigger = ctk.CTkButton(height=40, width=120, text_font=('Helvetica', 12),
                        text_color='white',
                        fg_color='#1E88E5',
                        command=generate)
trigger.configure(text='Generate')
trigger.place(x=190, y=780)

# main loop

app.mainloop()

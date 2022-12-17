import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from auth_token import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# creating app interface
app = tk.Tk()
app.geometry('532x622')
app.title('Text to Image Stable')
ctk.set_appearance_mode('dark')

prompt = ctk.CTkEntry(height=40, width=500,
                      text_font=('Arial', 20),
                      text_color='black',
                      fg_color='white')

prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=500, width=512)
lmain.place(x=10, y=110)

model_id = 'CompVis/stable-diffusion-v1-4'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision='fp16', torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)


def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]

    image.save('generatedimage.png')
    print
    img = ImageTk.PhotoImage(image)
    label = ctk.CTkLabel(image=img, height=320, width=320)
    label.image = img
    # label.pack()
    lmain.configure(image=img)


trigger = ctk.CTkButton(height=40, width=120, text_font=('Arial', 20),
                        text_color='white',
                        fg_color='green',
                        command=generate)
trigger.configure(text='Generate')
trigger.place(x=206, y=60)
app.mainloop()

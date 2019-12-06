import utils
import models
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

content_path = "style/mitsuha1.jpg"
style_path = "style/mosaic.jpg"

transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

device = ("cuda" if torch.cuda.is_available() else "cpu")

# network
vgg = models.vgg
vgg.load_state_dict(torch.load('vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)

decoder = models.decoder
decoder.load_state_dict(torch.load('decoder.pth'))
decoder.to(device)

content = transform(Image.open(str(content_path)))
style = transform(Image.open(str(style_path)))
style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)
with torch.no_grad():
    l_s = vgg(style)
    l_c = vgg(content)
    latent = utils.adain(l_c,l_s)
    out = decoder(latent)
    utils.save_image(out[0],'result.jpg')
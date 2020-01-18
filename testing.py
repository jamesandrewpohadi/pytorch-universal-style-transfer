import utils
import models
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

# hyperparams
method = 'adain' # adain | wct
decoder_weight = 'D_40000.pth'

styles = glob('test/style/*')

transform = transforms.Compose([
    transforms.Resize(size=(512)),
    transforms.CenterCrop(512),
    transforms.ToTensor()
])

device = ("cuda" if torch.cuda.is_available() else "cpu")

# network
vgg = models.vgg
vgg.load_state_dict(torch.load('vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)

decoder = models.decoder
decoder.load_state_dict(torch.load('weights/'+decoder_weight))
decoder.to(device)

test_dataset = utils.datasets.FlatFolderDataset('test/content', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

save_dir = os.path.join('test','result')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with torch.no_grad():
    for style_path in styles:
        style_name = os.path.basename(style_path).split('.')[0]
        print(style_name)
        style = transform(Image.open(str(style_path)))
        style = style.to(device).unsqueeze(0)
        l_s = vgg(style)
        for batch_id, (content) in tqdm(enumerate(test_loader),total=len(test_loader),leave=False):
            l_c = vgg(content.to(device))
            if method=='adain':
                latent = utils.adain(l_c,l_s)
            elif method=='wct':
                latent = utils.whiten_and_color(l_c.reshape(512,-1),l_s.reshape(512,-1)).reshape(1,512,107,107)
            out = decoder(latent)
            utils.save_image(out[0],'test/result/{}_{}.jpg'.format(style_name,batch_id))
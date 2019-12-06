import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import models
import utils
from telegram import Bot
import torch.nn as nn
import torch.utils.data as data
from utils import FlatFolderDataset

# hyperparameters
lr=0.0001
lr_decay=5e-5
dataset_path='dataset'
style_dir='style'
num_epoch=1
batch_size=1
save_every=10
out='ckpt'
content_weight=1
style_weight=10

device = ("cuda" if torch.cuda.is_available() else "cpu")

# use bot to train
bot = Bot('957726088:AAGFHUZMgVUAxSp4CxP8458qGIQxA4WJOFs')

def printb(m):
    bot.sendMessage(-391529154,m)
    print(m)

def sendPhoto(p):
    img = open(p,'rb')
    bot.sendPhoto(-391529154,img)

# network
vgg = models.vgg
vgg.load_state_dict(torch.load('vgg_normalised.pth'))
vgg.to(device)
vgg.train(False)
vgg_layers = list(vgg.children())
relu1_1 = nn.Sequential(*vgg_layers[:4])  # input -> relu1_1
relu2_1 = nn.Sequential(*vgg_layers[4:11])  # relu1_1 -> relu2_1
relu3_1 = nn.Sequential(*vgg_layers[11:18])  # relu2_1 -> relu3_1
relu4_1 = nn.Sequential(*vgg_layers[18:31])  # relu3_1 -> relu4_1
for layer in [relu1_1,relu2_1,relu3_1,relu4_1]:
    for param in layer.parameters():
        param.requires_grad = False

decoder = models.decoder
decoder.load_state_dict(torch.load('decoder.pth'))
decoder.to(device)

# optimizer
optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
style_dataset = FlatFolderDataset(style_dir, transform)
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=batch_size,
    sampler=utils.InfiniteSamplerWrapper(style_dataset),
    num_workers=16))

# training
batch_count = 1
batch_total_loss_sum = 0
print('start training ...')
for epoch in range (1, num_epoch+1):
    for batch_id, (content, _) in tqdm(enumerate(train_loader)):
        utils.adjust_learning_rate(optimizer, batch_count,lr,lr_decay)
        optimizer.zero_grad()

        style = next(style_iter).to(device)
        content = content.to(device)

        style_feats = [relu1_1(style)]
        style_feats.append(relu2_1(style_feats[-1]))
        style_feats.append(relu3_1(style_feats[-1]))
        style_feats.append(relu4_1(style_feats[-1]))
        content_feat = relu1_1(content)
        content_feat = relu2_1(content_feat)
        content_feat = relu3_1(content_feat)
        content_feat = relu4_1(content_feat)
        latent = utils.adain(content_feat, style_feats[-1])

        styled = decoder(latent)
        styled_feats = [relu1_1(styled)]
        styled_feats.append(relu2_1(styled_feats[-1]))
        styled_feats.append(relu3_1(styled_feats[-1]))
        styled_feats.append(relu4_1(styled_feats[-1]))

        loss_c = utils.calc_content_loss(styled_feats[-1], latent)
        loss_s = 0
        for i in range(4):
            loss_s += utils.calc_style_loss(styled_feats[i], style_feats[i])

        loss_c = content_weight * loss_c
        loss_s = style_weight * loss_s
        t_loss = loss_c + loss_s

        t_loss.backward()
        optimizer.step()
        batch_total_loss_sum += t_loss
        batch_count += 1
        if (batch_count % save_every == 0) or (batch_count==num_epoch*len(train_loader)):
            printb("========Iteration {}/{}========".format(batch_count, num_epoch*len(train_loader)))
            printb("\tTotal Loss:\t{:.2f}\n\tCurrent Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count,t_loss))
            sample_image_path = out + "/batch_" + str(batch_count) + ".png"
            utils.save_image(styled[0],sample_image_path)
            sendPhoto(sample_image_path)
            torch.save(decoder.state_dict(), out + '/D_'+str(batch_count)+".pth")
            printb("Saved sample tranformed image at {}".format(sample_image_path))
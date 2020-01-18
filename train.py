import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import models
import utils
import torch.nn as nn
import torch.utils.data as data

# hyperparameters
lr=1e-5
lr_decay=5e-7
content_weight=1.0
style_weight=10.0
num_epoch=1
batch_size=1
save_every=1000
content_dir='data/content'
style_dir='data/style'
out='ckpt'

device = ("cuda" if torch.cuda.is_available() else "cpu")

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
optimizer = torch.optim.SGD(decoder.parameters(), lr=lr, momentum=0, weight_decay=5e-4)

transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(content_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
style_dataset = utils.datasets.FlatFolderDataset(style_dir, transform)
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=batch_size,
    sampler=utils.datasets.InfiniteSamplerWrapper(style_dataset),
    num_workers=16))

# training
batch_count = 1
batch_total_loss_sum = 0
print('start training ...')
for epoch in range (1, num_epoch+1):
    print('Epoch:',epoch)
    for batch_id, (content, _) in tqdm(enumerate(train_loader),total=len(train_loader),leave=False):
        # utils.adjust_learning_rate(optimizer, batch_count,lr,lr_decay)
        optimizer.zero_grad()

        content = content.to(device)
        style = next(style_iter).to(device)[:content.shape[0]]

        style_feats = [relu1_1(style)]
        style_feats.append(relu2_1(style_feats[-1]))
        style_feats.append(relu3_1(style_feats[-1]))
        style_feats.append(relu4_1(style_feats[-1]))
        content_feat = relu1_1(content)
        content_feat = relu2_1(content_feat)
        content_feat = relu3_1(content_feat)
        content_feat = relu4_1(content_feat)
        latent = utils.adain(content_feat, style_feats[3])

        styled = decoder(latent)
        styled_feats = [relu1_1(styled)]
        styled_feats.append(relu2_1(styled_feats[-1]))
        styled_feats.append(relu3_1(styled_feats[-1]))
        styled_feats.append(relu4_1(styled_feats[-1]))

        loss_c = utils.loss.content_loss(styled_feats[-1], latent)
        loss_s = 0
        for i in range(4):
            # loss_s += utils.loss.gram_loss(styled_feats[i], style_feats[i])
            loss_s += utils.loss.mean_std_loss(styled_feats[i], style_feats[i])

        loss_c = content_weight * loss_c
        loss_s = style_weight * loss_s
        t_loss = loss_c + loss_s

        t_loss.backward()
        optimizer.step()
        batch_total_loss_sum += t_loss
        batch_count += 1
        if (batch_count % save_every == 0) or (batch_count==num_epoch*len(train_loader)):
            print("========Iteration {}/{}========".format(batch_count, num_epoch*len(train_loader)))
            print("\tTotal Loss:\t{:.2f}\n\tCurrent Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count,t_loss))
            sample_image_path = out + "/batch_" + str(batch_count) + ".png"
            utils.save_image(styled[0],sample_image_path)
            torch.save(decoder.state_dict(), 'weights/D_'+str(batch_count)+".pth")
            print("Saved sample tranformed image at {}".format(sample_image_path))

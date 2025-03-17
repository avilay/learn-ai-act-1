from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dutils
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets


MEANS = [0.5]
STDS = [0.5]
BATCH_SIZE = 32
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 25

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load():
    xforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])
    trainset = dsets.MNIST('/data/pytorch/mnist', download=True, train=True, transform=xforms)
    trainloader = dutils.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    return len(trainset), trainloader

def build_discriminator():
    discriminator = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, bias=False)),
        ('relu1', nn.LeakyReLU(0.2, inplace=True)),

        ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=False)),
        ('bn2', nn.BatchNorm2d(128)),
        ('relu2', nn.LeakyReLU(0.2, inplace=True)),

        ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)),
        ('bn3', nn.BatchNorm2d(256)),
        ('relu3', nn.LeakyReLU(0.2, inplace=True)),

        ('conv4', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)),
        ('bn4', nn.BatchNorm2d(512)),
        ('relu4', nn.LeakyReLU(0.2, inplace=True)),

        ('conv5', nn.Conv2d(in_channels=512, out_channels=1, kernel_size=6, bias=False)),
        ('sigmoid', nn.Sigmoid())
    ]))

    discriminator.to(DEVICE)
    discriminator.apply(init_weights)
    return discriminator

def build_generator():
    generator = nn.Sequential(OrderedDict([
        ('deconv1', nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=6, bias=False)),
        ('bn1', nn.BatchNorm2d(512)),
        ('relu1', nn.ReLU(512)),

        ('deconv2', nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)),
        ('bn2', nn.BatchNorm2d(256)),
        ('relu2', nn.ReLU(True)),

        ('deconv3', nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)),
        ('bn3', nn.BatchNorm2d(128)),
        ('relu3', nn.ReLU(True)),

        ('deconv4', nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, bias=False)),
        ('bn4', nn.BatchNorm2d(64)),
        ('relu4', nn.ReLU(True)),

        ('deconv5', nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, bias=False)),
        ('tanh', nn.Tanh())
    ]))

    generator.to(DEVICE)
    generator.apply(init_weights)
    return generator

def train_batch(discriminator, optimD, generator, optimG, loss_fn, images):
    real_label = 1
    fake_label = 0
    with torch.enable_grad():
        optimD.zero_grad()
        optimG.zero_grad()

        real_images = images.to(DEVICE)

        noise = torch.randn(BATCH_SIZE, 100, 1, 1, device=DEVICE)
        fake_images = generator.forward(noise)

        # Train discriminator - on real images
        labels = torch.full((BATCH_SIZE,), real_label, device=DEVICE)
        hvals = discriminator.forward(real_images).squeeze()
        lossD_real = loss_fn(hvals, labels)
        lossD_real.backward()

        # Train discriminator - on fake images
        labels.fill_(fake_label)
        hvals = discriminator.forward(fake_images.detach()).squeeze()
        lossD_fake = loss_fn(hvals, labels)
        lossD_fake.backward()

        # Update the discriminator weights
        optimD.step()

        # Train generator on fake images labeled as real
        labels.fill_(real_label)
        hvals = discriminator.forward(fake_images).squeeze()
        lossG = loss_fn(hvals, labels)
        lossG.backward()

        # Update the generator weights
        optimG.step()

        return lossD_real+lossD_fake, lossG

def main():
    discriminator = build_discriminator()
    generator = build_generator()
    loss_fn = nn.BCELoss()
    optimD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    len_trainset, trainloader = load()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        epoch_lossD = 0
        epoch_lossG = 0

        for batch, (images, _) in enumerate(trainloader):
            lossD, lossG = train_batch(discriminator, optimD, generator, optimG, loss_fn, images)
            epoch_lossD += lossD * len(images)
            epoch_lossG += lossG * len(images)
            if batch % 100 == 0:
                print(f'Batch {batch}/{len(trainloader)} - Discriminator Loss: {lossD:.3f}  Generator Loss: {lossG:.3f}')

        epoch_lossD = epoch_lossD / len_trainset
        epoch_lossG = epoch_lossG / len_trainset
        print(f'\n  Discriminator Loss: {epoch_lossD:.3f}  Generator Loss: {epoch_lossG:.3f}')

        # Checkpoint
        torch.save(discriminator.state_dict(), f'./models/mnist_gan_discriminator_{epoch}.pth')
        torch.save(generator.state_dict(), f'./models/mnist_gan_generator_{epoch}.pth')


if __name__ == '__main__':
    main()

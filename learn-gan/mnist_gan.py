import os
from pathlib import Path

import pretty_traceback
import torch as t
import torchvision as tv
from tqdm import tqdm

pretty_traceback.install()


# Hyperparams
N_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
Z_DIM = 64
GEN_HIDDEN_DIM = 128
DISC_HIDDEN_DIM = 128

IMG_DIM = 28 * 28
DATAROOT = Path.home() / "mldata" / "mnist"
RUNROOT = Path.home() / "mlruns" / "mnist_gan"


def save(gen, epoch):
    z = t.randn(25, Z_DIM)
    outputs = gen(z)
    images = outputs.detach().cpu().view(-1, 1, 28, 28)
    tv.utils.save_image(
        images, RUNROOT / f"{epoch}.png", nrow=5, padding=3, pad_value=1
    )


def gen_block(input_dim, output_dim):
    return t.nn.Sequential(
        t.nn.Linear(input_dim, output_dim),
        t.nn.BatchNorm1d(output_dim),
        t.nn.ReLU(inplace=True),
    )


class Generator(t.nn.Module):
    def __init__(self, z_dim, hidden_dim, im_dim):
        super().__init__()
        self.gen = t.nn.Sequential(
            gen_block(z_dim, hidden_dim),
            gen_block(hidden_dim, hidden_dim * 2),
            gen_block(hidden_dim * 2, hidden_dim * 4),
            gen_block(hidden_dim * 4, hidden_dim * 8),
            t.nn.Linear(hidden_dim * 8, im_dim),
            t.nn.Sigmoid(),
        )

    def forward(self, batch_noise):
        return self.gen(batch_noise)


def disc_block(input_dim, output_dim):
    return t.nn.Sequential(t.nn.Linear(input_dim, output_dim), t.nn.LeakyReLU(0.2))


class Discriminator(t.nn.Module):
    def __init__(self, im_dim, hidden_dim):
        super().__init__()
        self.disc = t.nn.Sequential(
            disc_block(im_dim, hidden_dim * 4),
            disc_block(hidden_dim * 4, hidden_dim * 2),
            disc_block(hidden_dim * 2, hidden_dim),
            t.nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_images):
        return self.disc(batch_images)


def training_loop(gen, gen_optim, disc, disc_optim, loss_fn, traindl):
    for epoch in range(1, N_EPOCHS + 1):
        disc_losses = []
        gen_losses = []
        for real_images, _ in tqdm(traindl):
            curr_batch_size = len(real_images)
            real_images = real_images.view(curr_batch_size, -1)

            # Train the discriminator
            disc_optim.zero_grad()
            z = t.randn(curr_batch_size, Z_DIM)
            fake_images = gen(z)
            disc_fake_outputs = disc(fake_images.detach())
            targets = t.zeros_like(disc_fake_outputs)
            disc_fake_loss = loss_fn(disc_fake_outputs, targets)
            disc_real_outputs = disc(real_images)
            targets = t.ones_like(disc_real_outputs)
            disc_real_loss = loss_fn(disc_real_outputs, targets)
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            # Train the generator
            gen_optim.zero_grad()
            z = t.randn(curr_batch_size, Z_DIM)
            fake_images = gen(z)
            disc_fake_outputs = disc(fake_images)
            targets = t.ones_like(disc_fake_outputs)
            gen_loss = loss_fn(disc_fake_outputs, targets)
            gen_loss.backward()
            gen_optim.step()

            # Metrics
            disc_losses.append(disc_loss.detach().item())
            gen_losses.append(gen_loss.detach().item())

        avg_disc_loss = sum(disc_losses) / len(disc_losses)
        avg_gen_loss = sum(gen_losses) / len(gen_losses)
        print(
            f"Epoch {epoch}: Discriminator Loss: {avg_disc_loss:.4f}, Generator Loss: {avg_gen_loss:.4f}"
        )
        save(gen, epoch)


def main():
    os.makedirs(RUNROOT, exist_ok=True)

    gen = Generator(Z_DIM, GEN_HIDDEN_DIM, IMG_DIM)
    gen_optim = t.optim.Adam(gen.parameters(), lr=LEARNING_RATE)

    disc = Discriminator(IMG_DIM, DISC_HIDDEN_DIM)
    disc_optim = t.optim.Adam(disc.parameters(), lr=LEARNING_RATE)

    loss_fn = t.nn.BCEWithLogitsLoss()

    traindl = t.utils.data.DataLoader(
        tv.datasets.MNIST(root=DATAROOT, transform=tv.transforms.ToTensor()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    training_loop(gen, gen_optim, disc, disc_optim, loss_fn, traindl)


if __name__ == "__main__":
    main()

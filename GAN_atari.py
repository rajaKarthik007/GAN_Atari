import gym
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


DISCR_FILTERS = 64
LATENT_VECTOR_SIZE = 100
GENER_FILTERS = 64
BATCH_SIZE = 32
IMG_SIZE = 64
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 100


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super().__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high), dtype=np.float32)
        
    def observation(self, observation):
        new_obs = cv2.resize(observation, (IMG_SIZE, IMG_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels = input_shape[0], out_channels = DISCR_FILTERS, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = DISCR_FILTERS , out_channels = DISCR_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = DISCR_FILTERS * 2, out_channels = DISCR_FILTERS * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = DISCR_FILTERS * 4, out_channels = DISCR_FILTERS * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = DISCR_FILTERS * 8, out_channels = 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv_pipe(x).view(-1, 1).squeeze(dim=1)
    
class Generator(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels = LATENT_VECTOR_SIZE, out_channels = GENER_FILTERS * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = GENER_FILTERS * 8, out_channels = GENER_FILTERS * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = GENER_FILTERS * 4, out_channels = GENER_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = GENER_FILTERS * 2, out_channels = GENER_FILTERS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = GENER_FILTERS, out_channels = output_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.pipe(x)
    
    
def iterate_batches(envs, batch_size = BATCH_SIZE):
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: np.random.choice(envs), None)
    
    while True:
        e = next(env_gen)
        obs, _, done, terminated, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if done or terminated:    
            e.reset()
            
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')]
    imput_shape = envs[0].observation_space.shape
    
    discriminator_net = Discriminator(imput_shape).to(device)
    generator_net = Generator(imput_shape).to(device)
    
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=generator_net.parameters(), lr=0.001)
    dis_optimizer = optim.Adam(params=discriminator_net.parameters(), lr=0.0001)
    writer = SummaryWriter()
    
    generator_losses = []
    discriminator_losses = []
    iter_no = 0
    
    true_labels = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)
    
    for batch in iterate_batches(envs):
        generator_inp = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
        batch = batch.to(device)
        generator_out = generator_net(generator_inp)
        
        #training discriminator
        dis_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        dis_out_true = discriminator_net(batch)
        dis_out_false = discriminator_net(generator_out.detach())
        dis_loss = objective(dis_out_true, true_labels) + objective(dis_out_false, fake_labels)
        dis_loss.backward()
        dis_optimizer.step()
        discriminator_losses.append(dis_loss.item())
        
        #training generator
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()
        dis_out = discriminator_net(generator_out)
        gen_loss = objective(dis_out, true_labels)
        gen_loss.backward()
        gen_optimizer.step()
        generator_losses.append(gen_loss.item())
        
        iter_no += 1
        
        if iter_no % REPORT_EVERY_ITER == 0:
            print(f"Iter {iter_no}: gen_loss={np.mean(generator_losses):.3e}, dis_loss={np.mean(discriminator_losses):.3e}")
            writer.add_scalar("gen_loss", np.mean(generator_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(discriminator_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", make_grid(generator_out.data[:64], normalize=True), iter_no)
            writer.add_image("real", make_grid(batch.data[:64], normalize=True), iter_no)
        
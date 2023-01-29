import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import wandb

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc= nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # get z dim 
            # output => image ! 
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256,img_dim),
            nn.Tanh(), # normalize inputs to [-1,1] so make outputs [-1,1]
        )
    
    def forward(self, x):
        return self.gen(x)

# HyperParameters etc. 
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1 # 784 
batch_size= 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

fixed_noise = torch.randn( (batch_size, z_dim)).to(device)

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ]
    )


dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()

# Tensorboard log init
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# wandb 
wandb.init(project='01.Vanilla_GAN', sync_tensorboard=True)



for epoch in range(num_epochs):
    print(f"Epochs : {epoch} / {num_epochs}")
    for batch_idx, (real,_) in enumerate(loader): # loader 를 불러올 때, (x,label) 로 불러오지만 GAN 은 label 이 필요없음. 
        real = real.view(-1,784).to(device)
        batch_size =real.shape[0]

        ### Train Discriminator : max log(D(x)) + log(1-D(G(z))) => 이 공식에서는 maximize 가 맞고 
        # => disc_real, disc_fake 의 각 prob 를 BCE 에 넘겨주면서 ,, log 함수의 모양이 바뀐다. 
        # # BCE Definition :  -(y * logq + (1-y) * log(1-q)) 
        # disc_real 은 -(y*log(q)) 함수
        # disc_fake 는 -((1-y) * log(1-q)) 함수
        # BCE 의 정의에 따라서, 함수를 구성할 때, 1에 가까울 수록 수렴하도록 disc_real 을 정의, 
        # 0에 가까울 수록 수렴하도록 disc_fake 를 정의 했다고 할 수 있다. 
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1) # prob 
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # BCEloss(predict, label)
        
        disc_fake = disc(fake).view(-1) # prob  
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) 
        
        lossD = (lossD_real + lossD_fake) /2
        disc.zero_grad()
        lossD.backward(retain_graph=True) # GAN Generator 는 disc 에 의해 1번, gen 에 의해 1번, 한 사이클 안에서, 총 2번 업데이트된다. 
        opt_disc.step()
        
        ### Train Generator : min log(1-D(G(z))) <-> max log(D(G(z))) # 두 표현방식이 사실상 같음. 
        # where the second option of maximizing doesn't suffer from saturating fradients 
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output) )
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx ==0:
            print(
                f"Epoch [{epoch} / {num_epochs} Batch [{batch_idx} / {len(loader)}] \
                    Loss D : {lossD:.4f}, Loss G : {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                
                writer_fake.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step +=1 




        



        
        
         
        
        









    
    
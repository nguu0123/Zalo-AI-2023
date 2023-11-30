from imagen_pytorch import NullUnet, Unet, ImagenTrainer, Imagen, ElucidatedImagen
import random
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from imagen_pytorch import t5
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

df = pd.read_csv('./laion_fahsion_zalo_label.csv')
T5_name = "google/t5-v1_1-base"

class Dataset(Dataset):
    def __init__(self, dataset_path, transforms=None, target_transform=None):
        self.dataset_path = dataset_path
        self.transform = self.transform = T.Compose([
            T.Resize((128,128)),
            T.RandomHorizontalFlip(),
            T.CenterCrop(128),
            T.ToTensor()
        ])
        self.target_transform = target_transform
        self.img_labels = df
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = self.dataset_path + 'images/' + self.img_labels.iloc[idx, 4]
        image = Image.open(img_path).convert('RGB')
        caption = str(self.img_labels.iloc[idx, 5])  
        if self.transform:
            image = self.transform(image)
        image = image.squeeze()
        caption, mask = t5.t5_encode_text([caption], return_attn_mask = True, name=T5_name)
        if len(caption) == 0:
            return None
        caption = torch.squeeze(caption)
        tokens = []
        padded_sentence = torch.zeros(256, 768)
        tokens.append(caption)
        tokens.append(padded_sentence)
        tokens = pad_sequence(tokens, True)
        return (image, tokens[0])


unet1 = Unet(
    dim = 128,
    cond_dim = 512,
    dim_mults = (1, 2, 3, 4),
    num_resnet_blocks = 3,
    attn_dim_head = 64,
    attn_heads = 8,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True),
    memory_efficient = False,
)
unet2 = Unet(
    dim = 128,
    cond_dim = 256, 
    dim_mults = (1, 2, 3, 4),
    num_resnet_blocks = (2, 4, 8, 8),
    attn_dim_head = 64,
    attn_heads = 8,
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True),
    memory_efficient = True,
)
# imagen, which contains the unet above

#google/t5-v1_1-base
imagen = ElucidatedImagen(
    unets = [unet1, unet2],
    image_sizes = (64, 128),
    cond_drop_prob = 0.1,
    num_sample_steps = (128, 64), # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
    sigma_min = 0.002,           # min noise level
    sigma_max = (80, 160),       # max noise level, @crowsonkb recommends double the max noise level for upsampler
    sigma_data = 0.5,            # standard deviation of data distribution
    rho = 7,                     # controls the sampling schedule
    P_mean = -1.2,               # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                 # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
    text_encoder_name=T5_name
)

random_seed = int(random.random() * 1000)
trainer = ImagenTrainer(
    imagen = imagen,
    split_random_seed=random_seed, 
    lr = 5e-5,
    #cosine_decay_max_steps = 1500000,
    #warmup_steps = 500
    #dl_tuple_output_keywords_names=('images', 'texts'),
)


# instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

dataset = Dataset('../data/train/')

batch_size = 4
trainer.add_train_dataset(dataset, batch_size = batch_size)
trainer.load('../saved_model/model.pt')
print('ok')

#text_embeds, mask = t5.t5_encode_text("The latest floor mats for cars", return_attn_mask = True, name=T5_name)
train_unet = 1
valid_losses = []
epoch_number = sys.argv[1]
for i in range(1500):
    loss = trainer.train_step(unet_number = train_unet, max_batch_size = 2*batch_size)
    print(f'loss u{train_unet}: {loss}')
trainer.save('../saved_model/model.pt')
#print(f'Mean valid loss: {mean(valid_losses)}')

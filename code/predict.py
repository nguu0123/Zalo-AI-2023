from imagen_pytorch import Unet, ImagenTrainer, ElucidatedImagen
import random

import pandas as pd
from imagen_pytorch import t5

T5_name = "google/t5-v1_1-base"
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
imagen = ElucidatedImagen(
    unets = [unet1, unet2],
    image_sizes = (64, 256),
    cond_drop_prob = 0.1,
    num_sample_steps = (128, 128),
    sigma_min = 0.002,
    sigma_max = (80, 160),
    sigma_data = 0.5,
    rho = 7,
    P_mean = -1.2,
    P_std = 1.2,
    S_churn = 80,
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
    text_encoder_name=T5_name
)

random_seed = int(random.random() * 1000)
trainer = ImagenTrainer(
    imagen = imagen,
).cuda()

trainer.load('../saved_model/model.pt')
df_test = pd.read_csv('../private/info_processed.csv')
text = list(df_test['eng'])
img = list(df_test['bannerImage'])
batch_size = 4
stop_unet = 2
for i in range(0, len(text), batch_size):
    prompt = text[i: i+batch_size]
    img_name = img[i: i+batch_size]
    info_embeds, mask = t5.t5_encode_text(prompt, return_attn_mask = True, name=T5_name)
    images = trainer.sample(batch_size = batch_size, text_embeds=info_embeds, return_pil_images = True, stop_at_unet_number=stop_unet) # returns List[Image]

    for j in range(batch_size):
        images[j] = images[j].resize((1024,533))
        images[j].save('../results/submission1/' + img_name[j])

for i in range(0, len(text), batch_size):
    prompt = text[i: i+batch_size]
    img_name = img[i: i+batch_size]
    info_embeds, mask = t5.t5_encode_text(prompt, return_attn_mask = True, name=T5_name)
    images = trainer.sample(batch_size = batch_size, text_embeds=info_embeds, return_pil_images = True, stop_at_unet_number=stop_unet) # returns List[Image]

    for j in range(batch_size):
        images[j] = images[j].resize((1024,533))
        images[j].save('../results/submission2/' + img_name[j])

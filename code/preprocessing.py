import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en)

def translate_vi2en(vi_texts: str) -> str:
    input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
    output_ids = model_vi2en.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True,
        max_length=1024,
    )
    en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    return en_texts

df_test = pd.read_csv("../data/private/info.csv")
df_test['caption'] = df_test['moreInfo'].replace(np.nan, '')
df_test['description'] = df_test['moreInfo'].replace(np.nan, '')
df_test['moreInfo'] = df_test['moreInfo'].replace(np.nan, '')

text_cap = list(df_test['caption'])
text_des = list(df_test['description'])
text_more_info = list(df_test['moreInfo'])
batch_size = 8
eng = []
for i in range(0, len(text_cap), batch_size):
    caption = text_cap[i: i+batch_size]
    des = text_des[i: i+batch_size]
    more_info = text_more_info[i: i+batch_size]
    all_info = []
    for j in range(0, len(caption)):
        all_info.append(caption[j] + ". " + des[j] + ". " + more_info[j])
    trans_text = translate_vi2en(all_info)
    eng.extend(trans_text)
df_test['eng'] = eng
df_test.to_csv("../data/private/info_processed.csv")

# whisper 모델 + adapter(PEFT 방식 파인튜닝 결과물)로 하나의 음성파일을 transcribe하는 코드

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import torch
import os

import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

peft_model_id_offline = './06210510'

language = "Korean"
task = "transcribe"

peft_config = PeftConfig.from_pretrained(peft_model_id_offline)

print(peft_config)

base_model_name_or_path = '/home/kist/fast_whisper/openai_whisperLargeV3/models--openai--whisper-large-v3/snapshots/1ecca609f9a5ae2cd97a576a9725bc714c022a93'

model = WhisperForConditionalGeneration.from_pretrained(
    base_model_name_or_path, load_in_8bit=False, device_map = 0
    ).to(device)



model = PeftModel.from_pretrained(model, peft_model_id_offline).to(device) #using adapter


print(peft_config.base_model_name_or_path)


tokenizer = WhisperTokenizer.from_pretrained(base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
#decoder_input_ids = torch.tensor([[1, 1]])*model.config.decoder_start_token_id
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


def transcribe(audio):
    with torch.cuda.amp.autocast():
        text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
        #print(text)
    return text

path1 = './예시음성파일.wav'
rtr = transcribe(path1)
print(rtr)

    













# Whisper inference code

## 개요

Whisper의 3가지 버전 모델을 각각 한 가지 음성에 대해 inference하는 코드입니다.

1. whisper_inference.py는 HuggingFace Whisper 모델을 inference하는데 사용됩니다.

2. peft_whisper_inference.py는 HuggingFace Whisper with PEFT adapter 모델을 inference하는데 사용됩니다.
여기서 PEFT adapter는 PEFT(parameter efficient fine tuning) 방식 파인튜닝의 결과로 생성된 요소로, 파인튜닝의 결과물이라고 생각하시면 되니다.

3. faster_whisper_inference.py는 faster_whisper 기반의 모델을 inference하는데 사용됩니다.

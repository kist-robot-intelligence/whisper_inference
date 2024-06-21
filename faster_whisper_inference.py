# faster_whisper 모델로 하나의 음성파일을 transcribe하는 코드

from faster_whisper import WhisperModel

#model = WhisperModel("large-v3")
model = WhisperModel('/home/kist/Desktop/converter/faster_merged_model_06210955')

segments, info = model.transcribe("./예시음성파일.wav")
for segment in segments:
    print("%s" % (segment.text))


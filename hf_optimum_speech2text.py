import librosa
from transformers import AutoProcessor
from optimum.intel import OVModelForSpeechSeq2Seq

model_id = "openai/whisper-large-v3"
ov_model_id = "C:/Users/chiauho.ong/.cache/huggingface/hub/whisper-large-v3-openvino"    # my convert model using optimum

# convert the model
# model_id = "openai/whisper-large-v3"
# model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
# model.save_pretrained("whisper-large-v3-openvino")

print("Load tokenizer")
tokenizer = AutoProcessor.from_pretrained(model_id)
print("Load model")
model = OVModelForSpeechSeq2Seq.from_pretrained(ov_model_id)


# hf_hub.snapshot_download(model_id, local_dir=model_path)
# Use whisperpipeline for inference. This is currently experimental code

#device = "CPU"
#pipe = openvino_genai.WhisperPipeline(model_path, device)
#print(pipe.generate(raw_speech))

speech, samplerate = librosa.load("voice.wav", sr=16000)
raw_speech = speech.tolist()
input_ids = tokenizer(raw_speech, sampling_rate=16000, return_tensors="pt")
outputs = model.generate(**input_ids, language="en")
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
print("end")

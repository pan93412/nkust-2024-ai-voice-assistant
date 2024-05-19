import os
import tempfile
import time
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import pyaudio
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.playback import play

class Recorder:
  def __init__(self, chunk: int = 1024, sample_format: int = pyaudio.paInt16, slience_seconds: int = 5, silence_thresh: int = -16):
    """
    Initializes the Microphone Recorder object.

    Args:
      chunk (int): 記錄聲音的樣本區塊大小。預設值 1024。
      sample_format (int): 樣本格式，可使用 paFloat32、paInt32、paInt24、paInt16、paInt8、paUInt8、paCustomFormat。預設值 pyaudio.paInt16。
      slience_seconds (int): 靜音多久 (seconds) 就停止錄音。預設值 5 秒。
      slience_thresh (int): 靜音的閾值。預設值 -16。
    """
    self.chunk = chunk
    self.sample_format = sample_format
    self.slience_seconds = slience_seconds
    self.slience_thresh = silence_thresh

  def record_until_slicence(self) -> AudioSegment:
    """錄到靜音為止。"""

    microphone_audio = pyaudio.PyAudio()
    recorded_audio_segment = AudioSegment.empty()

    channels = int(microphone_audio.get_default_input_device_info()["maxInputChannels"])
    fs = int(microphone_audio.get_default_input_device_info()["defaultSampleRate"])

    microstream_stream = microphone_audio.open(
        format=self.sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=self.chunk,
        input=True,
    )

    while len(detect_silence(recorded_audio_segment, min_silence_len=self.slience_seconds*1000, silence_thresh=-16, seek_step=100)) == 0:
      data = microstream_stream.read(self.chunk)
      recorded_audio_segment += AudioSegment(data, sample_width=2, channels=channels, frame_rate=fs)
      time.sleep(1/fs)

    recorded_audio_segment = cast(AudioSegment, recorded_audio_segment[:len(recorded_audio_segment) - self.slience_seconds])

    microstream_stream.stop_stream()
    microstream_stream.close()
    microphone_audio.terminate()

    return recorded_audio_segment


class Transcripter:
  def __init__(self, api_key: str, language: str = "zh"):
    self.client = OpenAI(api_key=api_key)
    self.language = language

  def transcribe(self, audio_segment: AudioSegment) -> str:
    with tempfile.TemporaryFile() as temp_audio_file:
        audio_segment.export(temp_audio_file, format="wav")
        temp_audio_file.seek(0)

        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", file=("recorded.wav", temp_audio_file.read()),
            language=self.language
        )

    return transcription.text


class Speaker:
  def __init__(self, api_token: str):
    self.client = OpenAI(api_key=api_token)

  def tts(self, text: str) -> AudioSegment:
    response = self.client.audio.speech.create(
      model="tts-1",
      voice="nova",
      input=text
    )

    with tempfile.TemporaryFile() as temp_audio_file:
      temp_audio_file.write(response.content)
      temp_audio_file.seek(0)

      return AudioSegment.from_mp3(temp_audio_file)

class Assistant:
  conversation_history: list[ChatCompletionMessageParam] = []

  def __init__(self,
               api_key: str, model_engine: str = "gpt-4o",
               prompt: str = "You are a helpful voice assistant named 'Alex' that answers in Chinese.",
               fallback_message: str = "你方便再說一次嗎？"):
    self.client = OpenAI(api_key=api_key)
    self.model_engine = model_engine
    self.conversation_history.append(
      {"role": "system", "content": prompt},
    )
    self.fallback_message = fallback_message


  def answer(self, user_input: str) -> str:
    self.conversation_history.append(
        {"role": "user", "content": user_input},
    )

    completion = self.client.chat.completions.create(
        model=self.model_engine, messages=self.conversation_history
    )
    self.conversation_history.append(
        {"role": "assistant", "content": completion.choices[0].message.content},
    )

    return completion.choices[0].message.content or self.fallback_message


def start_assist() -> None:
  openai_api_key = os.getenv("OPENAI_API_KEY")
  if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

  recorder = Recorder()
  transcripter = Transcripter(api_key=openai_api_key)
  assistant = Assistant(api_key=openai_api_key)
  speaker = Speaker(api_token=openai_api_key)

  last_user_message = ""
  stop_prompts = ["沒問題了", "不用了", "不用了謝謝", "謝謝"]
  stop_prompts_with_puncs = [f"{prompt}{punc}" for prompt in stop_prompts for punc in ["", "。", "！", "？"]]

  input("Start? type 'enter'.")

  while last_user_message not in stop_prompts_with_puncs:
    recorded = recorder.record_until_slicence()
    if recorded.duration_seconds == 0:
        break  # stop

    last_user_message = transcripter.transcribe(recorded)

    print(f"> {last_user_message}")

    model_response = assistant.answer(last_user_message)
    tts = speaker.tts(model_response)
    print(f"{model_response}")
    play(tts)


def main() -> int:
  start_assist()

  return 0

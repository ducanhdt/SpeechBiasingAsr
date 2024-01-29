from TTS.api import TTS
from TTS.tts.layers.tortoise.audio_utils import load_audio, load_voice, load_voices
# tts = TTS("tts_models/en/vctk/vits")
# tts = TTS("tts_models/en/multi-dataset/tortoise-v2")
tts = TTS("tts_models/en/ljspeech/glow-tts")
# tts = TTS("tts_models/en/vctk/fast_pitch")

# cloning `lj` voice from `TTS/tts/utils/assets/tortoise/voices/lj`
# with custom inference settings overriding defaults.
# tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
#                 file_path="output.wav",
#                 voice_dir="./",
#                 speaker="lj",
#                 num_autoregressive_samples=1,
#                 diffusion_iterations=10)

# # Using presets with the same voice
# tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
#                 file_path="output.wav",
#                 voice_dir="./",
#                 speaker="lj",
#                 preset="ultra_fast")

# Random voice generation
if tts.is_multi_lingual:
    print("Available Language:",tts.languages)
if tts.is_multi_speaker:
    print("Available Speaker:",tts.speakers)
speaker = "default"
# for speaker in tts.speakers[:10]:
tts.tts_to_file(text="antolian",
                file_path=f"tmp/output_{speaker}.wav",
                # speaker=speaker,
                )
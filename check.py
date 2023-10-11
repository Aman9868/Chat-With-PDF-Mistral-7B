import os
import torch
import whisper
import nltk
from nltk.tokenize import word_tokenize
from ttsmms import TTS
from moviepy.editor import VideoFileClip, concatenate_audioclips, AudioFileClip
import tempfile
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment


# Video and output paths
input_video_path = 'C:\\Work\\llama bot\\Credit Card Rewards 101 - Meeting Your Minimum Spend [Video 7 of 7].mp4'
output_video_path = 'output_video.mp4'

# Initialize Whisper and extract text from video
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("medium")
model.to(device)
options = whisper.DecodingOptions(language="en", fp16=False)
results = model.transcribe(input_video_path)
segments = results['segments']

# Calculate the average speaking rate for the entire original video
total_words = sum(len(segment['text'].split()) for segment in segments)
total_duration = sum(segment['end'] - segment['start'] for segment in segments)
average_speaking_rate = total_words / total_duration  # Words per second

# Function to replace text
def replace_text(text, old_word, new_word):
    tokens = word_tokenize(text)
    modified_tokens = [new_word if token.lower() == old_word.lower() else token for token in tokens]
    return " ".join(modified_tokens)

# Function to adjust audio speed
def adjust_speed(audio_path, speed=1.0):
    sound = AudioSegment.from_file(audio_path, format="wav")
    return sound.speedup(playback_speed=speed, crossfade=0)

# Replace text in segments
old_word = "credit"
new_word = "debit"
new_segments = [{
    'text': replace_text(segment['text'], old_word, new_word),
    'start': segment['start'],
    'end': segment['end']
} for segment in segments]

# Extract original audio from video
video = VideoFileClip(input_video_path)
original_audio = video.audio

# Create a list to store the paths of temporary audio files
temp_audio_paths = []

# Create a list to store only the modified audio clips
modified_audio_clips = []

# Splicing audio
for segment in new_segments:
    # Generate synthetic audio
    tts = TTS("voice/eng")
    wav = tts.synthesis(segment['text'])
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    temp_audio_paths.append(temp_audio_path)  # Save the path for later removal
    
    write_wav(temp_audio_path, wav['sampling_rate'], wav['x'])
    
    # Calculate the speaking rate for the synthetic audio
    synth_words = len(segment['text'].split())
    synth_duration = segment['end'] - segment['start']
    synth_speaking_rate = synth_words / synth_duration  # Words per second

    # Calculate the speed factor to match the original audio's average speaking rate
    speed_factor = average_speaking_rate / synth_speaking_rate

    # Adjust the speed of the synthetic audio
    adjusted_audio = adjust_speed(temp_audio_path, speed=speed_factor)
    adjusted_audio.export(temp_audio_path, format="wav")
    new_audio = AudioFileClip(temp_audio_path)
    modified_audio_clips.append(new_audio)  # Save only modified audio clips

# Use only the modified audio clips for concatenation
new_audio = concatenate_audioclips(modified_audio_clips)

# Replace audio in the original video
video = video.set_audio(new_audio)
video.write_videofile(output_video_path)

# Remove temporary audio files after final video has been created
for path in temp_audio_paths:
    os.remove(path)

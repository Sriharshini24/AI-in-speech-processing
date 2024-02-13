#!/usr/bin/env python
# coding: utf-8

# In[15]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

file_path = r"C:\Users\Sriharshini\Downloads\Recording.wav"
speech_signal, sample_rate = librosa.load(file_path, sr=None)

speech_derivative = np.diff(speech_signal) / (1 / sample_rate)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(speech_signal)) / sample_rate, speech_signal, color='black')
plt.title('Original Speech Signal')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(speech_derivative)) / sample_rate, speech_derivative, color='r')
plt.title('First Derivative of Speech Signal')

plt.tight_layout()
plt.show()

def generate_audio(signal, sr):
    display(Audio(signal, rate=sr))

print("Original Speech Signal:")
generate_audio(speech_signal, sample_rate)
print("First Derivative of Speech Signal:")
generate_audio(speech_derivative, sample_rate)


# In[21]:


import librosa
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\Sriharshini\Downloads\Recording.wav"
speech_signal, sample_rate = librosa.load(file_path, sr=None)
delta_f_1 = np.diff(speech_signal)
delta_f_2 = speech_signal[:-2] + speech_signal[2:] - 2 * speech_signal[1:-1]
def zero_crossings(signal):
    return np.where(np.diff(np.sign(signal)))[0]
zero_crossings_1 = zero_crossings(delta_f_1)
zero_crossings_2 = zero_crossings(delta_f_2)
differences_1 = np.diff(zero_crossings_1)
differences_2 = np.diff(zero_crossings_2)
avg_length_speech = np.mean(differences_1)
avg_length_silence = np.mean(differences_2)
print("Average length between consecutive zero crossings for speech regions:", avg_length_speech)
print("Average length between consecutive zero crossings for silence regions:", avg_length_silence)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(delta_f_1)) / sample_rate, delta_f_1, color='r')
plt.title('First Derivative of Speech Signal with Zero Crossings')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)  # Zero line
plt.scatter(zero_crossings_1 / sample_rate, np.zeros_like(zero_crossings_1), color='b', s=5)
plt.show()


# In[8]:


import librosa
import numpy as np
import matplotlib.pyplot as plt

files_1 = [r"E:\New folder (2)\Amrita1.wav",r"E:\New folder (2)\Bangalore1.wav",r"E:\New folder (2)\Cricket1.wav",r"E:\New folder (2)\Project1.wav",r"E:\New folder (2)\Speech_Processing1.wav"]
files_2 = [r"E:\New folder (2)\Amrita.wav",r"E:\New folder (2)\Bangalore.wav",r"E:\New folder (2)\Cricket.wav",r"E:\New folder (2)\Project.wav",r"E:\New folder (2)\Statement.wav"]
words = ['Cricket', 'Speech_Processing', 'Amrita_School_of_Engineering', 'Bangalore', 'Project_teams']
my_word_lengths = []
teammate_word_lengths = []

for word_file in files_1:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    my_word_lengths.append(length_seconds)

for word_file in files_2:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    teammate_word_lengths.append(length_seconds)

print("Lengths of the spoken words MINE:", my_word_lengths)
print("Lengths of the spoken words TeamMate:", teammate_word_lengths)

bar_width = 0.35
index = np.arange(len(words))
plt.figure(figsize=(12, 6))
plt.bar(index - bar_width/2, my_word_lengths, bar_width, label='My Words', color='black')
plt.bar(index + bar_width/2, teammate_word_lengths, bar_width, label="Teammate's Words", color='red')
plt.xlabel('Words')
plt.ylabel('Length (seconds)')
plt.title('Comparison of Spoken Words Length')
plt.xticks(index, words)
plt.legend()

plt.show()


# In[16]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
statement, sr = librosa.load(r"E:\New folder (2)\Statement.wav")

# Display the waveform
plt.figure(figsize=(10, 5))
librosa.display.waveshow(statement, sr=sr, color='black')
plt.title('STATEMENT')
plt.xlabel('Time')
plt.show()

question, sr = librosa.load(r"E:\New folder (2)\Question.wav")
plt.figure(figsize=(10, 5))
librosa.display.waveshow(statement, sr=sr,color='black')
plt.title('QUESTION')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





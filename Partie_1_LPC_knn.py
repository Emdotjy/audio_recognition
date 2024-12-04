import librosa
import os
import numpy as np
input_folder='./digit_dataset'

data_audio = []
data_Fe= []
labels = []
N = 60

for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder):
        continue
     
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
        filepath = os.path.join(subfolder, filename)
        audio, Fe = librosa.load(filepath)
        data_audio.append( audio)
        data_Fe.append(Fe)
        labels.append(dirname)

R = np.zeros(shape = (len(audio),N,N))
for num_audio,audio in enumerate(data_audio):
    Y = np.array([np.std(audio)**2]+[0]*N)
    for i in range(N):
        R[num_audio,0,i] = sum([audio[i+t]*audio[t] for t in range(1,N-i+1)])
        R[num_audio,i,0] = sum([audio[i+t]*audio[t] for t in range(1,N-i+1)])

print(R[1,:,:])





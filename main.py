import librosa 
import os
import numpy as np

def load_audio(file):
    audio, Fe = librosa.load(file)
    return audio, Fe

data_audio = []
data_Fe= []
labels = []
#N = 60

def parcourir_digit_dataset(input_folder):  
    for dirname in os.listdir(input_folder): 
        subfolder = os.path.join(input_folder, dirname) 
        if not os.path.isdir(subfolder):  
            continue 
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]: 
            filepath = os.path.join(subfolder, filename) 
            audio, Fe = load_audio(filepath)
            data_audio.append(audio)
            data_Fe.append(Fe)
            labels.append(dirname)



if __name__ == "__main__":
    file = 'digit_dataset/1/1_theo_47.wav'
    input_folder = 'digit_dataset'
    parcourir_digit_dataset(input_folder)
    audio, Fe = load_audio(file)
    #audio, Fe = transform_audio(file)
    #print(parcourir_digit_dataset(input_folder))
    #print("On obtient un vecteur contenant les valeurs des échantillions", audio)
    #print("Voici la taille de ce vecteur : ", len(audio))
    #print("La fréquence d'échantillonnage est ", Fe, "Hz")


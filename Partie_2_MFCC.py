import librosa
from librosa.feature import mfcc
import os
from get_frame import auto_get_frames
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import random

def load_data(input_folder, sr=16000):
    """
    Loads and processes .wav files from subfolders of `input_folder`. 
    Returns a dictionary: {
        'label1': [frames1, frames2, ...],
        'label2': [frames3, frames4, ...],
        ...
    }

    Arguments:
    -----------
    input_folder: str
        Path to the parent folder containing labeled subfolders.
    sr: int
        Sampling rate to use for loading audio files.
    """
    # This dictionary will map each label (dirname) to a list of frames
    data_by_label = {}

    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        
        audio_Fe_list = []
        # Traverse all .wav files in the current subfolder
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            filepath = os.path.join(subfolder, filename)
            audio, Fe = librosa.load(filepath, sr=sr)
            audio = audio / np.max(np.abs(audio))
            audio_Fe_list.append((audio,Fe))
            
        data_by_label[dirname] = audio_Fe_list

    return data_by_label

def split_train_test(data_list, train_ratio=0.8, randomize=True):
    """
    data_list: une liste d'éléments (ex: [ (audio1, Fe1), (audio2, Fe2), ... ])
    """
    if randomize:
        random.shuffle(data_list)  # Shuffle the list in place
    split_index = int(len(data_list) * train_ratio)
    train = data_list[:split_index]
    test  = data_list[split_index:]
    return train, test


if __name__ == "__main__":

    audio, Fe = librosa.load('1_theo_47.wav')


    input_folder='./digit_dataset'

    ordre_modele = 30

    data_audio = load_data(input_folder)

    mfcc_features = mfcc(y=audio, sr=Fe, n_mfcc=15, win_length=512,
    hop_length=512//2 )
    data_audio = load_data(input_folder)

    label_list = data_audio.keys()
    accuracy = []
    train_mfcc_dict  = dict()
    test_mfcc_dict = dict()
    lengths_train = dict()
    lengths_test = dict()


    for label in label_list:
        train_mfcc = []
        lengths_train[label] = []

        test_mfcc = []

        data_audio_Fe_train,data_audio_Fe_test = split_train_test(data_audio[label])

        data_audio_train = []
        data_Fe_train = []
        for (audio_train, fe_train) in data_audio_Fe_train:
            data_audio_train.append(audio_train)
            data_Fe_train.append(fe_train)

        data_audio_test = []
        data_Fe_test = []
        for (audio_test, fe_test) in data_audio_Fe_test:
            data_audio_test.append(audio_test)
            data_Fe_test.append(fe_test)

        for num_audio,audio_train in enumerate(data_audio_train):
            train_mfcc.append(mfcc(y=audio_train, sr=data_Fe_train[num_audio], n_mfcc=15, win_length=512,hop_length=512//2 ))
            lengths_train[label].append(len(train_mfcc[-1]))

        for num_audio,audio_test in enumerate(data_audio_test):
            test_mfcc.append(mfcc(y=audio_test, sr=data_Fe_test[num_audio], n_mfcc=15, win_length=512,hop_length=512//2 ))
            lengths_test[label].append(len(test_mfcc[-1]))
        
        train_mfcc_dict[label] = np.concatenate(train_mfcc, axis=0) 
        test_mfcc_dict[label] = np.concatenate(test_mfcc, axis=0) 

    #entrainement des modèles
    model_dict = dict()
    for label in label_list:
        model = hmm.GaussianHMM(n_components=4,n_iter=1000)
        model.fit(train_mfcc_dict[label], lengths_train[label]) 
        model_dict[label] = model

    # Prediction et test: 
    confusion_matrix = np.zeros((len(label_list),len(label_list)))
    for x,label_test in enumerate(label_list):

        # On prédit pour chaque classe 
        score = dict()
        for y,label_model in enumerate(label_list):
            score[label] = (model_dict(label_model).score(test_mfcc_dict[label_test],lengths_test[label_test]))
        
        prediction = max(score,key=score.get)
        confusion_matrix[(x,label_model.index(prediction))]


    
    print("\n Matrice de confusion: ")
    print(confusion_matrix)
    # Define class labels
    class_labels = [str(i) for i in range(10)]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add tick marks and labels
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)

    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, int(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    # Label the axes
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()



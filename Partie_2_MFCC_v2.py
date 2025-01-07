import librosa
from librosa.feature import mfcc
import os
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import random


def load_data(input_folder, sr=16000):
    """
    Loads and processes .wav files from subfolders of `input_folder`. 
    Returns a dictionary: {
        'label1': [(audio1, Fe1), (audio2, Fe2), ...],
        'label2': [(audio3, Fe3), (audio4, Fe4), ...],
        ...
    }
    """
    data_by_label = {}

    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        
        audio_Fe_list = []
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            filepath = os.path.join(subfolder, filename)
            audio, Fe = librosa.load(filepath, sr=sr)
            # Normalisation amplitude
            audio = audio / np.max(np.abs(audio))
            audio_Fe_list.append((audio, Fe))
        
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

def split_train_test_by_label(data_dict,train_ratio = 0.8, randomize = True):

    label_list = data_dict.keys()
    train_dict  = dict()
    test_dict = dict()

    for label in label_list:
        data_train,data_test = split_train_test(data_dict[label])
        train_dict[label] = data_train
        test_dict[label] = data_test

    return train_dict, test_dict

def calculate_mfcc_and_length_for_dict(data_audio_Fe_dict):
    mfcc_dict = {}
    length_dict = {}
    for label in data_audio_Fe_dict.keys():


        data_audio = []
        data_Fe_train = []
        for audio,Fe in data_audio_Fe_dict[label]:
            data_audio.append(audio)
            data_Fe_train.append(Fe)

        train_mfcc = []
        lengths= []
        for num_audio, audio_train in enumerate(data_audio):
            mfcc_features = mfcc(y=audio_train, sr=data_Fe_train[num_audio], n_mfcc=15, win_length=512, hop_length=512//2)
            train_mfcc.append(mfcc_features.T)
            lengths.append(len(mfcc_features.T))

        length_dict[label] = lengths
        mfcc_dict[label] = np.concatenate(train_mfcc, axis=0)

    return mfcc_dict, length_dict

if __name__ == "__main__":
    
    input_folder = './digit_dataset'
    sr = 16000
    n_mfcc = 15
    n_components = 4  # Nombre d'états HMM (exemple)

    # 1) Chargement des données
    data_audio = load_data(input_folder, sr=sr)
    
    # 2) Split data into train and test sets by label
    data_audio_Fe_train, data_audio_Fe_test = split_train_test_by_label(data_audio, train_ratio=0.8, randomize=True)

    # 3) Calculate MFCC and lengths for train and test sets
    train_mfcc_dict, lengths_train = calculate_mfcc_and_length_for_dict(data_audio_Fe_train)
    test_mfcc_dict, lengths_test = calculate_mfcc_and_length_for_dict(data_audio_Fe_test)

    # Liste de tous les labels (ex: ['0', '1', '2', ...])
    label_list = list(data_audio.keys())

    # 4) Entraînement des modèles HMM (un par label)
    model_dict = {}
    for label in label_list:
        model = hmm.GaussianHMM(n_components=n_components, n_iter=1000, covariance_type='diag')
        model.fit(train_mfcc_dict[label], lengths_train[label])
        model_dict[label] = model

    # 5) Test instance par instance
    num_labels = len(label_list)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for true_label_idx, true_label in enumerate(label_list):
        for mfcc_mat, nb_frames in zip(test_mfcc_dict[true_label], lengths_test[true_label]):
            best_score = -np.inf
            best_label_idx = None

            for candidate_label_idx, candidate_label in enumerate(label_list):
                model = model_dict[candidate_label]
                if model is None:
                    continue
                score = model.score(mfcc_mat, [nb_frames])
                if score > best_score:
                    best_score = score
                    best_label_idx = candidate_label_idx

            if best_label_idx is not None:
                confusion_matrix[true_label_idx, best_label_idx] += 1

    # Affichage
    print("\nMatrice de confusion (test instance par instance) :")
    print("(Ligne = vrai label, Colonne = label prédit)")
    print(label_list)
    print(confusion_matrix)

    # Plot
    plt.figure(figsize=(8,6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    plt.xticks(np.arange(num_labels), label_list, rotation=45)
    plt.yticks(np.arange(num_labels), label_list)

    thresh = confusion_matrix.max() / 2.
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(j, i, str(confusion_matrix[i, j]),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

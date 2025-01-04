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
    data_list: liste d'éléments (par ex. [(audio1, Fe1), (audio2, Fe2), ...])
    """
    if randomize:
        random.shuffle(data_list)  # mélange in-place

    split_index = int(len(data_list) * train_ratio)
    train = data_list[:split_index]
    test  = data_list[split_index:]
    return train, test


if __name__ == "__main__":
    input_folder = './digit_dataset'
    sr = 16000
    n_mfcc = 15
    n_components = 4  # Nombre d'états HMM (exemple)

    # 1) Chargement des données
    data_audio = load_data(input_folder, sr=sr)
    
    # Liste de tous les labels (ex: ['0', '1', '2', ...])
    label_list = list(data_audio.keys())

    # 2) Création des dict pour train/test : on va stocker
    #    séparément les MFCC de chaque fichier
    train_data = {}  # { label: [ (mfcc_file1, length_file1), (mfcc_file2, length_file2), ... ] }
    test_data  = {}

    for label in label_list:
        # data_audio[label] = liste de (audio, Fe)
        train_list, test_list = split_train_test(data_audio[label], train_ratio=0.8, randomize=True)

        # Crée deux listes (MFCC + length) pour train et test
        train_data[label] = []
        test_data[label]  = []

        # -- Train
        for audio_train, fe_train in train_list:
            mfcc_mat = mfcc(y=audio_train, sr=fe_train, n_mfcc=n_mfcc,
                            win_length=512, hop_length=256)
            # Par défaut shape = (n_mfcc, n_frames)
            mfcc_mat = mfcc_mat.T  # => (n_frames, n_mfcc)
            train_data[label].append((mfcc_mat, mfcc_mat.shape[0]))  # on stocke le (MFCC, length)

        # -- Test
        for audio_test, fe_test in test_list:
            mfcc_mat = mfcc(y=audio_test, sr=fe_test, n_mfcc=n_mfcc,
                            win_length=512, hop_length=256)
            mfcc_mat = mfcc_mat.T
            test_data[label].append((mfcc_mat, mfcc_mat.shape[0]))

    # 3) Entraînement des modèles HMM (un par label)
    #    => On concatène tous les MFCC d'un label, puis on appelle fit(...)
    model_dict = {}
    for label in label_list:
        # On récupère la liste de (mfcc_mat, length)
        train_samples = train_data[label]
        if len(train_samples) == 0:
            # Si aucun fichier train pour ce label, on peut ignorer ou créer un modèle vide
            print(f"[Warning] Pas de données train pour le label {label}")
            model_dict[label] = None
            continue

        # Concaténation
        X_list = []
        lengths_list = []
        for (mfcc_mat, nb_frames) in train_samples:
            X_list.append(mfcc_mat)
            lengths_list.append(nb_frames)

        # On obtient un grand array 2D (somme de tous les frames, n_mfcc)
        X_concatenated = np.concatenate(X_list, axis=0)

        # Création du modèle
        model = hmm.GaussianHMM(n_components=n_components, n_iter=1000, covariance_type='diag')
        model.fit(X_concatenated, lengths_list)
        model_dict[label] = model

    # 4) Test instance par instance
    # Construction de la matrice de confusion
    num_labels = len(label_list)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    # On va itérer sur chaque label "vrai" (true_label)
    for true_label_idx, true_label in enumerate(label_list):

        # Pour chaque fichier test dans test_data[true_label]
        for mfcc_mat, nb_frames in test_data[true_label]:
            # On calcule le score pour chaque label (modèle) possible
            best_score = -np.inf
            best_label_idx = None

            for candidate_label_idx, candidate_label in enumerate(label_list):
                model = model_dict[candidate_label]
                if model is None:
                    # Aucun modèle entraîné pour ce label => skip
                    continue
                # Score HMM
                score = model.score(mfcc_mat, [nb_frames])
                if score > best_score:
                    best_score = score
                    best_label_idx = candidate_label_idx

            # best_label_idx correspond au label prédit
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

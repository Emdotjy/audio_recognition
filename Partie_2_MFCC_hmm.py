import librosa
from librosa.feature import mfcc
import os
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import random


def load_data(input_folder, sr=16000, include_substring=None, exclude_substring=None):
    """
    Charge et traite les fichiers .wav à partir des sous-dossiers de `input_folder`.
    Retourne un dictionnaire : {
        'label1': [(audio1, Fe1), (audio2, Fe2), ...],
        'label2': [(audio3, Fe3), (audio4, Fe4), ...],
        ...
    }

    Arguments:
    -----------
    input_folder: str
        Chemin vers le dossier parent contenant les sous-dossiers étiquetés.
    sr: int
        Fréquence d'échantillonnage à utiliser pour charger les fichiers audio.
    include_substring: str ou None
        Si spécifié, inclure uniquement les fichiers contenant cette sous-chaîne.
    exclude_substring: str ou None
        Si spécifié, exclure les fichiers contenant cette sous-chaîne.
    """
    data_by_label = {}

    # Parcourir chaque répertoire dans le dossier d'entrée
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        
        audio_Fe_list = []
        # Parcourir chaque fichier .wav dans le sous-dossier
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            # Appliquer les filtres include_substring et exclude_substring
            if include_substring and include_substring not in filename:
                continue
            if exclude_substring and exclude_substring in filename:
                continue
            filepath = os.path.join(subfolder, filename)
            audio, Fe = librosa.load(filepath, sr=sr)
            # Normaliser l'amplitude
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

def split_train_test_by_label(data_dict, train_ratio=0.8, randomize=True):
    """
    Sépare les données en ensembles d'entraînement et de test en gardant les proportions pour chaque classe.

    Arguments:
    data_dict -- dictionnaire contenant les données audio par label
    train_ratio -- proportion des données à utiliser pour l'entraînement (par défaut 0.8)
    randomize -- booléen indiquant s'il faut mélanger les données avant de les séparer (par défaut True)

    Retourne:
    train_dict -- dictionnaire contenant les données d'entraînement par label
    test_dict -- dictionnaire contenant les données de test par label
    """
    label_list = data_dict.keys()
    train_dict = dict()
    test_dict = dict()

    # Itérer sur chaque label (clé du dictionnaire)
    for label in label_list:
        # Séparer les données pour ce label en ensembles d'entraînement et de test
        data_train, data_test = split_train_test(data_dict[label], train_ratio, randomize)
        train_dict[label] = data_train
        test_dict[label] = data_test

    return train_dict, test_dict


def calculate_mfcc_and_length_for_dict(data_audio_Fe_dict, n_mfcc, win_length, hop_length):
    """
    Calcule les coefficients cepstraux de fréquences Mel (MFCC) et les longueurs correspondantes 
    pour un dictionnaire d'audios étiquetés. Ces données sont formatées pour être utilisées avec 
    hmmlearn.hmm.GaussianHMM.fit.

    Arguments :
        - data_audio_Fe_dict (dict): Dictionnaire où les clefs sont des labels et les valeurs sont
          des listes de tuples (audio, fréquence d'échantillonnage).
        - n_mfcc (int): Nombre de coefficients MFCC à extraire.
        - win_length (int): Longueur de la fenêtre (en nombre d'échantillons).
        - hop_length (int): Décalage entre les fenêtres successives (en nombre d'échantillons).

    Retourne :
        - mfcc_dict (dict): Dictionnaire des MFCC concaténés pour chaque label.
        - length_dict (dict): Dictionnaire des longueurs (nombre de fenêtres) pour chaque fichier audio.
    """

    mfcc_dict = {}
    length_dict = {}

    # Parcours des labels du dictionnaire d'entrée
    for label in data_audio_Fe_dict.keys():
        
        # Extraction des données audio et fréquences d'échantillonnage associées
        data_audio = []  # Liste pour les fichiers audio de ce label
        data_Fe_train = []  # Liste pour les fréquences d'échantillonnage correspondantes
        for audio, Fe in data_audio_Fe_dict[label]:
            data_audio.append(audio)
            data_Fe_train.append(Fe)

        train_mfcc = []
        lengths = []

        # Calcul des MFCC pour chaque fichier audio
        for num_audio, audio_train in enumerate(data_audio):

            # Calcul des MFCC pour un fichier audio donné
            mfcc_features = mfcc(y=audio_train, sr=data_Fe_train[num_audio], 
                                  n_mfcc=n_mfcc, win_length=win_length, 
                                  hop_length=hop_length)

            # Transpose des MFCC pour que chaque ligne corresponde à une fenêtre
            train_mfcc.append(mfcc_features.T)

            # Stockage de la longueur (nombre de fenêtres) de cet audio
            lengths.append(len(mfcc_features.T))

        length_dict[label] = lengths

        # Concaténation des MFCC pour tous les fichiers audio d'un même label
        mfcc_dict[label] = np.concatenate(train_mfcc, axis=0)

    return mfcc_dict, length_dict

def test_model(model_dict, test_mfcc_dict, lengths_test):
    label_list = list(model_dict.keys())
    confusion_matrix = np.zeros((len(label_list), len(label_list)), dtype=int)
    for true_label_idx, true_label in enumerate(label_list):
        # Calculate cumulative sums of lengths for proper indexing
        cumsum = np.cumsum([0] + lengths_test[true_label][:-1])
        
        # Extract individual samples using cumulative indices
        mfcc_individual_samples = [
            test_mfcc_dict[true_label][start:start + length] 
            for start, length in zip(cumsum, lengths_test[true_label])
        ]
        for id_sample , mfcc_sample in enumerate(mfcc_individual_samples):

            best_score = -np.inf
            best_label_idx = None

            for candidate_label_idx, candidate_label in enumerate(label_list):
                model = model_dict[candidate_label]
                if model is None:
                    continue
                score = model.score(mfcc_sample)
                if score > best_score:
                    best_score = score
                    best_label_idx = candidate_label_idx

            if best_label_idx is not None:
                confusion_matrix[true_label_idx, best_label_idx] += 1
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    return accuracy, confusion_matrix
            
if __name__ == "__main__":
    
    input_folder = './digit_dataset'
    sr = 16000

    n_mfcc = 15
    win_length=512
    hop_length=512//2

    # 1) Chargement des données
    data_audio = load_data(input_folder, sr=sr, exclude_substring='theo')
    
    # 2) Split data into train and valid sets by label
    data_audio_Fe_train, data_audio_Fe_valid = split_train_test_by_label(data_audio, train_ratio=0.8, randomize=True)

    # 3) Calculate MFCC and lengths for train and valid sets
    train_mfcc_dict, lengths_train = calculate_mfcc_and_length_for_dict(data_audio_Fe_train, n_mfcc, win_length, hop_length)
    valid_mfcc_dict, lengths_valid = calculate_mfcc_and_length_for_dict(data_audio_Fe_valid, n_mfcc, win_length, hop_length)
    

    # Liste de tous les labels (ex: ['0', '1', '2', ...])
    label_list = list(data_audio.keys())

    # 4) Validate HMM models with different hyperparameters

#    n_components_list = [2,4,6,8]
#    n_iters = range(500, 5001, 500)
#    accuracies = np.zeros((len(n_components_list), len(n_iters)))
#    for id_component,n_components in enumerate(n_components_list):
#        for id_iter,n_iter in enumerate(n_iters):
#            
#            model_dict = {}
#            for label in label_list:
#                model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, covariance_type='diag')
#                model.fit(train_mfcc_dict[label], lengths_train[label])
#                model_dict[label] = model

#            num_labels = len(label_list)
#            confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

#            accuracy, confusion_matrix = test_model(model_dict, valid_mfcc_dict, lengths_valid)
#            print(f"n_components={n_components}, n_iter={n_iter}, accuracy={accuracy}")
#            accuracies[id_component, id_iter] = accuracy

#    plt.figure(figsize=(8,6))
#    plt.imshow(accuracies, interpolation='nearest', cmap=plt.cm.Blues)
#    plt.title('Validation Accuracy')
#    plt.colorbar()
#    plt.xticks(np.arange(len(n_iters)), n_iters)
#    plt.yticks(np.arange(len(n_components_list)), n_components_list)
#    plt.ylabel('n_components')
#    plt.xlabel('n_iter')
#    plt.tight_layout()
#    plt.show()

    # 5) Test final avec les audio de theo

    input_folder = './digit_dataset'
    sr = 16000

    n_mfcc = 15
    win_length=512
    hop_length=512//2

    data_audio = load_data(input_folder, sr=sr, exclude_substring='theo')
    train_mfcc_dict, lengths_train = calculate_mfcc_and_length_for_dict(data_audio, n_mfcc, win_length, hop_length)


    data_audio_Fe_test = load_data(input_folder, sr=sr, include_substring='theo')
    test_mfcc_dict, lengths_test = calculate_mfcc_and_length_for_dict(data_audio_Fe_test,n_mfcc, win_length, hop_length)

    n_components = 10
    n_iter = 5000

    model_dict = {}
    for label in label_list:
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, covariance_type='diag')
        model.fit(train_mfcc_dict[label], lengths_train[label])
        model_dict[label] = model

    # 5) valid instance par instance
    num_labels = len(label_list)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    accuracy, confusion_matrix = test_model(model_dict, valid_mfcc_dict, lengths_valid)


    print(f"Test accuracy: {accuracy}")
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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



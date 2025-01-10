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
    """
    Teste les modèles HMM sur les données de test en prédisant la classe de chaque échantillon.

    Arguments:
    model_dict -- Dictionnaire des modèles HMM entraînés par label 
    test_mfcc_dict -- Dictionnaire contenant les coefficients MFCC concaténés des fichiers test par label
    lengths_test -- Dictionnaire contenant les longueurs de chaque échantillon MFCC par label

    Retourne:
    accuracy -- Taux de classification correcte (somme diagonale / somme totale de la matrice de confusion)
    confusion_matrix -- Matrice de confusion (lignes = vraies classes, colonnes = classes prédites)
    """
    # Obtenir la liste des labels à partir des clés du dictionnaire des modèles
    label_list = list(model_dict.keys())
    # Initialiser la matrice de confusion
    confusion_matrix = np.zeros((len(label_list), len(label_list)), dtype=int)

    # Pour chaque vrai label
    for true_label_idx, true_label in enumerate(label_list):
        # Calculer les indices de début de chaque échantillon dans la matrice concaténée
        # en faisant la somme cumulée des longueurs des échantillons précédents
        cumsum = np.cumsum([0] + lengths_test[true_label][:-1])
        
        # Extraire chaque échantillon individuel à partir de la matrice concaténée
        # en utilisant les indices de début et les longueurs correspondantes
        mfcc_individual_samples = [
            test_mfcc_dict[true_label][start:start + length] 
            for start, length in zip(cumsum, lengths_test[true_label])
        ]

        # Pour chaque échantillon extrait
        for mfcc_sample in mfcc_individual_samples:

            # Initialiser le meilleur score et l'indice du meilleur label
            best_score = -np.inf
            best_label_idx = None

            # Pour chaque label candidat
            for candidate_label_idx, candidate_label in enumerate(label_list):
                model = model_dict[candidate_label]
                if model is None:
                    continue
                # Calculer le score du modèle sur l'échantillon
                score = model.score(mfcc_sample)
                # Mettre à jour le meilleur score si nécessaire
                if score > best_score:
                    best_score = score
                    best_label_idx = candidate_label_idx

            # Incrémenter la matrice de confusion
            if best_label_idx is not None:
                confusion_matrix[true_label_idx, best_label_idx] += 1

    # Calculer l'accuracy comme le rapport entre les prédictions correctes et le total
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
    
    # 2) Séparation des données en ensembles d'entraînement et de validation
    data_audio_Fe_train, data_audio_Fe_valid = split_train_test_by_label(data_audio, train_ratio=0.8, randomize=True)

    # 3) Calcul des MFCC et longueurs pour les données d'entraînement et de validation
    train_mfcc_dict, lengths_train = calculate_mfcc_and_length_for_dict(data_audio_Fe_train, n_mfcc, win_length, hop_length)
    valid_mfcc_dict, lengths_valid = calculate_mfcc_and_length_for_dict(data_audio_Fe_valid, n_mfcc, win_length, hop_length)
    

    # Liste de tous les labels (ex: ['0', '1', '2', ...])
    label_list = list(data_audio.keys())

    # 4) Entraînement et test des modèles HMM de validation

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
    # Chemin vers le dossier contenant les données audio
    input_folder = './digit_dataset'
    # Fréquence d'échantillonnage standard pour tous les fichiers
    sr = 16000

    # Paramètres pour l'extraction des MFCC
    n_mfcc = 15 
    win_length = 512  
    hop_length = 512//2 

    # Chargement des données d'entraînement (tous les fichiers sauf ceux contenant 'theo')
    data_audio = load_data(input_folder, sr=sr, exclude_substring='theo')
    train_mfcc_dict, lengths_train = calculate_mfcc_and_length_for_dict(data_audio, n_mfcc, win_length, hop_length)

    # Chargement des données de test (uniquement les fichiers contenant 'theo')
    data_audio_Fe_test = load_data(input_folder, sr=sr, include_substring='theo')
    test_mfcc_dict, lengths_test = calculate_mfcc_and_length_for_dict(data_audio_Fe_test, n_mfcc, win_length, hop_length)

    # Paramètres du modèle HMM
    n_components = 10  
    n_iter = 5000 

    # Entraînement d'un modèle HMM pour chaque label
    model_dict = {}
    for label in label_list:
        # Création et entraînement du modèle HMM gaussien
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, covariance_type='diag')
        model.fit(train_mfcc_dict[label], lengths_train[label])
        model_dict[label] = model

    # Initialisation de la matrice de confusion
    num_labels = len(label_list)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    # Test du modèle et calcul de la précision
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

    # Add text annotations to show the number in each cell
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(j, i, str(confusion_matrix[i, j]),
                    horizontalalignment='center',
                    verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



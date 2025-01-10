import librosa
import os
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import inv
from get_frame import auto_get_frames
import time
import random
import matplotlib.pyplot as plt
from numba import njit
from sklearn.metrics import ConfusionMatrixDisplay


def calcul_lpc_frame(f, ordre_modele):
    """
    Calcule les coefficients LPC pour une trame audio donnée.

    Arguments:
    f -- la trame audio
    ordre_modele -- l'ordre du modèle LPC

    Retourne:
    lpc -- les coefficients LPC normalisés
    """
    t_fenetre = len(f)

    # Calculer les autocovariances R(i) pour i allant de 0 à ordre_modele
    R = np.zeros((ordre_modele+1,1))
    for k in range(ordre_modele+1):
        R[k] = np.mean(f[0:t_fenetre-1-k] * f[k:t_fenetre-1])
    
    # Construire la matrice de Toeplitz à partir des autocovariances
    m_R = toeplitz(R)

    # Créer un vecteur avec 1 en première position et 0 ailleurs
    v = np.zeros((ordre_modele+1,1))
    v[0] = 1

    # Résoudre l'équation de Yule-Walker pour obtenir les coefficients LPC
    lpc = np.dot(inv(m_R), v)
    
    # Normaliser les coefficients LPC
    lpc = lpc / lpc[0]

    # Retourner les coefficients LPC sans la première valeur (qui est 1 après normalisation)
    return lpc[1:]

def normalize_lpc_coefficients(lpc_coeffs):
    return lpc_coeffs / np.sqrt(np.sum(lpc_coeffs**2))

def process_audio_to_lpc(data_audio, labels, ordre_modele, batch_size=50, prefix=""):
    """
    Convertit les données audio en coefficients LPC normalisés.
    
    Args:
        data_audio: Liste des données audio à traiter
        labels: Liste des labels correspondants
        ordre_modele: Ordre du modèle LPC
        batch_size: Taille du lot pour les messages de progression
        prefix: Préfixe pour les messages (ex: "train" ou "test")
    
    Returns:
        tuple: (lpc_arrays, processed_labels)
        - lpc_arrays: Liste de tableaux numpy contenant les coefficients LPC
        - processed_labels: Labels correspondant aux données traitées
    """
    index_to_pop = []
    lpc_arrays = []
    
    for i in range(len(data_audio)):

        if  i%batch_size == 0:
            print(f"Calcul des LPC des données {prefix} : {i} / {len(data_audio)}" )
        try:
            # Pré-allouer un tableau pour les coefficients LPC
            lpc_frames = np.zeros((len(data_audio[i]), ordre_modele ))
            for j, frame in enumerate(data_audio[i]):
                lpc_coeffs = normalize_lpc_coefficients(calcul_lpc_frame(frame, ordre_modele))    
                lpc_frames[j] = np.squeeze(lpc_coeffs)
            lpc_arrays.append(lpc_frames)


        except:
            index_to_pop.append(i)
            print(f"Un audio n'est pas pris en compte: le numéro {i} appartenant à la classe {labels[i]}")
    
    # Convertir et nettoyer les labels
    processed_labels = np.array(labels)
    processed_labels = np.delete(processed_labels, index_to_pop)
    
    return lpc_arrays, processed_labels

@njit
def distance_elastique(a, b):
    """
    Calcule la distance élastique entre deux séquences de coefficients LPC.

    Arguments:
    a -- première séquence de coefficients LPC
    b -- deuxième séquence de coefficients LPC

    Retourne:
    tableau_res[len(a)-1, len(b)-1] -- la distance élastique entre les deux séquences
    """
    tableau_res = np.zeros((len(a), len(b)))
    wv = 1.0
    wd = 1.0
    wh = 1.0

    for i in range(len(a)):
        for j in range(len(b)):
            # Calculer la distance locale dij comme la somme des carrés des différences entre les coefficients LPC
            dij = 0.0
            for k in range(len(a[i])):  
                dij += (a[i][k] - b[j][k]) ** 2
                
            if j == 0 and i == 0:
                tableau_res[i, j] = dij
            elif i == 0:
                tableau_res[i, j] = tableau_res[i, j-1] + dij * wh
            elif j == 0:
                tableau_res[i, j] = tableau_res[i-1, j] + dij * wv
            else:
                terme1 = tableau_res[i-1, j] + wv * dij
                terme2 = tableau_res[i-1, j-1] + wd * dij
                terme3 = tableau_res[i, j-1] + wh * dij
                tableau_res[i, j] = min(terme1, terme2, terme3)
                
    # Retourner la distance élastique finale, qui est le coût minimal accumulé pour aligner toute la première séquence sur la seconde
    return tableau_res[len(a)-1, len(b)-1]

@njit
def calculate_distance_matrix(train,test,batch_size):
    
    matrix = np.zeros(( len(test),len(train)))

    for i, lpc_one_audio_train in enumerate(train):
        if i%batch_size == 0:
            print(f"Calcul des distances avec la piste audio d'entrainement n°{i} sur {len(train)}")
        for j, lpc_one_audio_test in enumerate(test):
            matrix[j, i] = distance_elastique(lpc_one_audio_train, lpc_one_audio_test)
    return matrix

def k_min_args(list,k):
    sorted_indices=np.argpartition(list,k)
    return sorted_indices[:k]

def knn_predict(dists, labels_train , k):
    assert len(labels_train)== dists.shape[1]
    predicted_labels= []

    for test_dists in dists:
        nearest_neighbors = k_min_args(test_dists,k)

        #a dictionary for counting the labels of the neighbors
        neighbors_labels = {}
        for neighbor in nearest_neighbors:
            neighbors_labels[labels_train[neighbor]]= neighbors_labels.get(labels_train[neighbor],0)+1
       
        most_neighbor_label = max(neighbors_labels, key=neighbors_labels.get)
        predicted_labels.append(most_neighbor_label)
    return predicted_labels

def split_train_test(lists, train_ratio=0.8, randomize=True):
    """
    lists: liste de listes, par ex. [data_audio, data_Fe, labels].
    On suppose que toutes ces listes ont la même longueur.
    """
    # Longueur commune à toutes les listes
    n = len(lists[0])
    
    # Vérifier qu'elles ont toutes la même taille
    for lst in lists:
        if len(lst) != n:
            raise ValueError("Toutes les listes doivent avoir la même longueur.")
    
    # Générer les indices
    indices = list(range(n))
    
    # Mélanger les indices une seule fois
    if randomize:
        random.shuffle(indices)
    
    # Déterminer l'indice de coupure
    split_index = int(n * train_ratio)
    
    # Construire train_lists et test_lists en appliquant la même permutation
    train_lists = []
    test_lists = []
    
    for lst in lists:
        # lst_shuffled = [lst[i] for i in indices] si on veut vraiment garder la liste initiale intacte
        # ou bien on peut l'appliquer directement en compréhension de liste
        train_lists.append([lst[i] for i in indices[:split_index]])
        test_lists.append([lst[i] for i in indices[split_index:]])
    
    return train_lists, test_lists

def load_data(input_folder, Standart_FE=8000):
    '''
    Cette fonction charge les données audio à partir d'un dossier d'entrée, normalise les audios et extrait les frames.

    Arguments:
    input_folder -- le dossier contenant les sous-dossiers avec les fichiers audio .wav
    Standart_FE -- la fréquence d'échantillonnage à utiliser lors du chargement des fichiers audio (par défaut 8000)

    Retourne:
    data_audio -- une liste contenant les frames extraites de chaque fichier audio
    data_Fe -- une liste contenant les fréquences d'échantillonnage de chaque fichier audio
    labels -- une liste contenant les étiquettes (noms des sous-dossiers) de chaque fichier audio

    Structure des fichiers:
    Le dossier d'entrée doit contenir des sous-dossiers, chacun représentant une classe différente. 
    Chaque sous-dossier doit contenir des fichiers audio au format .wav. 
    Par exemple:
    input_folder/
        classe_1/
           audio1.wav
           audio2.wav
        classe_2/
            audio1.wav
        audio2.wav
    '''
    # Initialiser les listes pour stocker les données extraites, les fréquences d'échantillonnage et les étiquettes
    data_Fe = []
    labels = []
    data_audio = []

    # Parcourir chaque répertoire dans le dossier d'entrée
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        
        # Passer si ce n'est pas un répertoire
        if not os.path.isdir(subfolder):
            continue
        
        # Parcourir chaque fichier .wav dans le sous-dossier
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            filepath = os.path.join(subfolder, filename)
            
            # Charger le fichier audio avec la fréquence d'échantillonnage spécifiée
            audio, Fe = librosa.load(filepath, sr=Standart_FE)
            
            # Normaliser l'audio
            audio = audio / np.max(np.abs(audio))
            
            # Extraire les frames de l'audio
            frames = auto_get_frames(audio, Fe)
            
            # Ajouter les frames extraites, la fréquence d'échantillonnage et l'étiquette aux listes respectives
            data_audio.append(frames)
            data_Fe.append(Fe)
            labels.append(dirname)
    
    # Retourner les listes contenant les données extraites, les fréquences d'échantillonnage et les étiquettes
    return data_audio, data_Fe, labels     

if __name__ == "__main__":
    input_folder = './digit_dataset'

    # La sélection de paramètres a été réalisée au préalable
    ordre_modele = 10
    k = 5

    # Charger les données audio
    data_audio, data_Fe, labels = load_data(input_folder)

    # Diviser les données en ensembles d'entraînement et de test
    used_data, discarded_data = split_train_test([data_audio, data_Fe, labels], train_ratio=1)
    train_data, test_data = split_train_test(used_data, train_ratio=0.92)

    # Extraire les données d'entraînement et de test
    (data_audio_train, data_Fe_train, original_labels_train) = train_data
    (data_audio_test, data_Fe_test, original_labels_test) = test_data
    accuracy = []

    # Convertir les données audio en coefficients LPC
    list_of_audio_as_lpc_list_train, labels_train = process_audio_to_lpc(data_audio_train, original_labels_train, ordre_modele, 200, "train")
    list_of_audio_as_lpc_list_test, labels_test = process_audio_to_lpc(data_audio_test, original_labels_test, ordre_modele, 200, "test")

    # Calculer la matrice des distances
    matrix = calculate_distance_matrix(list_of_audio_as_lpc_list_train, list_of_audio_as_lpc_list_test, 200)

    # Prédire les étiquettes avec k-NN
    predicted_labels = knn_predict(matrix, labels_train, k)
    print(f"longueur des predictions: {len(predicted_labels)}, longueur des data_train : {len(list_of_audio_as_lpc_list_train)}, longueur des test : {len(list_of_audio_as_lpc_list_test)} ")

    # Calculer la précision
    accuracy = sum([(predicted_labels[i] == labels_test[i]) for i in range(min(len(labels_test), len(predicted_labels)))]) / min(len(labels_test), len(predicted_labels))
    print(f"On obtient un taux de classification correcte de {accuracy} %")
    

    confusion_matrix = np.zeros((10,10))
    for  num,prediction in enumerate(predicted_labels):
        confusion_matrix[int(labels_test[num]), int(prediction)] += 1
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

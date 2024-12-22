import librosa
import os
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import inv
from get_frame import auto_get_frames
import time
import random

def calcul_lpc_frame(f, ordre_modele):

    t_fenetre = len(f)

    R = np.zeros((ordre_modele+1,1))
    for k in range(ordre_modele+1):
        R[k] = np.mean(f[0:t_fenetre-1-k]*f[k:t_fenetre-1])
            
    m_R = toeplitz(R)

    v = np.zeros((ordre_modele+1,1))
    v[0] = 1

    lpc = np.dot(inv(m_R),v)
    lpc = lpc/lpc[0]

    return lpc[1:]





def distance_elastique(a,b):
    tableau_res = np.zeros((len(a),len(b)) )
    wv = 1
    wd = 1
    wh = 1
    for i in range(len(a)):
        for j in range(len(b)):
            dij = np.sum([(a[i] - b[j])**2])
            if j==0 and i == 0:
                tableau_res[i,j] = dij
            elif i ==0 :
                tableau_res[i,j] = tableau_res[i,j-1] + dij*wh
            elif j == 0:
                tableau_res[i,j] = tableau_res[i-1,j] + dij*wv
            else:                
                terme1 = tableau_res[i-1,j] + wv*dij     
                terme2 = tableau_res[i-1,j-1] + wd*dij  
                terme3 = tableau_res[i,j-1] + wh*dij  
                tableau_res[i,j] = min(terme1,terme2,terme3)
                
    return tableau_res[len(a)-1,len(b)-1] 

def k_min_args(list,k):
    sorted_indices=np.argpartition(list,k)
    return sorted_indices[:k]

    
def knn_predict(dists, labels_train , k):
    predicted_labels= []

    for test_dists in dists:
        nearest_neighbors = k_min_args(test_dists,k)

        #a dictionary for counting the labels of the neighbors
        neighbors_labels = {}
        for neighbor in nearest_neighbors:
            neighbors_labels[labels_train[neighbor]]= neighbors_labels.get(labels_train[neighbor],0)+1
       
        label_arg = np.argmax(neighbors_labels.values)
        most_neighbor_label = list(neighbors_labels.keys())[label_arg]
        predicted_labels.append(most_neighbor_label)
    return predicted_labels

def split_train_test(lists, train_ratio=0.8, randomize=True):
    train_lists = []
    test_lists = []
    for lst in lists:
        if randomize:
            random.shuffle(lst)  # Shuffle the list in place
        split_index = int(len(lst) * train_ratio)
        train_lists.append(lst[:split_index])
        test_lists.append(lst[split_index:])
    return train_lists, test_lists

def is_list_of_non_empty_lists(obj):
    # Vérifie que l'objet est une liste
    if not isinstance(obj, list):
        return False
    
    # Vérifie que l'objet n'est pas vide
    if len(obj) == 0:
        return False

    # Vérifie que tous les éléments sont des listes non vides
    if not all(isinstance(elem, list) and len(elem) > 0 for elem in obj):
        return False

    return True

if __name__ == "__main__":
    input_folder='./digit_dataset'

    data_audio_lpc = []
    data_Fe= []
    labels = []
    N = 60
    k = 5
    ordre_modele = 20
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            filepath = os.path.join(subfolder, filename)
            audio, Fe = librosa.load(filepath)
            frames = auto_get_frames(filepath)
            data_audio_lpc.append(frames) 
            data_Fe.append(Fe)
            labels.append(dirname)

    used_data, discarded_data = split_train_test([data_audio_lpc,data_Fe,labels], train_ratio=0.1)    
    train_data, test_data = split_train_test(used_data, train_ratio=0.92)
    (data_audio_train, data_Fe_train, labels_train) = train_data
    (data_audio_test, data_Fe_test, labels_test) = test_data
    list_of_audio_as_lpc_list_train = []
    list_of_audio_as_lpc_list_test = []
    index_to_pop = []
    for i in range(len(data_audio_train)):
        if i %50==0:
            print(f"calcul des lpc des données d'entrainement : {i} / {len(data_audio_train)}")
            print(f"on est dans les données de {labels_train[i]}")

        try:
            lpc_list_for_one_audio = []
            for frame in data_audio_train[i]:
                lpc_list_for_one_audio.append(calcul_lpc_frame(frame,ordre_modele))
            list_of_audio_as_lpc_list_train.append(lpc_list_for_one_audio)
        except:
            index_to_pop.append(i)
            print(f"un audio n'est pas pris en compte: la numéro {i} appartenant à la classe {labels_train[i]}")
    for index in index_to_pop:
            labels_train.pop(index) 

    index_to_pop = []
    for i in range(len(data_audio_test)):
        if i%20 == 0: 
            print(f"calcul des lpc des données de test : {i} / {len(data_audio_test)}")

        try:
            lpc_list_for_one_audio = []
            for frame in data_audio_test[i]:
                lpc_list_for_one_audio.append(calcul_lpc_frame(frame,ordre_modele))
            list_of_audio_as_lpc_list_test.append(lpc_list_for_one_audio)
        except:
            index_to_pop.append(i)
            print(f"un audio n'est pas pris en compte: la numéro {i} appartenant à la classe {labels_test[i]}")
    for index in index_to_pop:
            labels_test.pop(index) 

    list_of_audio_as_lpc_list_train_restrained = list_of_audio_as_lpc_list_train
    list_of_audio_as_lpc_list_test_restrained = list_of_audio_as_lpc_list_test

    matrix = np.zeros(( len(list_of_audio_as_lpc_list_test_restrained),len(list_of_audio_as_lpc_list_train_restrained)))
        
    
    for i, lpc_one_audio_train in enumerate(list_of_audio_as_lpc_list_train_restrained):
        print(f"Calcul des distances avec la piste audio d'entrainement n°{i} sur {len(list_of_audio_as_lpc_list_train_restrained)}")
        for j, lpc_one_audio_test in enumerate(list_of_audio_as_lpc_list_test_restrained):
            matrix[j, i] = distance_elastique(lpc_one_audio_train, lpc_one_audio_test)

    predicted_labels = knn_predict(matrix,labels_train,k)
    print(f"longuer des prediciont: {len(predicted_labels)}, longueur des data_train : {len(list_of_audio_as_lpc_list_train_restrained)}, longueur des test : {len(list_of_audio_as_lpc_list_test_restrained)} ")
    accuracy = sum([(predicted_labels[i] == labels_test[i]) for i in range(min(len(labels_test),len(predicted_labels)))])/min(len(labels_test),len(predicted_labels))

    print(f"On obient un taux de classification correcte de {accuracy*100} %")
    print("\n voici un exemple sur les 20 premières valeur/prédicions")

    for i in range(min(20,len(labels_test),len(predicted_labels))):
        print(f"Prédiction = {predicted_labels[i]}, Valeur réelle = {labels_test[i] } ")

    print( "et avant le Knn, premier 20*20 de la matrice des dist ")
    print(matrix[0:20,0:20])                    

    print("voici la première colonne de matrix, suivit des labels"  )
    for i, col  in enumerate(matrix):
        print(labels_train[i], col[i])






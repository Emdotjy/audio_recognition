import librosa
import os
import numpy as np
from scipy.linalg import toeplitz
from numpy.linalg import inv
from get_frame import auto_get_frames


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
    tableau_res = np.array((len(a),len(b)))
    wv = 1
    wd = 1
    wh = 1
    for i in range(len(a)):
        for j in range(len(b)):
            print(a[i])
            dij = np.mean([(a[i] - b[j])**2])
            if j==0 and i == 0:
                tableau_res[i,j] = dij
            elif i ==0 :
                tableau_res[i,j] = tableau_res[i,j-1] + dij
            elif j == 0:
                tableau_res[i,j] = tableau_res[i-1,j] + dij
            else:                
                terme1 = tableau_res[i-1,j] + wv*dij     
                terme2 = tableau_res[i-1,j-1] + wd*dij  
                terme3 = tableau_res[i,j-1] + wh*dij  
                tableau_res[i,j] = min(terme1,terme2,terme3)
                
    return tableau_res[len(a),len(b)] 

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

def split_train_test(lists, train_ratio=0.8):
    train_lists = []
    test_lists = []
    for lst in lists:
        split_index = int(len(lst) * train_ratio)
        train_lists.append(lst[:split_index])
        test_lists.append(lst[split_index:])
    return train_lists, test_lists


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

            data_audio_lpc.append(auto_get_frames(filepath)) 
            data_Fe.append(Fe)
            labels.append(dirname)

    train_data, test_data = split_train_test([data_audio_lpc,data_Fe,labels], train_ratio=0.8)
    (data_audio_train, data_Fe_train, labels_train) = train_data
    (data_audio_test, data_Fe_test, labels_test) = train_data    
    lpc_frames_train = []
    lpc_frames_test = []
    for i in range(len(data_audio_train)):
        lpc_frames_train.append( calcul_lpc_frame(data_audio_train[i],ordre_modele))

    for i in range(len(data_audio_test)):
        lpc_frames_train.append( calcul_lpc_frame(data_audio_test[i],ordre_modele))


    matrix = np.zeros((len(lpc_frames_train), len(lpc_frames_test)))
        
    
    for i, lpc_one_audio_train in enumerate(lpc_frames_train):
        for j, lpc_one_audio_test in enumerate(lpc_frames_test):
            matrix[i, j] = distance_elastique(lpc_one_audio_train, lpc_one_audio_test)
        
    predicted_labels = knn_predict(matrix,labels_train,k)
    assert len(predicted_labels) == len(labels_test)
    accuracy = sum([(predicted_labels[i] == labels_test[i]) for i in range(len(predicted_labels))])
    print(f"On obient un taux de classification correcte de {accuracy*100} %")







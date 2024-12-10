import librosa
import os
import numpy as np
input_folder='./digit_dataset'

data_audio = []
data_Fe= []
labels = []
N = 60


def calcul_lpc_frame(f, ordre_modele):

    t_fenetre = len(f)

    R = np.zeros((ordre_modele+1,1))
    for k in range(ordre_modele+1):
        R[k] = np.mean(f[0:t_fenetre-1-k]*f[k:t_fenetre-1])
            
    m_R = toeplitz(R)

    v = np.zeross((ordre_modele+1,1))
    v[0] = 1

    lpc = np.dot(inv(m_R),v)
    lpc = lpc/lpc[0]

    return lpc[1:]





def distance_elastique(a,b):
    tableau_res = np.array(shape = (len(a),len(b)))
    wv = 1
    wd = 1
    wh = 1
    for i in range(len(a)):
        for j in range(len(b)):
            if i ==0 :
                tableau_res[i,j] = j
            else:
                dij = 1*(np.mean((a[i] - b[j])**2))
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





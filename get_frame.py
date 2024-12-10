import librosa
import numpy as np

def nearest_power_of_2(n):
    """
    Trouve la puissance de 2 la plus proche d'un nombre donné.
    """
    return 2 ** int(np.round(np.log2(n)))

def auto_get_frames(file_path: str):
    """
    Découpe automatiquement un fichier audio en frames.
    La taille des frames, le hop_length et la durée des frames sont choisis en fonction
    de la durée de l'audio et de son taux d'échantillonnage.
    
    Args:
    - file_path (str): Chemin vers le fichier audio à traiter.
    
    Returns:
    - frames (np.ndarray): Un tableau 2D où chaque colonne représente une frame.
    - sr (int): Le taux d'échantillonnage du fichier audio.
    """
    # Charger l'audio
    y, sr = librosa.load(file_path, sr=None)  # y = signal, sr = taux d'échantillonnage
    
    # Calculer la durée totale de l'audio en secondes
    audio_duration = len(y) / sr
    
    # Choisir la durée de la frame en millisecondes en fonction de la durée de l'audio
    if audio_duration < 5:          # Audio court (< 5 s)
        frame_duration_ms = 20
    elif 5 <= audio_duration <= 30: # Audio moyen (5 à 30 s)
        frame_duration_ms = 40
    else:                           # Audio long (> 30 s)
        frame_duration_ms = 50

    # Calculer la taille de la frame et la convertir en puissance de 2
    frame_length = int(frame_duration_ms * sr / 1000)  # Convertir la durée (ms) en échantillons
    frame_length = nearest_power_of_2(frame_length)  # Ajuster à la puissance de 2 la plus proche
    
    # Calculer le hop_length (chevauchement de 75% par défaut)
    hop_length = frame_length // 4  # 25% de la frame (overlap de 75%)
    
    # Découper le signal en frames
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    
    # Afficher des infos
    print(f"Taux d'échantillonnage : {sr} Hz")
    print(f"Durée totale de l'audio : {audio_duration:.2f} secondes")
    print(f"Durée de la frame choisie : {frame_duration_ms} ms")
    print(f"Nombre d'échantillons par frame (frame_length) : {frame_length}")
    print(f"Décalage entre frames (hop_length) : {hop_length}")
    print(f"Nombre total de frames : {frames.shape[1]}")
    
    return frames

if __name__=="__main__":
    # Chemin vers le fichier audio
    file_path = './digit_dataset/0/0_jackson_0.wav'

    # Appeler la fonction pour découper automatiquement en frames
    frames = auto_get_frames(file_path)

    # Afficher des informations sur les frames
    print(f"Nombre total de frames : {frames.shape[1]}")
    print(f"Taille de chaque frame : {frames.shape[0]}")
    print(f"Exemple de la première frame : {frames[:, 0]}")

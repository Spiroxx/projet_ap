import cv2
import os
import numpy as np

def unsharp_mask(image, kernel_size=(5, 5), alpha=1.5, beta=-0.5, gamma=0):
    """
    Applique un masque flou (unsharp mask) à l'image pour la rendre plus nette.
    - image : image d'entrée (BGR)
    - kernel_size : taille du noyau pour le flou gaussien
    - alpha, beta, gamma : paramètres pour cv2.addWeighted
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, gamma)
    return sharpened

# Chemin vers la vidéo MP4
video_path = 'data/vid1.mp4'
# Dossier de sortie pour enregistrer les images
output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS :", fps)
# Ici, on prend une image toutes les secondes (interval = fps)
interval = int(fps)

# Seuil pour la méthode Tenengrad (à ajuster selon vos besoins)
tenengrad_threshold = 50  # valeur indicative

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        # Convertir le frame en niveaux de gris et calculer la variance du gradient (Tenengrad)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        tenengrad_value = np.mean(gradient_magnitude)
        
        if tenengrad_value >= tenengrad_threshold:
            # Appliquer le filtre unsharp mask pour améliorer la netteté
            sharpened_frame = unsharp_mask(frame, kernel_size=(5, 5), alpha=1.5, beta=-0.5, gamma=0)
            output_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_filename, sharpened_frame)
            print(f"Image sauvegardée (Tenengrad={tenengrad_value:.2f}): {output_filename}")
            saved_count += 1
        else:
            print(f"Frame {frame_count} floue (Tenengrad={tenengrad_value:.2f}), image ignorée.")

    frame_count += 1

cap.release()
print("Extraction terminée.")

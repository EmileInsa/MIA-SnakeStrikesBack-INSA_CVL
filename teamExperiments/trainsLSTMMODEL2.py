import os
import numpy as np
import pandas as pd
os.chdir(r'C:\snake2-beta-insa-main/data/teamExperiments')
from attackDopel import*
from tensorflow.keras import*
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
# Étape 1 : Charger les données publiques directement depuis le DataFrame
print("Chargement des données publiques depuis publicData_Tasks34...")
public_data = pd.read_parquet('../publicData/publicDatasetTask3-4.parquet')
public_data = public_data.iloc[:, :7]  # Jours 1-7
print(public_data.values.shape)


#%%
# Étape 2 : Préparer les séquences pour LSTM
# Préparer les séquences pour LSTM en utilisant uniquement les colonnes 1 à 4 comme entrée
# et les colonnes 5 à 7 comme cibles
def create_sequences(data, input_steps, output_steps):
    sequences, targets = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        seq = data[i:i + input_steps, :4]  # Colonnes 1 à 4 (features)
        target = data[i + input_steps, 4:7]  # Colonnes 5 à 7 (targets)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

print("Préparation des séquences...")
sequences, targets = create_sequences(public_data.values, input_steps=200, output_steps=3)

# Vérification des dimensions
print(f"Forme des séquences (features) : {sequences.shape}")  # (nb_samples, 7, 4)
print(f"Forme des cibles (targets) : {targets.shape}")        # (nb_samples, 3)


#%%
# Étape 3 : Construire le modèle LSTM
print("Construction du modèle LSTM...")
model = Sequential()
model.add(Input(shape=(200, 4)))  # 20 pas de temps avec 4 features (colonnes 1 à 4)
model.add(LSTM(128, activation='relu', return_sequences=False))  # LSTM principal
#model.add(Dropout(0.2))  # Ajout d'une régularisation pour éviter le surapprentissage
model.add(Dense(3))  # Prédire les colonnes 5, 6, et 7
model.compile(optimizer='adam', loss='mse')




#%%
# Étape 4 : Entraîner le modèle
print("Entraînement du modèle...")
model.fit(sequences, targets, epochs=20, batch_size=32, validation_split=0.2)

#%%
# Étape 5 : Sauvegarder le modèle
model.save("results/lstm_model_task3.keras")  # Format natif TensorFlow


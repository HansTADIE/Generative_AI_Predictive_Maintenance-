import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC

# --- param ---
INPUT_CSV = r'C:/Users/tadie/OneDrive/Bureau/Cursus Ingénieur/BDTN/Veille technologique/Methode_augmentation_des_donnes/ai4i2020.csv'
OUT_CSV = r'C:/Users/tadie/OneDrive/Bureau/Cursus Ingénieur/BDTN/Veille technologique/Methode_augmentation_des_donnes/ai4i_smoted_preserve_props.csv'
scale_factor = 2.0   # <--- changer ici : 2.0 = doubler la taille totale (exemple)
# ---------------

df = pd.read_csv(INPUT_CSV)

# Conserver toutes les colonnes originales
original_columns = df.columns.tolist()

cat_cols = ['Product ID', 'Type']
num_cols = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
features = cat_cols + num_cols
target = 'Machine failure'

# Colonnes supplémentaires à conserver
additional_cols = [col for col in original_columns if col not in features + [target]]

X = df[features].copy()
y = df[target].astype(int).copy()

# Stocker les données supplémentaires
additional_data = df[additional_cols].copy()

# encodage catégoriel
enc = OrdinalEncoder()
X_cat = enc.fit_transform(X[cat_cols])
X_num = X[num_cols].to_numpy()
X_all = np.hstack([X_cat, X_num])
categorical_feature_indices = list(range(len(cat_cols)))

# --- calcul des cibles pour garder les proportions ---
orig_counts = y.value_counts().to_dict()      # ex. {0:9661, 1:339}
total_orig = len(y)
# proportions exactes
proportions = {cls: count/total_orig for cls, count in orig_counts.items()}

# taille cible totale (arrondie)
target_total = int(round(total_orig * scale_factor))

# comptes désirés par classe (arrondis)
desired_counts = {cls: int(round(proportions[cls] * target_total)) for cls in proportions}

# ajustement pour que la somme soit target_total (corrige erreur d'arrondi)
sum_desired = sum(desired_counts.values())
if sum_desired != target_total:
    # ajouter la différence à la classe majoritaire (ou à n'importe laquelle)
    # on choisit la classe la plus fréquente dans les données d'origine
    maj_cls = max(proportions, key=proportions.get)
    desired_counts[maj_cls] += (target_total - sum_desired)

# s'assurer qu'on ne demande pas moins que l'existant (SMOTE ne fait que sur-échantillonner)
for cls, orig_count in orig_counts.items():
    if desired_counts.get(cls, 0) < orig_count:
        # si la cible est plus petite que l'original on la remet à l'original
        desired_counts[cls] = orig_count

print("Comptes initiaux:", orig_counts)
print("Comptes cibles (après agrandissement) :", desired_counts)

# --- SMOTENC : sampling_strategy accepte un dict {class_label: nb_final} ---
sm = SMOTENC(categorical_features=categorical_feature_indices,
             sampling_strategy=desired_counts,
             random_state=42)

X_res, y_res = sm.fit_resample(X_all, y)

# reconstruction en arrondissant les valeurs catégorielles générées
X_res_cat = X_res[:, :len(cat_cols)]
X_res_num = X_res[:, len(cat_cols):]

# arrondir et clip pour ne pas dépasser les indices de catégories valides
X_res_cat_int = np.rint(X_res_cat).astype(int)

# clip chaque colonne catégorielle entre 0 et n_categories-1 (sécurise inverse_transform)
for col_idx in range(X_res_cat_int.shape[1]):
    max_idx = len(enc.categories_[col_idx]) - 1
    X_res_cat_int[:, col_idx] = np.clip(X_res_cat_int[:, col_idx], 0, max_idx)

X_res_cat_decoded = enc.inverse_transform(X_res_cat_int)

# Reconstruction du DataFrame avec toutes les colonnes
df_res = pd.DataFrame(np.hstack([X_res_cat_decoded, X_res_num]), columns=features)
for c in num_cols:
    df_res[c] = pd.to_numeric(df_res[c])
df_res[target] = y_res.astype(int)

# Gestion des données supplémentaires pour les nouvelles lignes
# Pour les lignes originales, on conserve les valeurs originales
# Pour les nouvelles lignes générées, on initialise avec des valeurs par défaut (0 pour les flags, moyenne pour les numériques, etc.)

# Identifier les indices des nouvelles lignes
original_indices = list(range(len(df)))
new_indices = list(range(len(df), len(df_res)))

# Pour chaque colonne supplémentaire, gérer l'extension
for col in additional_cols:
    if col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:  # Colonnes de flags
        # Pour les nouvelles lignes, initialiser à 0
        new_values = np.zeros(len(new_indices))
    elif df[col].dtype in ['int64', 'float64']:  # Colonnes numériques
        # Pour les nouvelles lignes, utiliser la moyenne des valeurs originales
        new_values = np.full(len(new_indices), df[col].mean())
    else:  # Colonnes catégorielles ou autres
        # Pour les nouvelles lignes, utiliser la première valeur (ou une autre stratégie)
        new_values = np.full(len(new_indices), df[col].iloc[0] if len(df) > 0 else None)
    
    # Combiner les valeurs originales et nouvelles
    combined_values = np.concatenate([df[col].values, new_values])
    df_res[col] = combined_values

# Réorganiser les colonnes dans l'ordre original
df_res = df_res[original_columns]

df_res.to_csv(OUT_CSV, index=False)
print("Taille avant:", df.shape, "après SMOTENC:", df_res.shape)

# Vérification
df1 = pd.read_csv(INPUT_CSV)
df2 = pd.read_csv(OUT_CSV)

print("Distribution dans le fichier d'origine :")
print(df1['Machine failure'].value_counts())
print(f"0: {df1['Machine failure'].value_counts()[0]} échantillons")
print(f"1: {df1['Machine failure'].value_counts()[1]} échantillons")

print("Distribution dans le fichier modifié :")
print(df2['Machine failure'].value_counts())
print(f"0: {df2['Machine failure'].value_counts()[0]} échantillons")
print(f"1: {df2['Machine failure'].value_counts()[1]} échantillons")

# Vérifier que toutes les colonnes sont conservées
print("\nColonnes dans le fichier original:", df1.columns.tolist())
print("Colonnes dans le fichier augmenté:", df2.columns.tolist())
print("Toutes les colonnes sont conservées:", set(df1.columns) == set(df2.columns))
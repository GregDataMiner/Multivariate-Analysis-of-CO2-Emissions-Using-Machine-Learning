# -*- coding: utf-8 -*-

# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc

# Lire les données
# Charger les données depuis un fichier CSV
path_to_csv=".../DataBaseFR.csv""
data = pd.read_csv(path_to_csv, header=0, sep=",", low_memory=False)
print(data.info())



# Étape 1 : Construction du Dataset

# Traiter les cas des données manquantes / "Fiabiliser" les données
# Convertir toutes les colonnes en valeurs numériques
data = data.apply(pd.to_numeric, errors='coerce')
# Interpoler les valeurs manquantes pour garantir la continuité des données
data = data.interpolate(method='linear', limit_direction='forward', axis=0)
# Ajouter une colonne calculée manuellement pour remplacer une variable corrompue
data['GDP_Per_Capita'] = data['GDP'] / data['Population']

# Construire la base de données finale propre
# Sélection des variables pertinentes pour l'analyse
variables = ['Industry_value_added', 'Electric_power_consumption', 'Exports_of_goods_and_services', 'Energy_use','GDP_Per_Capita','CO2_emissions']
data = data[variables]
# Interpoler à nouveau les valeurs manquantes si nécessaire
data = data.interpolate(method='linear', limit_direction='forward', axis=0)

# Décrire / Visualiser les données
# Calculer et afficher la matrice de corrélation
correlation_matrix = data.corr()
print(correlation_matrix['CO2_emissions'].sort_values(ascending=False))
# Visualiser la matrice de corrélation avec une heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matrice de corrélation des variables numériques réduites")

# Analyse
# Diviser en variables explicatives (X) et cible (y)
X = data.drop(columns=['CO2_emissions'])  # Utiliser toutes les variables explicatives
y = data['CO2_emissions']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)




# Étape 2 : Modèles prédictifs

# Régression linéaire
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Forêt aléatoire
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

# Étape 3 : Évaluation des modèles
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}")

evaluate_model(y_test, y_pred_lr, "Régression linéaire")
evaluate_model(y_test, y_pred_rf, "Forêt aléatoire")
evaluate_model(y_test, y_pred_gbr, "Gradient Boosting")

# Visualisation des résultats des prédictions
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true)), y_true, label='Valeurs réelles', alpha=0.7, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, label='Valeurs prédites', alpha=0.7, color='red')
    plt.title(f"Valeurs réelles vs prédictions - {model_name}")
    plt.xlabel("Index")
    plt.ylabel("CO2 Emissions")
    plt.legend()
    plt.show()

# Graphiques pour chaque modèle
plot_predictions(y_test, y_pred_lr, "Régression linéaire")
plot_predictions(y_test, y_pred_rf, "Forêt aléatoire")
plot_predictions(y_test, y_pred_gbr, "Gradient Boosting")

# Analyse avancée avec courbes ROC
def calculate_roc_and_auc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# Binariser les cibles pour les courbes ROC
threshold = y.median()  # Utiliser la médiane comme seuil
y_train_bin = (y_train > threshold).astype(int)  # Haute émission = 1, Basse émission = 0
y_test_bin = (y_test > threshold).astype(int)

# Courbes ROC pour les modèles
fpr_lr, tpr_lr, auc_lr = calculate_roc_and_auc(y_test_bin, y_pred_lr)
fpr_rf, tpr_rf, auc_rf = calculate_roc_and_auc(y_test_bin, y_pred_rf)
fpr_gbr, tpr_gbr, auc_gbr = calculate_roc_and_auc(y_test_bin, y_pred_gbr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Régression linéaire (AUC = {auc_lr:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Forêt aléatoire (AUC = {auc_rf:.2f})")
plt.plot(fpr_gbr, tpr_gbr, label=f"Gradient Boosting (AUC = {auc_gbr:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Hasard (AUC = 0.50)")
plt.title("Courbe ROC des modèles avec toutes les variables explicatives")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

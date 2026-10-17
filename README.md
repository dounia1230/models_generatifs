# TP — Partie 1 : Modèle discriminatif

## Objectif

Dans cette première partie, on utilise un réseau de neurones pour reconnaître des chiffres manuscrits à partir du dataset MNIST. Le but est de comprendre comment entraîner un modèle de classification, puis de vérifier s'il arrive à prédire correctement les chiffres sur des images qu'il n'a pas vues pendant l'entraînement.

## Outils utilisés

- Python
- PyTorch et torchvision
- Matplotlib
- NumPy
- Torcheval (pour le calcul de l'accuracy)

Le code est dans le notebook `lab.ipynb`. Il peut être exécuté sur CPU ou sur GPU si CUDA est disponible.

## 1. Préparation des données

On commence par charger MNIST avec `torchvision`. Le dataset contient des images de chiffres de 0 à 9, en niveaux de gris, de taille 28 × 28 pixels.

Il est séparé en deux ensembles :

- **60 000 images** pour l'entraînement ;
- **10 000 images** pour le test.

Les images sont converties en tenseurs avec `ToTensor()`. On utilise ensuite des `DataLoader` avec des lots de 128 images. Les données d'entraînement sont mélangées à chaque passage.

On affiche aussi quelques images avec leurs labels pour voir à quoi ressemblent les données.

## 2. Modèle de classification

On utilise un réseau entièrement connecté assez simple :

1. `Flatten` transforme l'image 28 × 28 en un vecteur de 784 valeurs.
2. Une couche linéaire passe de 784 à 128 neurones.
3. `ReLU` ajoute une non-linéarité.
4. Une dernière couche linéaire donne 10 sorties, une pour chaque chiffre.

Le modèle apprend à associer une image `x` à son chiffre `y`, autrement dit à approximer **P(y|x)**.

## 3. Entraînement

Pour entraîner le classifieur, on utilise :

- la fonction de perte `CrossEntropyLoss` ;
- l'optimiseur `Adam` avec un learning rate de `0.001` ;
- **3 epochs**.

À chaque batch, le modèle fait une prédiction, calcule la perte, puis met à jour ses paramètres avec la rétropropagation.

La perte moyenne est affichée après chaque epoch pour suivre l'entraînement.

## 4. Évaluation

Une fois l'entraînement terminé, on teste le modèle sur les 10 000 images de test.

On calcule l'**accuracy**, c'est-à-dire le pourcentage d'images pour lesquelles le chiffre prédit correspond au vrai label.

Dans le notebook, cette mesure est d'abord calculée manuellement, puis avec `MulticlassAccuracy` de Torcheval. Les deux méthodes servent à mesurer la même chose.

## Conclusion

Cette première partie permet de mettre en place un classifieur simple avec PyTorch, depuis le chargement des données jusqu'à l'évaluation. On peut suivre l'apprentissage grâce à la loss et mesurer les performances du modèle sur des données de test avec l'accuracy.

## Exécution

Installer les bibliothèques nécessaires :

```bash
pip install torch torchvision matplotlib numpy torcheval notebook
```

Puis ouvrir le notebook :

```bash
jupyter notebook lab.ipynb
```

Exécuter les cellules de la première partie dans l'ordre, jusqu'à la section `Partie 2 : Autoencoders`.

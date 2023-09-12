# Burgers APHYNITY

Ce code est un version modifié de APHYNITY (https://github.com/yuan-yin/APHYNITY) qui permet de le faire fonctionner avec les equations de Burgers.  

------ 

Les fichiers placés dans la racine ainsi que ce placés dans datasets/ sont des fichiers déjà présent dans la version originale d'APHYNITY mais qui ont été adaptés pour les équations de Burgers.  

mylib/	contient des fichiers de type utils qui implémente notamment des fonctionnalité de visualisation, de forçage, des paramètres LES et DNS, des schémas numériques...  


----- 

Pour faire fonctionner le forçage, il faut dupliquer la bibliothèque 'torchdiffeq' en renommant son double 'torchdiffeq2' et remplacer le fichier torchdiffeq2/_impl/solvers.py de la bibliothèque dupliquée par celui présent dans le dossier torchdiffeq de ce repo.  

# CYTECH ING3 - Optimisation Métaheuristiques - FOG/CLOUD Task Offloading.


## Introduction du Sujet
---


## Initialisation

L'initialisation du projet se réalise à travers les étapes suivantes: 

1. Clonage du Projet   
La première étape est de cloner le sujet dans le répertoire de votre choix. Pour ce faire vous pouvez utiliser la commande suivante depuis votre terminal GIT:    
` https://github.com/dorian-roux/ing3-optimisation-metaheuristiques.git `


2. Pour utiliser un environnement conda ayant les libraries requises pour le bon fonctionnement du projet exécutez la commande suivante:  
`conda env create -f envs.yaml`


3. Exécution du Notebook principal `Study_NSGA_II.ipynb`. 
Ce notebook présente les résultats finaux atteints grâce à l'algorithme de NSGA_II sur notre population de données. Le notebook utilise les fonctions présente dans le dossier `utils` avec les scripts suivantes:
- `generateData` qui permet de gérer la génération de données (Noeuds, Machines Virtuelles, Dispositifs/Capteurs, Tâches).
- `taskOffloading` qui permet de construire la partie offloading des différentes tâches et de les assigners aux VMs respectivement FOG et CLOUD.
- `setPopulations` qui permet d'automatiser la création de donnée et l'offloading des tâches.
- `NSGA_II` qui permet d'exécuter l'algorithme NSGA_II depuis nos jeux données. Les différentes fonctions de l'algorithme peuvent être retrouvées telles que **tournament**, **crossover**, **mutation**, etc.
- `displayFigs` qui permet de construire les figures depuis les solutions de l'algorithme NSGA-II. 
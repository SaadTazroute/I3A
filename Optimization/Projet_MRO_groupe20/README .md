
# Frequency Allocation Problem

Ce projet vise à résoudre le problème d'allocation de fréquences, dans une vision de problème d'optimisation sous-contraintes.
Nous avons pu modéliser ceci en sorte de **COP** (Constraint Optimization Problem) et en sous forme de **CSP** (Constraint Satisfaction Problem).

Nous avons résolu les COP avec deux solvers : **ACE** et **CHOCO**.
Le problème CSP est résolu avec **Toulbar2**
Les fichiers **xml** représentent les instances sous forme **XCSP3**.




# Reformulation de nos données

Afin de mieux manipuler les données qu'on dispose, nous simplifions l'accès au fichier JSON à travers le script **reformulate_general.py**

Les fichiers de données utilisés pour la résolution sont des fichiers simplifiés par le biais de cette fonction.

Il faut impérativement l'appliquer avant de résoudre.s 


# Resolution de COP
Pour résoudre le problème sous forme COP, il faut lancer le script **essai_cop_mod2.py**  qui permet de créer l'instance COP à partir de notre modélisation et nos données, et faire la résolution avec un solver choisi.

 - **data** : fichiers json contenant le fichier de données
 - **variant** : définit la fonction objective. low : les plus bas fréquences , band : Bande passante, card : nombre de fréquence disctints
 - **variant**: Ace ou choco
 - **filemane** : fichier log qui contient le résultat de la résolution + temps de résolution et autres info


        ! python3 essai_cop_mod2.py -data=fileinput.json -variant=band -solver=[ace,v]   filename='fileoutput' 
        !with open(filename+"_output.txt", 'w') as f:
        f.write(str(cap)) 

# Resolution de CSP
Pour lancer la résolution du problème CSP, il faut lancer le script toulbar_csp.py

	! python3 toulbar_csp.py -data=fileinput.json filename='fileoutput'





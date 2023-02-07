import sys
import pytoulbar2
import json
import itertools


f = open(sys.argv[1])
data = json.load(f)
f.close()


with open(sys.argv[1]) as json_data:
    data = json.load(json_data)

num_stations = data["num_stations"]
emetteur = data["emetteur"]
recepteur = data["recepteur"]
delta = data["delta"]
region_station = data["region_station"]
nb_region_station = data["nb_region_station"]
nb_max_freq = data["nb_max_freq"]
interfer_stat_x =data["interfer_stat_x"]
interfer_stat_y = data["interfer_stat_y"]
interfer_Delta_xy =data["interfer_Delta_xy"]
liaisons_stat_x = data["liaisons_stat_x"]
liaisons_stat_y = data["liaisons_stat_y"]

nb_stations, nb_regions, nb_liaisons, nb_interfer = len(num_stations), len(nb_max_freq), len(liaisons_stat_x),len(interfer_Delta_xy)

top = 9999 # "infini" ; cout pour la violation des contraintes dures
cout_molle = 1

# create a new empty cost function network
Problem = pytoulbar2.CFN(top)

# ajouter les variables de frequence d'emission 
for i in range(nb_stations):
    Problem.AddVariable('freq_emi_' + str(i), ["f"+str(i) for i in emetteur[i]])
    
# ajouter les variables de frequence de reception 
for i in range(nb_stations):
    Problem.AddVariable('freq_rec_' + str(i), ["f"+str(i) for i in recepteur[i]])


##contrainte entre les freq d'emission et de reception de chaque station : contraintes dures

for i in range(nb_stations):
    emeteur_name, recepteur_name = 'freq_emi_'+str(i), 'freq_rec_'+str(i)
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[emeteur_name], Problem.Variables[recepteur_name]):
        if abs(int(combinaison[0][1:])-int(combinaison[1][1:])) == delta[i]:
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(top) # violation de la contrainte dure
    Problem.AddFunction(scope=[emeteur_name, recepteur_name], costs=diff)
    
    
##contrainte pour éviter l'interference, je considère deux stations 1 et 2 ; emeteur_1 est la freq_eme de 
## 1 et recepteur_2 la freq recepteur de 2. Au départ nous savons considéré cette contrainte comme étant une contrainte dure
##vu que l'énoncé dit que  si deux stations sont proches l’une de l’autre, les frequences utilisees par ces stations "doivent" etre suffisamment espacees pour eviter les interferences. Finalement, nous avons décidé de les considerer comme contrainte molle, voir rapport.

#[ abs(freq_emi[interfer_stat_x[j]] - freq_rec[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)],

for j in range(nb_interfer):
    emetteur_1, recepteur_2 = 'freq_emi_'+str(interfer_stat_x[j]), 'freq_rec_'+str(interfer_stat_y[j])
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[emetteur_1], Problem.Variables[recepteur_2]):
        if abs(int(combinaison[0][1:])-int(combinaison[1][1:])) >= interfer_Delta_xy[j]:
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(cout_molle) # violation de la contrainte dure
    Problem.AddFunction(scope=[emetteur_1, recepteur_2], costs=diff)
    
    
#[ abs(freq_rec[interfer_stat_x[j]] - freq_emi[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)],

for j in range(nb_interfer):
    recepteur_1, emetteur_2 = 'freq_rec_'+str(interfer_stat_x[j]), 'freq_emi_'+str(interfer_stat_y[j])
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[recepteur_1], Problem.Variables[emetteur_2]):
        if abs(int(combinaison[0][1:])-int(combinaison[1][1:])) >= interfer_Delta_xy[j]:
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(cout_molle) # violation de la contrainte dure
    Problem.AddFunction(scope=[recepteur_1, emetteur_2], costs=diff)    
    
    
#[ abs(freq_emi[interfer_stat_x[j]] - freq_emi[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)], 

for j in range(nb_interfer):
    emetteur_1, emetteur_2 = 'freq_emi_'+str(interfer_stat_x[j]), 'freq_emi_'+str(interfer_stat_y[j])
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[emetteur_1], Problem.Variables[emetteur_2]):
        if abs(int(combinaison[0][1:])-int(combinaison[1][1:])) >= interfer_Delta_xy[j]:
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(cout_molle) # violation de la contrainte dure
    Problem.AddFunction(scope=[emetteur_1, emetteur_2], costs=diff)
  
    
#[abs(freq_rec[interfer_stat_x[j]] - freq_rec[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)],

for j in range(nb_interfer):
    recepteur_1, recepteur_2 = 'freq_rec_'+str(interfer_stat_x[j]), 'freq_rec_'+str(interfer_stat_y[j])
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[recepteur_1], Problem.Variables[recepteur_2]):
        if abs(int(combinaison[0][1:])-int(combinaison[1][1:])) >= interfer_Delta_xy[j]:
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(cout_molle) # violation de la contrainte dure
    Problem.AddFunction(scope=[recepteur_1, recepteur_2], costs=diff)

    
##contrainte relative aux liaisons : contraintes dures

#[freq_emi[liaisons_stat_x[j]] == freq_rec[liaisons_stat_y[j]] for j in range(nb_liaisons)],
for j in range(nb_liaisons):
    emetteur_1, recepteur_2 = 'freq_emi_'+str(liaisons_stat_x[j]), 'freq_rec_'+str(liaisons_stat_y[j])
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[emetteur_1], Problem.Variables[recepteur_2]):
        if int(combinaison[0][1:])==int(combinaison[1][1:]):
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(top) # violation de la contrainte dure
    Problem.AddFunction(scope=[emetteur_1, recepteur_2], costs=diff)
    

#[freq_rec[liaisons_stat_x[j]] == freq_emi[liaisons_stat_y[j]] for j in range(nb_liaisons)],
for j in range(nb_liaisons):
    recepteur_1, emetteur_2 = 'freq_rec_'+str(liaisons_stat_x[j]), 'freq_emi_'+str(liaisons_stat_y[j])
    diff = []
    # boucle de produit cartesien des valeurs des domaines des definitions
    for combinaison in itertools.product(Problem.Variables[recepteur_1], Problem.Variables[emetteur_2]):
        if int(combinaison[0][1:])==int(combinaison[1][1:]):
            diff.append(0) # pas de coût s'il n'y a pas violation de la contrainte
        else:
            diff.append(top) # violation de la contrainte dure
    Problem.AddFunction(scope=[recepteur_1, emetteur_2], costs=diff)


## contraintes relatives au nombre maximal de fréquecne par région : contrainte de préférence ou de souhait donc, elle est #molle


Problem.Dump(str(sys.argv[1]).split(".")[0]+ ".cfn")
Problem.Option.FullEAC = False
Problem.Option.showSolutions = True
Problem.Solve()


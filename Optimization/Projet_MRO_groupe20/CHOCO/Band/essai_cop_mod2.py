from pycsp3 import *

##data


num_stations,emetteur,recepteur,delta,region_station, nb_region_station, nb_max_freq,interfer_stat_x,interfer_stat_y, interfer_Delta_xy, liaisons_stat_x, liaisons_stat_y = data 

nb_stations, nb_regions, nb_liaisons, nb_interfer = len(num_stations), len(nb_max_freq), len(liaisons_stat_x),len(interfer_Delta_xy)


## les variables et leurs domaines

freq_emi = VarArray( size = nb_stations, dom = lambda i: emetteur[i])
freq_rec = VarArray( size = nb_stations, dom = lambda i: recepteur[i])


#les contraintes
satisfy (
    ##contrainte entre les freq d'emission et de reception de chaque station
    [abs(freq_emi[i] - freq_rec[i])== delta[i] for i in num_stations], 
    
    
    ##contrainte pour éviter l'interference
    
      #pour chaque interference j, on recupère le numéro des stations qui sont à cette interférence, 
      #grâce à interfer_stat_x[j] et liaisons_stat_y[j]
    
    [ abs(freq_emi[interfer_stat_x[j]] - freq_rec[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)],
    
    [ abs(freq_rec[interfer_stat_x[j]] - freq_emi[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)],
    
    [ abs(freq_emi[interfer_stat_x[j]] - freq_emi[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)], 
    
    [ abs(freq_rec[interfer_stat_x[j]] - freq_rec[interfer_stat_y[j]]) >= interfer_Delta_xy[j] for j in range(nb_interfer)],


    ##contrainte relative aux liaisons
    [freq_emi[liaisons_stat_x[j]] == freq_rec[liaisons_stat_y[j]] for j in range(nb_liaisons)],
    
    [freq_rec[liaisons_stat_x[j]] == freq_emi[liaisons_stat_y[j]] for j in range(nb_liaisons)],
    
    ##contrainte relative au nombre maximal de fréquecne par région
    
    #d'abord, on identifie les fréquences emi, et recep des stations appartenant à la région r : ce sont les argurments de zip; 
    # et on transforme en liste l'itérateur zip avec list
    # Cette liste est une liste de tuples de fréquence emi et recep, et on essaie donc de transformer cette liste de tuples 
    # en une simple liste; en utilisant : [item for sublist in tuples_list for item in sublist]
    
    [ NValues( [item for sublist in list( zip( [freq_emi[i] for i in range(nb_stations) if region_station[i] == r ], [freq_rec[i] for i in range(nb_stations) if region_station[i] == r ] )) for item in sublist] ) <= nb_max_freq[r] for r in range(nb_regions)],
)

##fonction objectif 1
if variant("card"):
    minimize(

        NValues([item for sublist in list(zip([freq_emi[i] for i in range(nb_stations)],[freq_rec[i] for i in range(nb_stations)])) for item in sublist])

    )
    
##fonction objectif 2    
elif variant("low"):
    #mininimize(max(max(freq_emi),max(freq_rec)))
    minimize(max([item for sublist in list(zip([freq_emi[i] for i in range(nb_stations)],[freq_rec[i] for i in range(nb_stations)])) for item in sublist]))
    #maximize([item for sublist in list(zip([freq_emi[i] for i in range(nb_stations)],[freq_rec[i] for i in range(nb_stations)])) for item in sublist])
    
##fonction objectif 3
elif variant("band"): 
    #minimize(max(freq_emi,freq_rec) - min(freq_emi,freq_rec))
    minimize(max(max(freq_emi),max(freq_rec)) - min(min(freq_emi),min(freq_rec)))
    #minimize(max([item for sublist in list(zip([freq_emi[i] for i in range(nb_stations)],[freq_rec[i] for i in range(nb_stations)])) for item in sublist]) - min([item for sublist in list(zip([freq_emi[i] for i in range(nb_stations)],[freq_rec[i] for i in range(nb_stations)])) for item in sublist]))
    #minimize(max([item for sublist in list(zip([freq_emi[i] for i in range(nb_stations)],[freq_rec[i] for i in range(nb_stations)])) for item in sublist]))

    
      


 

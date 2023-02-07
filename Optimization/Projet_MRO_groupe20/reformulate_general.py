import json
import os 
import numpy as np



def reformulate_data(data_path, saving_path = "./"):
    for file in os.listdir(data_path): ##pour chaque fichier json contenu dans  le repertoire data_path
        
        with open(os.path.join(data_path,file)) as json_data: ## ouvrir le fichier sous forme d'un dictionnaire
            my_dict = json.load(json_data)
            
        my_new_dico = {} ## créer un nouveau dictionaire et reformuler les données en mettant tout en forme de liste

        my_new_dico["num_stations"] = [my_dict["stations"][i]["num"] for i in range(len(my_dict["stations"]))]
        my_new_dico["emetteur"] = [my_dict["stations"][i]["emetteur"] for i in range(len(my_dict["stations"]))]

        my_new_dico["recepteur"] = [my_dict["stations"][i]["recepteur"] for i in range(len(my_dict["stations"]))]
        my_new_dico["delta"] = [my_dict["stations"][i]["delta"] for i in range(len(my_dict["stations"]))]
        
        
        num_region = np.unique( np.array([my_dict["stations"][i]["region"] for i in range(len(my_dict["stations"]))]) ).tolist()
        
        
        my_new_dico["region_station"] = [my_dict["stations"][i]["region"] for i in range(len(my_dict["stations"]))]
        my_new_dico["nb_region_station"] = [int(np.sum(np.array(my_new_dico["region_station"])== j)) for j in num_region] ##nombre de stations par région

        my_new_dico["nb_max_freq"] = my_dict["regions"]

        my_new_dico["interfer_stat_x"] = [my_dict["interferences"][i]["x"] for i in range(len(my_dict["interferences"])) ]
        my_new_dico["interfer_stat_y"] = [my_dict["interferences"][i]["y"] for i in range(len(my_dict["interferences"])) ]
        my_new_dico["interfer_Delta_xy"] = [my_dict["interferences"][i]["Delta"] for i in range(len(my_dict["interferences"]))] 

        my_new_dico["liaisons_stat_x"] = [my_dict["liaisons"][i]["x"] for i in range(len(my_dict["liaisons"]))] 
        my_new_dico["liaisons_stat_y"] = [my_dict["liaisons"][i]["y"] for i in range(len(my_dict["liaisons"]))] 
        
        ##sauvegarder le nouveau dictionnaire au format json dans le répertoire saving_path
        with open(os.path.join(saving_path,"reformulate_" + file), 'w') as fp:
            json.dump(my_new_dico, fp)
            
            

example_path = "D:/M2 IAAA/MRO_Modélisation et résolution pour l’optimisation/projet/donnees/donnees_cop"
os.listdir(example_path)

reformulate_data(data_path = example_path, saving_path = "./")

example_path_2 = "D:/M2 IAAA/MRO_Modélisation et résolution pour l’optimisation/projet/donnees/donnees_wcsp"
os.listdir(example_path_2)
            
reformulate_data(data_path = example_path_2, saving_path = "D:/M2 IAAA/MRO_Modélisation et résolution pour l’optimisation/projet/data_wcsp")
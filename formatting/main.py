# coding: utf8
import pickle, operator
import numpy as np
import pandas as pd
import csv_sfr
import concatenate_csv
import csv_stat
import csv_threads
import os

#Parameters
data_directory = './formatting/data/SFR/rawdata'
threads_directory = './formatting/data/SFR/threads'
csv_concatenated = './formatting/data/SFR/csv_concatenated.csv'
id_interventionid_concatenated = './formatting/data/SFR/id_interventionid_concatenated.csv'
threads_concatenated = './formatting/data/SFR/threads_concatenated.csv'
data_file= './formatting/data/SFR/messages_formatted.csv'
data_file_threads ='./formatting/data/SFR/messages_formatted_motifs.csv'
data_final = './formatting/data/SFR/messages_catégories_motifs.csv'
new_directory = './formatting/sfr'

########### Formatting des csv messages ##############
# Take a directory of .csv file, concatenate them, formate them
# to a csv with 4 columns label;sentence;conversation;intervention_id

concatenation = concatenate_csv.concatenate_csv(data_directory)
concatenation.concatenate_csv(csv_concatenated)

formatting = csv_sfr.csv_sfr(csv_concatenated)
formatting.problem_detection(3,lecture = False)
formatting.dataframe_to_csv(data_file)

######## Prise en compte des motifs avecs le fichier Threads ########

concatenation2 = concatenate_csv.concatenate_csv(threads_directory)
concatenation2.concatenate_csv(threads_concatenated)

formatting_id = csv_threads.csv_threads(csv_concatenated)
formatting_id.supprimer_cat(['source_id','intervention_id','id'])
formatting_id.df = formatting_id.df.rename(columns={'id':'first_content_id'})
formatting_id.dataframe_to_csv(id_interventionid_concatenated)
print(formatting_id.df.columns.values)

formatting_threads = csv_threads.csv_threads(threads_concatenated)
formatting_threads.selection_ligne()
formatting_threads.supprimer_cat(['first_content_id','custom_motif','custom_code_motif','close_cause'])
formatting_threads.fusion_csv(id_interventionid_concatenated,'first_content_id','inner')
formatting_threads.dataframe_to_csv(data_file_threads)


########### Formatting final ##################

formatting_final = csv_threads.csv_threads(data_file_threads)
formatting_final.fusion_csv(data_file,'intervention_id','outer')
formatting_final.dataframe_to_csv(data_final)

############## Quelques statistiques possibles ############
# A revoir cette partie
stat = csv_stat.csv_stat(data_file)
stat.list_of_categories()

########################################################

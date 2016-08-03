# coding: utf-8
import pandas as pd
import numpy as np
import csv
import time, operator
import os

class concatenate_csv():


    def __init__(self,dossier):
        self.dossier = dossier

    def concatenate_csv(self,name):
        final_csv=open(name,"a")
        i=0
        for fn in os.listdir(self.dossier):
            if fn.endswith(".csv"):
                i+=1
                print i
                # first file:
                if i==1:
                    with open(self.dossier+'/'+fn,"r") as csv1:
                        print csv1
                        for raw in csv1:
                            final_csv.write(raw)
                # now the rest:
                else:
                    with open(self.dossier+'/'+fn,"r") as csv_other:
                        csv_other.next() # skip the header
                        for raw in csv_other:
                            final_csv.write(raw)
        final_csv.close()
        self.concatenated_csv = name

    def check_concat_work(self):
        conca = pd.read_csv(self.concatenated_csv, delimiter=';',low_memory=False)
        row_count = list()
        for fn in os.listdir(dossier):
            fichier = pd.read_csv(self.dossier+'/'+fn, delimiter=';',low_memory=False)
            lon = fichier.shape[0]
            row_count.append(lon)
            print lon
        print sum(row_count)
        print conca.shape[0]

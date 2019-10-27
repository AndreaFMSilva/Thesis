# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:58:13 2019

@author: Andrea Silva
"""

##############################################################################################################################
#########           CASOS POSITIVOS           #########
from Bio import SeqIO
import random
import os
    
class PositiveCases():
    
    def __init__(self, filename, maximo = 1000, minimo = 20):
        self.filename = filename
        self.maximo = maximo
        self.minimo = minimo
        self.fastas_ids = []
        
    def get_sequence(self, record):
        """
        Returns the record´s sequence
        """
        return str(record.seq)

    def get_fasta_id(self, record):
        """
        Returns the Id of the fasta record
        """
        return record.id
    
    def select_positive_cases(self):
        handle = open("../Data/Positive_cases/"+self.filename, "rU")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        
        for record in records:
            seq = self.get_sequence(record)
            if len(seq) >= 20 and len(seq)<=1000:
                self.fastas_ids.append(self.get_fasta_id(record))
        
        return self.fastas_ids


##EXECUTAR CASOS POSITIVOS:
pos_cases = PositiveCases("tcdb.txt")
pos_cases_ids = pos_cases.select_positive_cases()
print("NUMERO DE CASOS POSITIVOS: ", len(pos_cases_ids))


##############################################################################################################################
#########           CASOS NEGATIVOS           #########

class Negative_Cases():
    def __init__(self, number):
        self.number = number
        self.total_list = []
    
    def Chose_rand(self):
        """
        Returns a list of 13788 randomly chosen negative cases from the 466621 total negative cases 
        """
        self.total_list = list(range(0,443040))
        select=self.number
        random_selected= random.sample(self.total_list,select)
        checked_random_numbers = []
        print("1º RANDOM SELECTED: ", len(random_selected))#, random_selected)
        
        for num in random_selected:
            self.total_list.remove(num)
        print("ALL DELETED")
        
        outliners = []
        for indice in random_selected:
            seq = self.get_sequence(indice)
            if len(seq)<20 or len(seq)>1000:
                print("OUTLINER")
                outliners.append(indice)
                out = True
                while out:
                    new_indice = random.sample(self.total_list,1)
                    self.total_list.remove(new_indice[0])
#                    print("NEW INDICE")
                    new_indice_seq = self.get_sequence(new_indice[0])
                    if len(new_indice_seq)<20 or len(new_indice_seq)>1000:
                        print("NEW INDICE IS A OUTLINER")
                        out = True
                    else:
                        print("NEW INDICE CHECKED")
                        checked_random_numbers.append(new_indice[0])
                        print(len(checked_random_numbers))
                        out = False
                        
                        
            else:
                checked_random_numbers.append(indice)
                print(len(checked_random_numbers))
        
        print(" CHECKED RANDOM SELECTED DONE: ")#, checked_random_numbers)
        
        return (checked_random_numbers)
    
    def get_sequence(self, indice):
        """
        Returns the record´s sequence
        """
        handle = open("../Data/Negative_cases/uniprot-NOT+transport+NOT+transporter+reviewed_yes+NOT+transmembrane.fasta", "rU")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        
        return str(records[indice].seq)
    
    def Save_Fasta(self, Neg_cases):
        """
        Reads a file containing all the negative cases and writes the fastas
        of those in Neg_cases
        """
        handle = open("../Data/Negative_cases/uniprot-NOT+transport+NOT+transporter+reviewed_yes+NOT+transmembrane.fasta", "rU")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        
        directory="../Data/Dataset"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file=open("../Data/Dataset/negative_cases_WithLimitsOf20and1000.fasta","w")
        seqs_out = []
        for i in Neg_cases:
            file.write(str(records[i].format("fasta")))
            if len(records[i].seq)<20 or len(records[i].seq)>1000:
                seqs_out.append(i)
        print("SEQS OUT: ", seqs_out)
        file.close
        print("FIM")
      
        
##EXECUTAR CASOS NEGATIVOS:
neg_cases = Negative_Cases(len(pos_cases_ids))
random_sel = neg_cases.Chose_rand()
print("NUMERO DE CASOS NEGATIVOS: ", len(random_sel))
neg_cases.Save_Fasta(random_sel)






##############################################################################################################################
#########           CRIAR DATASET           #########
import numpy as np
import pandas as pd
from Bio import Entrez
import re
import requests

class Dataset_Creation():    
    def __init__(self, pos_cases_ids):
        self.dataset=np.array(('Fasta ID','Uniprot Accession', 'Sequence', 'Is transporter?','TCDB ID','Taxonomy Domain'),dtype=object)
        self.pos_cases_ids = pos_cases_ids   
        
    def get_sequence(self,record):
        """
        Returns the record´s sequence
        """
        return str(record.seq)
    
    def get_fasta_id(self,record):
        """
        Returns the Id of the fasta record
        """
        return record.id
    
    def get_uniprot_accession(self,record):
        """
        Returns the Uniprot Acession number of the record
        """
        y=record.id[0:3]
        if y == "gnl": #positive case
            IDs=record.id.split("|")
            ID=IDs[2]
        else: #negative case
            IDs=record.id.split("|")
            ID=IDs[1]
        return ID
    
    def reverse(self,text):
        """
        Returns the text backwards
        ex: input = text; output = txet
        """
        return (text[::-1])
    
    def check_if_transporter(self,record):
        """
        Verifies if the record is a transporter protein (returns 1) or a negative case (returns 0)
        """
        y=record.id[0:3]
        if y == "gnl": #positive case
            return "1"
        else: return "0" #negative case
        
    def get_TCDB_id(self, record):
        y=record.id[0:3]
        if y == "gnl": #positive case
            ID=str(record.id)
            rev=""
            i=len(ID)-1
            while ID[i] != "|":
                rev+= ID[i]
                i-=1
            return str(self.reverse(rev))
        else: 
            return "0"
        
    def get_domain(self,record):
        try:
            Entrez.email = "andreameirelesilva@hotmail.com"
            Uniprot_Asc = self.get_uniprot_accession(record)   #Vai buscar o UniprotID
            print(Uniprot_Asc)
            handle = Entrez.efetch(db="protein", id=Uniprot_Asc, rettype="gb", retmode="text")
            record = SeqIO.read(handle, format="genbank")
            dic=record.annotations
            taxonomy=dic.get("taxonomy")
            dom=taxonomy[0]
            return dom
             
        except:
            try:
                session=requests.Session()
                Uniprot_Asc = self.get_uniprot_accession(record)
                url="http://www.uniprot.org/uniprot/"+str(Uniprot_Asc)
                resp=session.get(url)
                Organism=str(re.findall("OX=\d+",resp.text))
                Org=re.findall('\d+',Organism)
                    
                url2="http://www.uniprot.org/taxonomy/"+Org[0]
                session2=requests.Session()
                resp2=session2.get(url2)   
                job_content=resp2.text
                Bact=re.findall("property\=\"rdfs:subClassOf\">Bacteria<", job_content)
                Eukaryota=re.findall("property\=\"rdfs:subClassOf\">Eukaryota<", job_content)
                Archaea=re.findall("property\=\"rdfs:subClassOf\">Archaea<", job_content)
                if Eukaryota:
                    dom="Eukaryota"
                elif Bact:
                    dom="Bacteria"
                elif (Archaea):
                    dom="Archaea"
                else:
                    dom="Unknown"
                    
                return dom
            except:
                print("Error")
                dom="Unknown"
                return dom
            
    def negative_cases(self):
        print("             NEGATIVE CASES            ")    
        directory="../Data/Datasets/Dataset1"
        if not os.path.exists(directory):
            os.makedirs(directory)        
        handle2 = open("../Data/Dataset/negative_cases_WithLimitsOf20and1000.fasta", "rU")
        records2 = list(SeqIO.parse(handle2, "fasta"))
        handle2.close()
        for record in records2:
            data=np.array((self.get_fasta_id(record),self.get_uniprot_accession(record),self.get_sequence(record),self.check_if_transporter(record),self.get_TCDB_id(record),self.get_domain(record)),dtype=object)
            self.dataset=np.vstack((self.dataset,data))
    
    def positive_cases(self):
        print("             POSITIVE CASES            ")
        handle = open("../Data/Positive_cases/tcdb.txt", "rU")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        for record in records:
            if self.get_fasta_id(record) in self.pos_cases_ids:
                data=np.array((self.get_fasta_id(record),self.get_uniprot_accession(record),self.get_sequence(record),self.check_if_transporter(record),self.get_TCDB_id(record),self.get_domain(record)),dtype=object)
                self.dataset=np.vstack((self.dataset,data))
                
    def create_joined_dataset(self):
        df=pd.DataFrame(self.dataset[1:], index=None, columns=self.dataset[0])
        print("TAMANHO DO DATAFRAME TOTAL: ", len(df))
        df.to_csv("../Data/Dataset/DataDoms_WithLimitsOf20and1000.csv", sep = ",")

dataset_creation = Dataset_Creation(pos_cases_ids)
dataset_creation.negative_cases()
dataset_creation.positive_cases()
dataset_creation.create_joined_dataset()
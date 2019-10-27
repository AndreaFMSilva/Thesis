# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:17:41 2019

@author: Andrea Silva
"""


##############################################################################################################################
#########           FEATURES CREATION           #########
import pickle
import xlrd
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import PhobiusAPI as P_API
import BOMP_API as B_API
import LocTreeAPI as L_API
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
from Bio import Entrez
import requests
import re

class Features_Creation():
    def __init__(self, data):
        self.data = data
    
    def count_aminoacid(self,aminoacid,sequence):
        """
        Returns the number of a given aminoacid in a sequence
        """
        sequence=sequence
        count=sequence.count(aminoacid)
        return count
    
    def create_aminoacid_composition(self): #utilização desta feature em:"Functional discrimination of membrane proteins using machine learning techniques"
        """
        Returns an array with the aminoacid composition of each example in the dataset
        """
        FEATURES=np.array(("Alanine-A","Arginine-R","Asparagine-N","Aspartiv Acid-D","Cysteine-C","Glumanine-Q",
                           "Glutamic acid-E","Glycine-G","Histidine-H","Isoleucine-I","Leucine-L","Lysine-K",
                           "Methionine-M","Phenylalanine-F","Proline-P","Serine-S","Threonine-T","Tryptophan-W",
                           "Tyrosine-Y","Valine-V"))
        
        aminoacids=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        #            0   1   2   3   4   5   6   7   8   9   10  11  12  13  14 15   16  17 18   19
        index = range(len(self.data))
        for row in index:
            sequence=self.data.values[row][3]
            feature=np.array((self.count_aminoacid(aminoacids[0],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[1],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[2],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[3],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[4],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[5],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[6],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[7],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[8],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[9],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[10],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[11],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[12],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[13],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[14],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[15],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[16],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[17],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[18],sequence)/len(sequence),
                              self.count_aminoacid(aminoacids[19],sequence)/len(sequence)))
            FEATURES=np.vstack((FEATURES,feature))
        return FEATURES
    
    def create_aminoacid_physico_chemical_composition(self): 
        """
        Returns an array with the aminoacid composition based on the physico-chemical properties of each example in the dataset
        """
        FEATURES=np.array(("Charged(DEKHR)","Aliphatic(ILV)","Aromatic(FHWY)","Polar(DERKQN)","Neutral(AGHPSTY)","Hydrophobic(CFILMVW)","+charged(KRH)","-charged(DE)","Tiny(ACDGST)","Small(EHILKMNPQV)","Large(FRWY)"))
        
        aminoacids=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        #            0   1   2   3   4   5   6   7   8   9   10  11  12  13  14 15   16  17 18   19
        index = range(len(self.data))
        for row in index:
            sequence=self.data.values[row][3]
            feature=np.array(((self.count_aminoacid(aminoacids[3], sequence)+self.count_aminoacid(aminoacids[6], sequence)+self.count_aminoacid(aminoacids[11], sequence)+self.count_aminoacid(aminoacids[8], sequence)+self.count_aminoacid(aminoacids[1], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[9], sequence)+self.count_aminoacid(aminoacids[10], sequence)+self.count_aminoacid(aminoacids[19], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[13], sequence)+self.count_aminoacid(aminoacids[8], sequence)+self.count_aminoacid(aminoacids[17], sequence)+self.count_aminoacid(aminoacids[18], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[3], sequence)+self.count_aminoacid(aminoacids[6], sequence)+self.count_aminoacid(aminoacids[1], sequence)+self.count_aminoacid(aminoacids[11], sequence)+self.count_aminoacid(aminoacids[5], sequence)+self.count_aminoacid(aminoacids[2], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[0], sequence)+self.count_aminoacid(aminoacids[7], sequence)+self.count_aminoacid(aminoacids[8], sequence)+self.count_aminoacid(aminoacids[14], sequence)+self.count_aminoacid(aminoacids[15], sequence)+self.count_aminoacid(aminoacids[16], sequence)+self.count_aminoacid(aminoacids[18], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[4], sequence)+self.count_aminoacid(aminoacids[13], sequence)+self.count_aminoacid(aminoacids[9], sequence)+self.count_aminoacid(aminoacids[10], sequence)+self.count_aminoacid(aminoacids[12], sequence)+self.count_aminoacid(aminoacids[19], sequence)+self.count_aminoacid(aminoacids[17], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[11], sequence)+self.count_aminoacid(aminoacids[1], sequence)+self.count_aminoacid(aminoacids[8], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[3], sequence)+self.count_aminoacid(aminoacids[6], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[0], sequence)+self.count_aminoacid(aminoacids[4], sequence)+self.count_aminoacid(aminoacids[3], sequence)+self.count_aminoacid(aminoacids[7], sequence)+self.count_aminoacid(aminoacids[15], sequence)+self.count_aminoacid(aminoacids[16], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[6], sequence)+self.count_aminoacid(aminoacids[8], sequence)+self.count_aminoacid(aminoacids[9], sequence)+self.count_aminoacid(aminoacids[10], sequence)+self.count_aminoacid(aminoacids[11], sequence)+self.count_aminoacid(aminoacids[12], sequence)+self.count_aminoacid(aminoacids[2], sequence)+self.count_aminoacid(aminoacids[14], sequence)+self.count_aminoacid(aminoacids[5], sequence)+self.count_aminoacid(aminoacids[19], sequence))/len(sequence),
                              (self.count_aminoacid(aminoacids[13], sequence)+self.count_aminoacid(aminoacids[1], sequence)+self.count_aminoacid(aminoacids[17], sequence)+self.count_aminoacid(aminoacids[18], sequence))/len(sequence)))
            FEATURES=np.vstack((FEATURES,feature))
        return FEATURES
    
    def count_dipeptide(self,dipeptide,sequence):
        """
        Returns the number of a given aminoacid in a sequence
        """
        sequence=sequence
        count=sequence.count(dipeptide)
        return count
    
    def dipeptide_composition(self,sequence):
        aminoacids=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        Dipeptides=[]
        for item in aminoacids:
            pep1=item
            for item in aminoacids:
                pep2=str(pep1)+str(item)
                Dipeptides.append(pep2)
        #print(Dipeptides)
        dipcompo=[]
        for dip in Dipeptides:
            count=self.count_dipeptide(dip, sequence)
            dipcomposition=count/(len(sequence)-1)
            dipcompo.append(dipcomposition)
        
        dipcompotuple=tuple(dipcompo)
        return dipcompotuple
    
    def create_dipeptide_composition(self,): 
        """
        Returns an array with the dipeptide composition of each example in the dataset
        """
        aminoacids=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        Dipeptides=[]
        for item in aminoacids:
            pep1=item
            for item in aminoacids:
                pep2=str(pep1)+str(item)
                Dipeptides.append(pep2)
        Dipeptidestuple=tuple(Dipeptides)
        
        FEATURES=np.array(Dipeptidestuple)
        
        index = range(len(self.data))
        for row in index:
            print("Creating Dipeptide composition. Row:%s"  % row)
            sequence=self.data.values[row][3]
            feature=np.array(self.dipeptide_composition(sequence))
            FEATURES=np.vstack((FEATURES,feature))
        return FEATURES
    
    def create_PhobiusJobIds(self): 
        AllPhobiusJobIds=[]
        index = range(len(self.data))
        for row in index:
            try:
                print (row)
                if row % 500 == 0:
                    time.sleep(180)
                seq=self.data.values[row][3]
                PhobiusId= (P_API.run(email="andreameirelesilva@hotmail.com",
                           format="short",
                           stype="protein",
                           sequence=seq))
                PhobiusId=str(PhobiusId)
                PhobiusId=PhobiusId.strip("b'")
                print(PhobiusId)
                AllPhobiusJobIds.append(PhobiusId)        
            except:
                i=True
                while i:
                    try:
                        print (row)
                        if row % 500 == 0:
                            time.sleep(180)
                        seq=self.data.values[row][3]
                        PhobiusId= (P_API.run(email="andreameirelesilva@hotmail.com",
                                        format="short",
                                        stype="protein",
                                        sequence=seq))
                        PhobiusId=str(PhobiusId)
                        PhobiusId=PhobiusId.strip("b'")
                        print(PhobiusId)
                        AllPhobiusJobIds.append(PhobiusId)
                        i=False
                    except:
                        print("Error on row %s" % row )    
                
        with open("../Data/Dataset/PhobiusJobIds_WithLimitsOf20and1000.txt", 'wb') as f:
            pickle.dump(AllPhobiusJobIds, f)
            
    def create_num_alphahelices_signalpeptide(self, PhobiusJobIds): #utilizacao desta feature em: "TransportTP- A two-phase classification approach for membrane transporter prediction and characterization"
        FEATURES=np.array(("Number of alpha helices","Is a signal peptide present"))
        index = range(len(self.data))
        for row in index:
            print(row)
            try:
                res=P_API.result(job_id=PhobiusJobIds[row],result_type="out")
                 
                AlphaHelices=str(res["TM"])
                numberAlphaHelices=int(AlphaHelices.strip("b'"))
                 
                SignalPeptide=str(res["SP"])
                if SignalPeptide.strip("b'") == "Y" or "found":
                    HaveSignalPeptide=1
                elif SignalPeptide.strip("b'") == "0":
                    HaveSignalPeptide=0
                else:
                    HaveSignalPeptide=0
                    print("Error on row :%s" % row)
                     
                feature=np.array((numberAlphaHelices,HaveSignalPeptide))
             
                FEATURES=np.vstack((FEATURES,feature))
            except:
    
                try:
                    res=P_API.result(job_id=PhobiusJobIds[row],result_type="out")
                    AlphaHelices=str(res["TM"])
                    numberAlphaHelices=int(AlphaHelices.strip("b'"))
                            
                    SignalPeptide=str(res["SP"])
                    if SignalPeptide.strip("b'") == "Y" or "found":
                        HaveSignalPeptide=1
                    elif SignalPeptide.strip("b'") == "0":
                        HaveSignalPeptide=0
                    else:
                        pass
                except:
                    print("Error on row %s: result error" % row)
                
                    try:
                        res=P_API.result(job_id=PhobiusJobIds[row],result_type="out")
                        AlphaHelices=str(res["TM"])
                        numberAlphaHelices=int(AlphaHelices.strip("b'"))
                                
                        SignalPeptide=str(res["SP"])
                        if SignalPeptide.strip("b'") == "Y" or "found":
                            HaveSignalPeptide=1
                        elif SignalPeptide.strip("b'") == "0":
                            HaveSignalPeptide=0
                        else:
                            pass
                                
                    except:
                        numberAlphaHelices=0
                        SignalPeptide=0
                        print("Error on row %s: values =0" % row)
                             
                feature=np.array((numberAlphaHelices,HaveSignalPeptide))
                     
                FEATURES=np.vstack((FEATURES,feature))            
                       
        return FEATURES
    
    def create_BOMPJobIds(self):
        AllBOMPJobIds=[]
        index = range(len(self.data))
        for row in index:
            try:
                print (row)
                if row % 500 == 0:
                    time.sleep(180)
                seq=self.data.values[row][3]
                seq=">SeqName\n"+seq
                BOMPId= (B_API.run(seqs=seq))
                
                print(BOMPId)
                AllBOMPJobIds.append(BOMPId)        
            except:
                i=True
                while i:
                    try:
                        print (row)
                        if row % 500 == 0:
                            time.sleep(180)
                        seq=self.data.values[row][3]
                        seq=">SeqName\n"+seq
                        BOMPId= (B_API.run(seqs=seq))
                
                        print(BOMPId)
                        AllBOMPJobIds.append(BOMPId) 
                        i=False
                    except:
                        print("Error on row %s" % row )    
                
        with open("../Data/Dataset/BOMPJobIds_WithLimitsOf20and1000.txt", 'wb') as f:
            pickle.dump(AllBOMPJobIds, f)
            
    def create_betabarrels(self,BOMPJobIds): 
        FEATURES=np.array(("Number of Beta-barrels"))
        index = range(len(self.data))
        for row in index:
            print(row)
            print(BOMPJobIds[row])
            try:
                res=B_API.result(job_id=BOMPJobIds[row])              
                feature=np.array((res))
            except:
                try:
                    res=B_API.result(job_id=BOMPJobIds[row])
                    feature=np.array((res))
                except:
                    res=0
                    feature=np.array((res))
                    print("Error on row %s: result error" % row)
    
            FEATURES=np.vstack((FEATURES,feature))            
                       
        return FEATURES
    
    def create_LocTree3JobIds(self):
        Entrez.email = "andreameirelesilva@hotmail.com"
        AllLocTree3JobIds=[]
        index = range(len(self.data))
        for row in index:
            try:
                print (row)
                if row % 500 == 0:
                    time.sleep(180)
                fulldom=self.data.values[row][6]
                print(fulldom)
                if fulldom =="Eukaryota":
                    dom ="euka"
                elif fulldom =="Bacteria":
                    dom="bact"
                elif fulldom =="Archaea":
                    dom="arch"
                else:
                    dom="arch"
                print(dom)
                seq=self.data.values[row][3]
                seq=">SeqName\n"+seq
                reqid,LocTree3Id=(L_API.run(domain=dom, email="andreameirelesilva@hotmail.com",sequence=seq))
                print(LocTree3Id)
                print("----------------------------------")
                AllLocTree3JobIds.append(LocTree3Id)   
            except:
                i=True
                while i:
                    try:
                        print (row)
                        if row % 500 == 0:
                            time.sleep(180)
                        fulldom=self.data.values[row][6]
                        if fulldom =="Eukaryota":
                            dom ="euka"
                        elif fulldom =="Bacteria":
                            dom="bact"
                        elif fulldom =="Archaea":
                            dom="arch"
                        else:
                            dom="arch"
                        print(dom)
                        seq=self.data.values[row][3]
                        seq=">SeqName\n"+seq
                        reqid,LocTree3Id=(L_API.run(domain=dom, email="andreameirelesilva@hotmail.com",sequence=seq))
                        print(LocTree3Id)
                        print("----------------------------------")
                        AllLocTree3JobIds.append(LocTree3Id)
                        i=False
                    except:
                        print("Error on row %s" % row )
                        AllLocTree3JobIds.append(LocTree3Id)
                
        with open("../Data/Dataset/Loctree3JobIds_WithLimitsOf20and1000.txt", 'wb') as f:
            pickle.dump(AllLocTree3JobIds, f)
    
    def create_location_prediction(self, Loctree3JobIds):
        FEATURES=np.array((["error","chloroplast","chloroplast membrane","cytosol","endoplasmic reticulum",
                            "endoplasmic reticulum membrane","extra-cellular","ﬁmbrium","golgi apparatus",
                            "golgi apparatus membrane","mitochondrion","mitochondrion membrane","nucleus",
                            "nucleus membrane","outer membrane","periplasmic space","peroxisome",
                            "peroxisome membrane","plasma membrane","plastid","vacuole","vacuole membrane",
                            "secreted","cytoplasm","inner membrane"]))
        index = range(len(self.data))
        #namelist=[]
        for row in index:
            print(row)
            print(Loctree3JobIds[row])
            try:
                res=L_API.result(result_url=Loctree3JobIds[row])
                res=res.lower()
                print(res) 
                if res=="chloroplast":
                    resfinal=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="chloroplast membrane":
                    resfinal=[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="cytosol":
                    resfinal=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="endoplasmic reticulum":
                    resfinal=[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="endoplasmic reticulum membrane":
                    resfinal=[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="extra-cellular":
                    resfinal=[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="ﬁmbrium":
                    resfinal=[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="golgi apparatus":
                    resfinal=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="golgi apparatus membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="mitochondrion":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="mitochondrion membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="nucleus":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="nucleus membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
                elif res =="outer membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
                elif res =="periplasmic space":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
                elif res =="peroxisome":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
                elif res =="peroxisome membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
                elif res =="plasma membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
                elif res =="plastid":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
                elif res =="vacuole":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]  
                elif res =="vacuole membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
                elif res =="secreted":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
                elif res =="cytoplasm":
                    resfinal=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
                elif res =="inner membrane":
                    resfinal=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]   
                else:
                    print("Error in the location name")    
                    resfinal=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                print(resfinal)
                
                feature=np.array((resfinal))
            except:    
                resfinal=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                feature=np.array((resfinal))
                print("Error on row %s: result error" % row)
            
            FEATURES=np.vstack((FEATURES,feature))

        return FEATURES
    
    def create_transporter_related_pfam_domains(self, Transport_Pfam_domains):#utilizacao desta feature em: "TransportTP- A two-phase classification approach for membrane transporter prediction and characterization"
        Entrez.email = "andreameirelesilva@hotmail.com"
        AllPfams=[]
        AllTransporterPfams=[]   
        FEATURES=np.array(("Number of Transporter related Pfam domains"))
        index= range(len(self.data))#(27576)
        for row in index:
            ProteinPfams=[]
            print(row)
            if row % 500 == 0:
                time.sleep(180)
            numPfamDomains=0
            Uniprot_Acc = self.data.values[row][2]
            try:
                try:
                    handle = Entrez.efetch(db="protein", id=Uniprot_Acc, rettype="gb", retmode="text")
                    record = SeqIO.read(handle, format="genbank")
                    dic=record.annotations
                    db_source = (dic.get("db_source"))
                    Pfams= re.findall("(Pfam:.+?),",db_source)
                    print(Pfams)
                    for Pfam in Pfams:
                        PF=Pfam[5:]
                        AllPfams.append(PF)
                        if PF in Transport_Pfam_domains:
                            numPfamDomains+=1
                            ProteinPfams.append(PF)
                    
                except:
                    
                    url="http://www.uniprot.org/uniprot/"+str(Uniprot_Acc)
                    source_code=requests.get(url)
                    plain_text=source_code.text
                    soup=BeautifulSoup(plain_text)
                    links=""
                    for link in soup.findAll("a"):
                        links+=(str(link.get("href"))+",")
                    PfamLinks= re.findall("(http://pfam.xfam.org/family/.{7})",links)
                    UniquePfamLinks=list(set(PfamLinks))  
                    #print(UniquePfamLinks)
                    for PfamLink in UniquePfamLinks:
                        PfamId=PfamLink[-7:]
                        AllPfams.append(PfamId)
                        if PfamId in Transport_Pfam_domains:
                            numPfamDomains+=1
                            ProteinPfams.append(PfamId) 
                    
            except:
                print("Error in row %s" %(row))
            
            AllTransporterPfams.append(ProteinPfams)           
            feature=np.array((numPfamDomains))
            FEATURES=np.vstack((FEATURES,feature))
        
        with open("../Data/Dataset/Protein_Pfams_WithLimitsOf20and1000.txt", 'wb') as f:
            pickle.dump(AllTransporterPfams, f)
        
        with open("../Data/Dataset/Protein_All_Pfams_WithLimitsOf20and1000.txt", 'wb') as g:
            pickle.dump(AllPfams, g)
            
        print(AllTransporterPfams)  
        print(FEATURES)
        return FEATURES
    
    def Pfam_domains(self):
        workbook= xlrd.open_workbook("../Data/Pfam_domains_transporter.xlsx")
        worksheet= workbook.sheet_by_name("Folha1")
        Pfam_domains=[]
        for row in range(worksheet.nrows):
            #print(row)
            EntryText=str(worksheet.row(row))
            Entry=EntryText[7:(len(EntryText)-2)]
            Pfam_domains.append(Entry)
                
        return(Pfam_domains)
        
    def create_features_csv(self):
        # AMINO ACID COMPOSITION
        print("AMINO ACID COMPOSITION")
        amino_composition=self.create_aminoacid_composition()
        CSV_amino_composition=pd.DataFrame(amino_composition[1:], index=None, columns=amino_composition[0])
        CSV_amino_composition.to_csv("../Data/Dataset/Features/Feature_aminoacid_composition_WithLimitsOf20and1000.csv", sep=',')
        
        # AMINO PHYSICO CHEMICAL COMPOSITION
        print("AMINO PHYSICO CHEMICAL COMPOSITION")
        amino_physico_chemical_composition=self.create_aminoacid_physico_chemical_composition()
        CSV_amino_physico_chemical_composition=pd.DataFrame(amino_physico_chemical_composition[1:], index=None, columns=amino_physico_chemical_composition[0])
        CSV_amino_physico_chemical_composition.to_csv("../Data/Dataset/Features/Feature_aminoacid_physico_chemical_composition_WithLimitsOf20and1000.csv", sep=',')    

        #DIPEPTIDE COMPOSITION
        print("DIPEPTIDE COMPOSITION")
        dip_composition=self.create_dipeptide_composition()
        CSV_dip_composition=pd.DataFrame(dip_composition[1:],index=None,columns=dip_composition[0])
        CSV_dip_composition.to_csv("../Data/Dataset/Features/Feature_dipeptide_composition_WithLimitsOf20and1000.csv", sep=',')

        #PFAM DOMAINS
        print("PFAM DOMAINS")
        Transport_Pfam_domains=self.Pfam_domains()        
        PfamDomains=self.create_transporter_related_pfam_domains(Transport_Pfam_domains)        
        CSV_Pfam_domains=pd.DataFrame(PfamDomains[1:], index=None, columns=PfamDomains[0])
        CSV_Pfam_domains.to_csv("../Data/Dataset/Features/Feature_Pfam_domains_WithLimitsOf20and1000.csv",sep=",")
        
        #ALPHA HELICES AND SIGNAL PEPTIDE
        print("ALPHA HELICES AND SIGNAL PEPTIDE")
        self.create_PhobiusJobIds()
        with open("../Data/Dataset/PhobiusJobIds_WithLimitsOf20and1000.txt", 'rb') as f:
            PhobiusJobIds= pickle.load(f)   
        Alphahelices_Signalpeptide=self.create_num_alphahelices_signalpeptide(PhobiusJobIds)      
        CSV_Alphahelices_Signalpeptide=pd.DataFrame(Alphahelices_Signalpeptide[1:],index=None, columns=Alphahelices_Signalpeptide[0])
        CSV_Alphahelices_Signalpeptide.to_csv("../Data/Dataset/Features/Feature_Alphahelices_Signalpeptide_WithLimitsOf20and1000.csv",sep=",")

        #BETA BARRELS
        print("BETA BARRELS")
        self.create_BOMPJobIds()
        with open("../Data/Dataset/BOMPJobIds_WithLimitsOf20and1000.txt", 'rb') as f:
            BOMPJobIds= pickle.load(f)
        BetaBarrels=self.create_betabarrels(BOMPJobIds)
        CSV_BetaBarrels=pd.DataFrame(BetaBarrels[1:],index=None, columns=BetaBarrels[0])
        CSV_BetaBarrels.to_csv("../Data/Dataset/Features/Feature_BetaBarrels_WithLimitsOf20and1000.csv",sep=",")
        
        #LOCATION PREDICTION
        print("LOCATION PREDICTION")
        self.create_LocTree3JobIds()   
        with open("../Data/Dataset/Loctree3JobIds_WithLimitsOf20and1000.txt", 'rb') as f:
            Loctree3JobIds= pickle.load(f)
        LocationPrediction=self.create_location_prediction(Loctree3JobIds)
        CSV_LocationPred=pd.DataFrame(LocationPrediction[1:],index=None, columns=LocationPrediction[0])
        CSV_LocationPred.to_csv("../Data/Dataset/Features/Feature_Location_Prediction_WithLimitsOf20and1000.csv",sep=",")
        print("FIM DA LOCATION PREDICTION")


    
    def create_file_all_features(self):
        print("CRIANDO FILES")
        # AMINO ACID COMPOSITION
        feature1 = pd.read_csv("../Data/Dataset/Features/Feature_aminoacid_composition_WithLimitsOf20and1000.csv")
        ft1=np.array(feature1)
        ft1=np.delete(ft1,0,1)
        np.savetxt("../Data/Dataset/Features/aminoacid_composition_WithLimitsOf20and1000.csv",ft1,delimiter=",")
        
        # AMINO PHYSICO CHEMICAL COMPOSITION
        feature9 = pd.read_csv("../Data/Dataset/Features/Feature_aminoacid_physico_chemical_composition_WithLimitsOf20and1000.csv")
        ft9=np.array(feature9)
        ft9=np.delete(ft9,0,1)
        np.savetxt("../Data/Dataset/Features/aminoacid_physico_chemical_composition_WithLimitsOf20and1000.csv",ft9,delimiter=",")

        #DIPEPTIDE COMPOSITION
        feature11=pd.read_csv("../Data/Dataset/Features/Feature_dipeptide_composition_WithLimitsOf20and1000.csv")
        ft11=np.array(feature11)
        ft11=np.delete(ft11,0,1)
        np.savetxt("../Data/Dataset/Features/dipeptide_composition_WithLimitsOf20and1000.csv",ft11,delimiter=",")
        
        #PFAM DOMAINS
        feature4=pd.read_csv("../Data/Dataset/Features/Feature_Pfam_domains_WithLimitsOf20and1000.csv")
        ft4=np.array(feature4)
        ft4=np.delete(ft4,0,1)
        sc_X = StandardScaler()
        ft4 = sc_X.fit_transform(ft4)
        np.savetxt("../Data/Dataset/Features/Transporter_Pfam_domains_WithLimitsOf20and1000.csv",ft4,delimiter=",")

        #ALPHA HELICES AND SIGNAL PEPTIDE
        feature6=pd.read_csv("../Data/Dataset/Features/Feature_Alphahelices_Signalpeptide_WithLimitsOf20and1000.csv")
        ft6=np.array(feature6)
        ft6=np.delete(ft6,0,1)
        alpha_helc, sig_pep = np.hsplit(ft6, 2)
        #Aplicaçao do StandardScaler
        sc_X = StandardScaler()
        alpha_helc_std = sc_X.fit_transform(alpha_helc)
        ft6 = np.hstack((alpha_helc_std,sig_pep))
        np.savetxt("../Data/Dataset/Features/Alphahelices_Signalpeptide_WithLimitsOf20and1000.csv", ft6,delimiter=",")
        
        #BETA BARRELS
        feature7=pd.read_csv("../Data/Dataset/Features/Feature_BetaBarrels_WithLimitsOf20and1000.csv")
        ft7=np.array(feature7)
        ft7=np.delete(ft7,0,1)
        np.savetxt("../Data/Dataset/Features/BetaBarrels_WithLimitsOf20and1000.csv", ft7, delimiter=",")
        
        #LOCATION PREDICTION
        feature8=pd.read_csv("../Data/Dataset/Features/Feature_Location_Prediction_WithLimitsOf20and1000.csv")
        ft8=np.array(feature8)
        ft8=np.delete(ft8,0,1)
        np.savetxt("../Data/Dataset/Features/Location_Prediction_WithLimitsOf20and1000.csv", ft8, delimiter=",")
        
        
        aminoacid_composition=np.genfromtxt("../Data/Dataset/Features/aminoacid_composition_WithLimitsOf20and1000.csv",delimiter=",")
        aminoacid_composition.astype(np.float64)
        aminoacid_physico_chemical_composition=np.genfromtxt("../Data/Dataset/Features/aminoacid_physico_chemical_composition_WithLimitsOf20and1000.csv",delimiter=",")
        aminoacid_physico_chemical_composition.astype(np.float64)
        Dipeptide_composition=np.genfromtxt("../Data/Dataset/Features/dipeptide_composition_WithLimitsOf20and1000.csv",delimiter=",")
        Dipeptide_composition.astype(np.float64)
        Transporter_Pfam_domains=np.genfromtxt("../Data/Dataset/Features/Transporter_Pfam_domains_WithLimitsOf20and1000.csv", delimiter=",")
        Transporter_Pfam_domains.astype(np.int64)
        Ahelices_Signalpeptide=np.genfromtxt("../Data/Dataset/Features/Alphahelices_Signalpeptide_WithLimitsOf20and1000.csv", delimiter=",")
        Ahelices_Signalpeptide.astype(np.int64)
        Bbarrels=np.genfromtxt("../Data/Dataset/Features/BetaBarrels_WithLimitsOf20and1000.csv",delimiter=",")
        Bbarrels.astype(np.int64)
        LocPred=np.genfromtxt("../Data/Dataset/Features/Location_Prediction_WithLimitsOf20and1000.csv",delimiter=",")
        LocPred.astype(np.int64)
        
        features=np.hstack((aminoacid_composition,aminoacid_physico_chemical_composition))
        features=np.hstack((features,Dipeptide_composition))
        features=np.hstack((features,Transporter_Pfam_domains[:,np.newaxis]))
        features=np.hstack((features,Ahelices_Signalpeptide))
        features=np.hstack((features,Bbarrels[:,np.newaxis]))
        features=np.hstack((features,LocPred))
        features.astype(np.float64)
        
        np.savetxt("../Data/Dataset/Features/Features_Dataset_WithLimitsOf20and1000.csv",features,delimiter=",")
        
        print("FIM , tamanho: ", len(features[0]))
  
      
#####EXECUTAR FEATURES
data=pd.read_csv("..Data/Dataset/DataDoms_WithLimitsOf20and1000.csv",sep=",")
features = Features_Creation(data)
features.create_features_csv()
features.create_file_all_features()
    
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:43:16 2019

@author: Andrea Silva
"""

import re
import urllib
import requests
from bs4 import BeautifulSoup
import time
import string
import random
import pickle


def id_generator(size, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def run(domain,email, sequence):
    """
            Submit a job with the specified parameters.
            
            domain      Eukaryota, Bacteria or Archaea  
            email       User e-mail address. See Why do you need my e-mail address?
            sequence    Protein sequence to analyse in fasta 
    """

    url = 'https://rostlab.org/services/loctree3/submit-job.php'
    
    md5=id_generator(8)+"-"+id_generator(4)+"-4"+id_generator(3)+"-y"+id_generator(3)+"-"+id_generator(12)
    #print(md5)
    
    args = {'domain':domain,
            'email': email,
            'sequence': sequence,
            'md5':md5}

    session=requests.Session()
    resp=session.post(url,data=args)   
    
    reqid=resp.text[11:18].strip("b'")
    #print(reqid)
    
    result_url=("https://rostlab.org/services/loctree3/results.php?id="+str(md5))

    return reqid , result_url

def LoadingFinished(reqid):
    arg2={'jid':reqid}
    
    session=requests.Session()
    resp2=session.post(url="https://rostlab.org/~loctree3/qjobstat.php",data=arg2)
    
    while resp2.text[14:21] == "waiting":
        resp2=requests.get(url="https://rostlab.org/~loctree3/qjobstat.php",params=arg2)
        #print(resp2.text)
        time.sleep(10)
    while resp2.text[14:21]=="running":
        resp2=requests.get(url="https://rostlab.org/~loctree3/qjobstat.php",params=arg2)
        #print(resp2.text)
        time.sleep(10)
    
    #print("Loading has finished")
    return True
       
    
def result(result_url):
    
    session=requests.Session()
    resp3=session.get(url=result_url)
    #print(resp3.text)
    
    #===========================================================================
    # s=re.search("<div role\=\"table_body\">((.|\n)*)<div role\=\"table_cell\">PSI-BLAST</div>", resp3.text)
    # table=s.group(0)
    # print(table)
    # tableCells=re.findall("<div role\=\"table_cell\">.*?</div>", table)
    # print(tableCells[4])
    # res=re.search(">.*?<", tableCells[4])
    # result=res.group(0)
    # finalres=result.strip("<>")
    # print(finalres)
    #   
    # return str(finalres)
    #===========================================================================
    
    s=re.findall("<div role=\"table_cell\">(.*)</div>", resp3.text)
    
    return s[11]
    




if __name__ == "__main__":
    #===========================================================================
    # NumDataset=input("Qual o nome do dataset?")
    #       
    #  
    # with open("../Data/Datasets/Dataset"+str(NumDataset)+"/Loctree3JobIdsFix.txt", 'rb') as f:
    #     Loctree3JobIds= pickle.load(f)
    # with open("../Data/Datasets/Dataset"+str(NumDataset)+"/Loctree3ReqIdsFix.txt", 'rb') as g:
    #     Loctree3ReqIds= pickle.load(g)       
    #  
    # a=Loctree3JobIds[17899]    
    # print(a)
    # b=Loctree3ReqIds[17899]
    # print(b)    
    #  
    # LoadingFinished(b)
    # res=result(a)
    # print(res)
    #===========================================================================
    
    url = 'https://rostlab.org/services/loctree3/submit-job.php'
    
    seq=""">tr|Q85AA8|Q85AA8_CHICK Cytochrome c oxidase subunit 1 OS=Gallus gallus GN=COX1 PE=3 SV=1
MTFINRWLFSTNHKDIGTLYLIFGTWAGMAGTALSLLIRAELGQPGTLLGDDQIYNVIVT
AHAFVMIFFMVMPIMIGGFGNWLVPLMIGAPDMAFPRMNNMSFWLLPPSFLLLLASSTVE
AGAGTGWTVYPPLAGNLAHAGASVDLAIFSLHLAGVSSILGAINFITTIINMKPPALSQY
QTPLFVWSVLITAILLLLSLPVLAAGITMLLTDRNLNTTFFDPAGGGDPILYQHLFWFFG
HPEVYILILPGFGMISHVVAYYAGKKEPFGYMGMVWAMLSIGFLGFIVWAHHMFTVGMDV
DTRAYFTSATMIIAIPTGIKVFSWLATLHGGTIKWDPPMLWALGFIFLFTIGGLTGIVLA
NSSLDIALHDTYYVVAHFHYVLSMGAVFAILAGFTHWFPLFTGFTLHPSWTKAHFGVMFT
GVNLTFFPQHFLGLAGMPRRYSDYPDAYTLWNTLSSIGSLISMTAVIMLMFIVWEALSAK
RKVLQPELTATNIEWIHGCPPPYHTFE
    """
    dom="euka"     
      
    reqid,LocTree3Id=(run(domain=dom, email="danielvarzim@hotmail.com",sequence=seq))
    print(reqid)
    print(LocTree3Id)
    LoadingFinished(reqid)
    res=result(LocTree3Id)
    print(res)
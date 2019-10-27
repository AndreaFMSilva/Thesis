# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:45:57 2019

@author: Andrea Silva
"""

import re
import urllib
import requests
import time



def run(seq,evalue="1.0", ga="0"):
    """
            Submit a job with the specified parameters.
            seq         Protein sequence to analyse in fasta format.
            ga            Gathering threshold
            evalue       Select desired E-value (default is 1.0)
             
    """

    url = 'http://pfam.xfam.org/search/sequence'

    args = {'seq': seq,
            'ga': ga,
            'evalue':evalue,
            'submit':"Submit"}
    
    data = urllib.parse.urlencode(args)
    data=str.encode(data)
    
    session=requests.Session()
    resp=session.post(url,data=args)   
    job_content=resp.text
    #print(job_content)
    
    job=re.findall("http://pfam.xfam.org/search/sequence/resultset/.*?\"", job_content)
    print(job)
    if len(job) == 0:
        job_id = job
    else: 
        job_id=job[0].strip('\"')
    #print(job[0].strip("\""))
#    job_id=job[0].strip('\"')
    return job_id

def result(job_id):
    
    url=job_id
    
    session2=requests.Session()
    resp2=session2.get(url)
    job_content2=resp2.text
    #print(job_content2)  
    PfamLinks=re.findall("http://pfam.xfam.org/family/.{7}", job_content2)
    #print(PfamLinks)
    AllPfams=[]
    UniquePfamLinks=list(set(PfamLinks))
    for PfamLink in UniquePfamLinks:
        PfamId=PfamLink[-7:]
        AllPfams.append(PfamId)
    
    return AllPfams
        
if __name__ == '__main__':
    seq="MAGAASPCANGCGPSAPSDAEVVHLCRSLEVGTVMTLFYSKKSQRPERKTFQVKLETRQITWSRGADKIEGAIDIREIKEIRPGKTSRDFDRYQEDPAFRPDQSHCFVILYGMEFRLKTLSLQATSEDEVNMWIRGLTWLMEDTLQAATPLQIERWLRKQFYSVDRNREDRISAKDLKNMLSQVNYRVPNMRFLRERLTDLEQRTSDITYGQFAQLYRSLMYSAQKTMDLPFLEASALRAGERPELCRVSLPEFQQFLLEYQGELWAVDRLQVQEFMLSFLRDPLREIEEPYFFLDEFVTFLFSKENSIWNSQLDEVCPDTMNNPLSHYWISSSHNTYLTGDQFSSESSLEAYARCLRMGCRCIELDCWDGPDGMPVIYHGHTLTTKIKFSDVLHTIKEHAFVASEYPVILSIEDHCSIAQQRNMAQYFKKVLGDTLLTKPVDIAADGLPSPNQLKRKILIKHKKLAEGSAYEEVPTSVMYSENDISNSIKNGILYLEDPVNHEWYPHYFVLTSSKIYYSEETSSDQGNEDEEEPKEASGSTELHSNEKWFHGKLGAGRDGRHIAERLLTEYCIETGAPDGSFLVRESETFVGDYTLSFWRNGKVQHCRIHSRQDAGTPKFFLTDNLVFDSLYDLITHYQQVPLRCNEFEMRLSEPVPQTNAHESKEWYHASLTRAQAEHMLMRVPRDGAFLVRKRNEPNSYAISFRAEGKIKHCRVQQEGQTVMLGNSEFDSLVDLISYYEKHPLYRKMKLRYPINEEALEKIGTAEPDYGALYEGRNPGFYVEANPMPTFKCAVKALFDYKAQREDELTFTKSAIIQNVEKQEGGWWRGDYGGKKQLWFPSNYVEEMVSPAALEPEREHLDENSPLGDLLRGVLDVPACQIAVRPEGKNNRLFVFSISMASVAHWSLDVAADSQEELQDWVKKIREVAQTADARLTEGKMMERRKKIALELSELVVYCRPVPFDEEKIGTERACYRDMSSFPETKAEKYVNKAKGKKFLQYNRLQLSRIYPKGQRLDSSNYDPLPMWICGSQLVALNFQTPDKPMQMNQALFLAGGHCGYVLQPSVMRDEAFDPFDKSSLRGLEPCAICIEVLGARHLPKNGRGIV"
    job_id=run(seq)
    time.sleep(25)
    res=result("http://pfam.xfam.org/search/sequence/resultset/107CF382-9246-11E6-BFD9-AF4B5F09777C")
    print(res)
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:44:32 2019

@author: Andrea Silva
"""

import re
import urllib
import requests

def parameters():
    """
            List parameter names.
    """

    url = 'http://www.ebi.ac.uk/Tools/services/rest/phobius/parameters/'
    req = urllib.request.Request(url)
    content = urllib.request.urlopen(req).read()

    params = re.findall(b'<id>(\w+)</id>', content)
    #params=str(params)
    

    return "Parameters: {0}".format(params)


def parameterdetails(parameter):
    """
            Get detailed information about a parameter.
    """

    if parameter not in parameters():
        print ("ERROR: {0} is not a valid parameter".format(parameter))
        return

    url = "http://www.ebi.ac.uk/Tools/services/rest/phobius/parameterdetails/{0}".format(
        parameter)

    req = urllib.request.Request(url)
    content = urllib.request.urlopen(req).read()

    # TODO: fazer um parser para o xml em content. Talvez fazer um json disso.
    return content


def run(email, format, stype, sequence, title=None):
    """
            Submit a job with the specified parameters.
            email        User e-mail address. See Why do you need my e-mail address?
            title        an optional title for the job.
            format        output format
            stype        input sequence type
            sequence    Protein sequence to analyse. The use of fasta formatted sequence is recommended.
    """

    url = 'http://www.ebi.ac.uk/Tools/services/rest/phobius/run/'

    opener = urllib.request.build_opener(urllib.request.HTTPHandler(debuglevel=0))

    args = {'email': email,
            'format': format,
            'stype': stype,
            'sequence': sequence}

    if title:
        args['title'] = title

    data = urllib.parse.urlencode(args)
    data=str.encode(data)

    job_id = opener.open(url, data=data).read()

    return job_id


def status(job_id):
    """
            Get the status of a submitted job.
    """

    url = 'http://www.ebi.ac.uk/Tools/services/rest/phobius/status/{0}'.format(
        job_id)

    req = urllib.request.Request(url)
    content = urllib.request.urlopen(req).read()

    return content


def resulttypes(job_id):
    """
            Get available result types for a finished job.
    """

    url = 'http://www.ebi.ac.uk/Tools/services/rest/phobius/resulttypes/{0}'.format(
        job_id)

    req = urllib.request.Request(url)
    content = urllib.request.urlopen(req).read()

    # TODO: fazer um parser para o xml em content. Talvez fazer um json disso.
    return content

# print resulttypes('phobius-R20160725-142653-0636-4423627-pg')

def result(job_id, result_type):
    """
        Get the job result of the specified type.    
    """

    if result_type not in {"out", "sequence"}:
        print ("{0} is not a valid result type. Choose 'out' or 'sequence'".format(result_type))
        return

    url = "http://www.ebi.ac.uk/Tools/services/rest/phobius/result/{0}/{1}".format(job_id, result_type)

    req = urllib.request.Request(url)
    content = urllib.request.urlopen(req).read()

    res = dict()
    if result_type == "out":

        out = content.split(b"\n")[1].split()
        # Assumi que os outputs tipo out tem sempre estes 4 parametros

        res["SEQENCE ID"] = out[0]
        res["TM"] = out[1]
        res["SP"] = out[2]
        if len(out) > 3:
            res["PREDICTION"] = out[3]

        return res
    
    elif result_type == "sequence":
        
        return content

if __name__ == "__main__":
    print ("Python API for Phobius (REST)")
    print ()
    print ("Parameters function")
    print (parameters())
    print ("#################################################")
    print ("Parameterdetails function")
    print ("    format")
    print ("#################################################" )
    print (parameterdetails("format"))
    print ("#################################################")
    print ("    stype")
    print (parameterdetails("stype"))
    print ("#################################################")
    print ("    sequence")
    print (parameterdetails("sequence"))
    print ("#################################################")
    print ("run function")
    print (run(email="danielvarzim@hotmail.com",
             format="short",
            stype="protein",
            sequence="MPPMLSGLLARLVKLLLGRHGSALHWRAAGAATVLLVIVLLAGSYLAVLAERGAPGAQLITYPRALWWSVETATTVGYGDLYPVTLWGRLVAVVVMVAGITSFGLVTAALATWFVGREQERRGHFVRHSEKAAEEAYTRTTRALHERFDRLERMLDDNRR"))
     
    print ("#################################################")
    print ("status function")
    print (status("phobius-R20160817-165435-0672-31738254-oy"))
    print ("#################################################")
    print ("resulttypes function")
    print (resulttypes("phobius-R20160817-165435-0672-31738254-oy"))
    print ("#################################################")
    print ("result function")
    print (" result_type = out")
    print (result(job_id="phobius-R20160817-165435-0672-31738254-oy",
        result_type="out"))
    print ("#################################################")
    print (" result_type = sequence")
    print (result(job_id="phobius-R20160817-165435-0672-31738254-oy",
        result_type="sequence"))
"""

@description:
    
    Reformat MPQA Lexicon file 

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import re

# Read the MPQA Lexicon file
filename = "subjclueslen1-HLTEMNLP05.tff"
infile = open(filename, 'r')
content = infile.readlines()
infile.close()

outfile = open("MPQA_Lexicon.csv", 'w')


for line in content:
    
    # Retrieve the part of type, word, and priorpolarity using regular expression
    lexicon = re.search(".*type=(\S+).*word1=(\S+).*priorpolarity=(\S+)", line)
    
    # Get the lexicon score
    lexicon_score = 0
    
    if lexicon.group(3) == "positive":
        lexicon_score = 1
    elif lexicon.group(3) == "negative":
        lexicon_score = -1
            
    outfile.write("%s,%s,%d\n" % (lexicon.group(1), lexicon.group(2), lexicon_score))
    
outfile.close()
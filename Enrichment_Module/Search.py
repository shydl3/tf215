#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
"""
This script is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""
def run(searcher, analyzer):
        tokenFile = open("/home/user/data/Enrichment_Module/DeepCom_JAVA/train_tokens.txt")
        descFile = open("/home/user/data/Enrichment_Module/DeepCom_JAVA/train_desc.txt")
        similarityToken=open("/home/user/data/Enrichment_Module/DeepCom_JAVA/train_IR_code_tokens.txt","w")
        similarityDesc=open("/home/user/data/Enrichment_Module/DeepCom_JAVA/train_IR_code_desc.txt","w")
        for i,(line,desc) in enumerate(zip(tokenFile,descFile)):
            command=str(line)
            if len(command)>2000:
                command=command[:2000]
            sourceDesc=str(desc).strip()
            try:
                query = QueryParser("contents", analyzer).parse(QueryParser.escape(command))
            except:
                similarityToken.write("error"+'\n')
                similarityDesc.write(""+'\n')
                similarityToken.flush()
                similarityDesc.flush()
                continue
            scoreDocs = searcher.search(query, 2).scoreDocs
            # print ("%s total matching documents." % len(scoreDocs))
            doc = searcher.doc(scoreDocs[0].doc)
            description=doc.get("descriptions")
            description=str(description).strip()
            if sourceDesc!=description:
                code=doc.get("contents")
                similarityToken.write(code)
                similarityDesc.write(description+'\n')
                similarityToken.flush()
                similarityDesc.flush()
            else:
                doc = searcher.doc(scoreDocs[1].doc)
                description=doc.get("descriptions")
                description=str(description).strip()
                code=doc.get("contents")
                similarityToken.write(code)
                similarityDesc.write(description+'\n')
                similarityToken.flush()
                similarityDesc.flush()
            if i%10==0:
                print(i)

        tokenFile.close()
        similarityDesc.close()
        similarityToken.close()

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print ('lucene', lucene.VERSION)
    base_dir = "/home/user/data/Enrichment_Module/IndexFiles_DeepCom.index"
    directory = SimpleFSDirectory(Paths.get(base_dir))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer()
    run(searcher, analyzer)
    del searcher

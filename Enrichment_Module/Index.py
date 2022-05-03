#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.analysis.core import WhitespaceAnalyzer

"""
This class is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir, analyzer):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        # analyzer=WhitespaceAnalyzer(analyzer)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(root, writer)
        writer.commit()
        writer.close()
        print ('done')

    def indexDocs(self, root, writer):

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(True)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        # t2 = FieldType()
        # t2.setStored(True)
        # t2.setTokenized(False)
        # t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        tokenFile = open(root+"train_tokens.txt")
        descFile = open(root+"train_desc.txt")
        item = str(tokenFile.readline())
        descItem = str(descFile.readline())
        # id=1
        while item and descItem:
            doc = Document()
            if len(item) > 0:
                doc.add(Field("descriptions", descItem, t1))
                doc.add(Field("contents", item, t1))
                # doc.add(Field("id", str(id), t1))
            else:
                print ("warning: no content in %s" % "train_tokens.txt")
                print ("warning: no desctiption in %s" % "train_desc.txt")
            writer.addDocument(doc)
            item=tokenFile.readline()
            descItem=descFile.readline()
            # id=id+1
        tokenFile.close()
        descFile.close()
                    
                # except (Exception, e):
                #     print ("Failed in indexDocs:", e)

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print ('lucene', lucene.VERSION)
    start = datetime.now()
    base_dir = "/home/user/data/Enrichment_Module/IndexFiles_DeepCom.index"
    IndexFiles("/home/user/data/Enrichment_Module/DeepCom_JAVA/", base_dir,
                StandardAnalyzer())
    end = datetime.now()
    print (end - start)


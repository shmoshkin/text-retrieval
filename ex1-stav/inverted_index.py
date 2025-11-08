import glob
import os
import re
import xml.etree.ElementTree as ElementTree

class InvertedIndex():
    def __init__(self, dir):
        self.dir = dir
        self.indices = {}
        self.docs_list = []
        self.doc_index = 1

    def extract_docs(self, limit=None):
        count = 0
        
        for filepath in sorted(glob.iglob(self.dir + '**/**', recursive=True)):
            if limit and count >= limit:
                break
            
            if os.path.isfile(filepath):
                self.docs_list.extend(self.extract_file(filepath))
                count+=1
    
    def extract_file(self, filepath):
        lst_internal_docs = []
        with open(filepath, 'r') as f:
            file_content = f.read()
        
        print(f"Current file is {filepath}")

        # There are sone edge case when the file is not a standard XML file and contains <DOCNO>APXXXXXX-XXXX</DOCNO></TEXT>
        # which is not a valid XML file (Has a closing tag without an open one)
        if ("</DOCNO>\n</TEXT>" in file_content):
            print(f"File {filepath} is has non standard tags, fixing it")
            file_content = file_content.replace("</DOCNO>\n</TEXT>", "</DOCNO>\n<TEXT>\n</TEXT>")

        file_content = '<ROOT>' + file_content + '</ROOT>'
        root_content = ElementTree.fromstring(file_content)
        for doc in root_content:
            docno = doc.find('DOCNO').text.strip()

            # Sort doctext to include just words with one space between each one and without any new line
            doctext = ""
            for subtext in doc.iter('TEXT'):
                doctext += subtext.text.strip().replace('\n', '') + " "
            doctext = doctext.strip()
            doctext = re.sub(' +', ' ', doctext)

            lst_internal_docs.append({
                'doc_index': self.doc_index,
                'doc_id': docno,
                'doc_text': doctext
            })

            self.doc_index+=1
            
        return lst_internal_docs
    
    def extract_indices_from_text(self, limit=None):
        if not limit or limit > len(self.docs_list):
            limit = len(self.docs_list)

        for idx, doc_object in enumerate(self.docs_list[:limit]):
            doc_index = doc_object['doc_index']
            doc_id = doc_object['doc_id']
            if idx % 1000 == 0:
                print(f"Indexed {idx} docs out of {limit}")
            for word in set(doc_object['doc_text'].split()):
                if not word in self.indices:
                    self.indices[word] = []
                
                self.indices[word].append((doc_index, doc_id))
    
    def get_top_n_words(self, n, print=False):
        most_common_words = sorted(self.indices, key=lambda k: len(self.indices[k]), reverse=True)[:n]
        if print:
            self.print_internal(most_common_words)
        return most_common_words

    def get_bottom_n_words(self, n, print=False):
        most_common_words = sorted(self.indices, key=lambda k: len(self.indices[k]), reverse=False)[:n]
        if print:
            self.print_internal(most_common_words)
        return most_common_words
    
    def print_internal(self, most_common_words):
        for word in most_common_words:
            word_indices_to_print = []
            for doc in self.indices[word]:
                word_indices_to_print.append(f"{doc[0]} ({doc[1]})")
            str_docs = " -> ".join(word_indices_to_print)
            print(f"{word} -> {str_docs}")
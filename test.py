import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
) 

PERSIST_DIR = "./storage"
DATA_PATH = "./data"

class Parser:
    def __init__(self,path=DATA_PATH):
        self.documents = SimpleDirectoryReader(path).load_data()

    def get_parsed_documnets(self):
        return self.documents
    

class Indexer:
    def __init__(self,documents):
        self.docs = documents
        self.index = VectorStoreIndex.from_documents(self.docs)
    
    def get_index(self):
        return self.index
    

def storage_pipeline():
    if not os.path.exists(PERSIST_DIR):
        print("Storing documents....")
        parser = Parser()
        documents = parser.get_parsed_documnets()
        index = Indexer(documents).get_index()
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else :
        print("fetching storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index

    


if __name__ == "__main__":
    index = storage_pipeline()
    user_qes = input("Enter your question : ")
    query_engine = index.as_query_engine()
    response = query_engine.query(user_qes)
    print(response)


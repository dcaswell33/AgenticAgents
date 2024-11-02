import logging
import sys
import os
from typing import List
from dotenv import load_dotenv
from llama_index.core import (StorageContext, VectorStoreIndex, load_index_from_storage, Settings)
from llama_index.core.agent import AgentRunner
from llama_index.core import SimpleDirectoryReader
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.llm_compiler.step import LLMCompilerAgentWorker
from llama_index.readers.wikipedia import WikipediaReader

class AgentDocument:
    """
    Document store object which contains the title of the content (for caching), 
        URL for where to download/retrieve original content, 
        source:['wiki', 'pdf'], 
        and a text description used for the tool description
    """
    def __init__(self, title:str, url:str, source:str, description:str):
        self.title = title
        self.url = url
        self.source = source
        self.description = description

class AssimilatorAgent:
    """
    Runs the Compiler pattern to support efficient agentic workflow in multi-document retrievals
    """

    def __init__(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        
        # Configure the LLM for the subordinate evaluation (experts)
        # Change LLM here for other models, e.g., OpenAI(temperature=0, model="gpt-4"
        load_dotenv()

        # Settings.llm = Ollama(model="llama3.2", request_timeout=120.0, temperature=0) #using OpenAI by default
        # Settings.embed_model = OllamaEmbedding(
        #     model_name="llama3.2",
        #     base_url="http://localhost:11434",
        #     ollama_additional_kwargs={"mirostat": 0},
        # )
        Settings.callback_manager = CallbackManager([])

        # Build list of the tools
        self.query_engine_tools = []
    
    def chat(self, question):
        """
        Asks the LLM a question 
        """
        agent_worker = LLMCompilerAgentWorker.from_tools(
            self.query_engine_tools,
            llm=Settings.llm,
            verbose=True,
            callback_manager=Settings.callback_manager,
        )  
        
        agent = AgentRunner(agent_worker, callback_manager=Settings.callback_manager)

        response = agent.chat(question)

        return str(response)
  
    def add_document(self, doc):
        
        def get_wiki_document(title):
            reader = WikipediaReader()
            return reader.load_data(title, auto_suggest=False) # auto_suggest gets around the wiki 'e' error

        def get_url_document(pdf_url):
            """
            loads document content from a URL. URL may be a file path
            """
            reader = SimpleDirectoryReader(
                input_files=[pdf_url]
            )
            files = reader.load_data()
            return files
        
        def content_loader(document:AgentDocument):
            """
            Toggles which loader to send the data
            """
            if document.source == "wiki":
                return get_wiki_document(document.url)
            if document.source == "pdf":
                return get_url_document(document.url)
            
        node_parser = SentenceSplitter()

        if not os.path.exists(f"./data/{doc.title}"):
            logging.log(logging.INFO, f"{doc.title} not found, retrieving.")
            document = content_loader(doc)
            nodes = node_parser.get_nodes_from_documents(document)
            # build vector index
            vector_index = VectorStoreIndex(
                nodes, #service_context=ServiceContext.from_defaults(llm=llm),
                llm=Settings.llm, embed_model=Settings.embed_model,
                callback_manager=Settings.callback_manager
            )
            vector_index.storage_context.persist(persist_dir=f"./data/{doc.title}")
        else:
            logging.log(logging.INFO, f"{doc.title} found, loading to memory.")
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"./data/{doc.title}"),
                llm=Settings.llm, embed_model=Settings.embed_model,
                callback_manager=Settings.callback_manager,
            )
        # define query engines
        vector_query_engine = vector_index.as_query_engine()

        # define tools
        self.query_engine_tools.append(
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{doc.title}",
                    description=(doc.description),
                ),
            )
        )

    def add_documents(self, documents:List[AgentDocument]):
        """
        Loads the specified data into a local data store that is ready for LLM processing.
        If the data is already present, loads the data into memory.
        """
        for doc in documents:
            self.add_document(doc)

    def list_documents(self):
        return [tool.metadata.name for tool in self.query_engine_tools]


def main():
    
    documents = []
    # test documents from wiki
    for doc in ["Toronto", "Seattle", "Denver", "Boston", "Miami"]:
        documents.append(
            AgentDocument(doc, doc, 'wiki', 
                     "Useful for questions related to specific aspects of"
                        f" {doc} (e.g. the history, arts and culture, sports, demographics, or more)."))

    # documents.extend([
    #    Agent_Document("NIST80053", "./NIST.SP.800-53r5.pdf", "pdf", "NIST 800-53 revision 5 describing security implementation controls that should be used across governement agencies"),
    #    Agent_Document("MARSEVol1", "./MARS-E v2-2-Vol-1_Final-Signed_08032021-1.pdf", "pdf", "MARS-E v2 Volume 1 describing security implementation controls that should be used across financial services"),
    #    Agent_Document("MARSEVol2", "./MARS-E-v2-2-Vol-2-AE-ACA-SSP_Final_08032021.pdf", "pdf", "MARS-E v2 Volume 2 describing security implementation controls that should be used across financial services"),
    # ])

    # Create embedding models for the key documents
    agent = AssimilatorAgent()
    agent.add_documents(documents)
    
    response = agent.chat("What is the biggest historical difference between Denver and Miami?")
    print(response)

if __name__ == "__main__" :
    main()
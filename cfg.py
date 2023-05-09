import logging
import os
from dataclasses import dataclass

import openai
from buster.busterbot import BusterConfig
from buster.retriever import Retriever
from buster.utils import get_retriever_from_extension
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# set openAI creds
openai.api_key = os.getenv("OPENAI_API_KEY")

# hf hub information
REPO_ID = "jerpint/buster-cluster-dataset"
DB_FILE = "documents_mila.db"
HUB_TOKEN = os.environ.get("HUB_TOKEN")
# download the documents.db hosted on the dataset space
logger.info(f"Downloading {DB_FILE} from hub...")
hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=DB_FILE,
    token=HUB_TOKEN,
    local_dir=".",
    local_dir_use_symlinks=False,
)
logger.info("Downloaded.")

# setup retriever
retriever: Retriever = get_retriever_from_extension(DB_FILE)(DB_FILE)

documents_filepath = "./documents.db"
buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_prompt": "I'm sorry, but I am an AI language model trained to assist with questions related to the Mila Cluster. I cannot answer that question as it is not relevant to the cluster or its usage. Is there anything else I can assist you with?",
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
    },
    retriever_cfg={
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    },
    completion_cfg={
        "name": "ChatGPT",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": True,
            "temperature": 0,
        },
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo",
    },
    prompt_cfg={
        "max_tokens": 3500,
        "text_before_documents": (
            "You are a chatbot assistant answering technical questions about the Mila Cluster, a GPU cluster for Mila Students."
            "You are a chatbot assistant answering technical questions about the Mila Cluster."
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documentation."
            "If the answer is in the documentation, summarize it in a helpful way to the user."
            "If it isn't, simply reply that you cannot answer the question because it is not available in your documentation."
            "If it is a coding related question that you know the answer to, answer but warn the user that it wasn't taken directly from the documentation."
            "Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "Here is the documentation: "
            "<DOCUMENTS> "
        ),
        "text_before_prompt": (
            "<\DOCUMENTS>\n"
            "REMEMBER:\n"
            "You are a chatbot assistant answering technical questions about the Mila Cluster, a GPU cluster for Mila Students."
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documentation above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "3) Do not reference any links, urls or hyperlinks in your answers.\n"
            "4) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "5) Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "'I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?'"
            "For example:\n"
            "What is the meaning of life for a cluster bot?\n"
            "I'm sorry, but I am an AI language model trained to assist with questions related to the Mila Cluster. I cannot answer that question as it is not relevant to its usage. Is there anything else I can assist you with?"
            "Now answer the following question:\n"
        ),
    },
    document_source="mila",
)

import os
import zipfile
import requests
import pandas as pd
import time

from buster.documents_manager import DeepLakeDocumentsManager

from buster.docparser import get_all_documents
from buster.parser import SphinxParser


# Extract all documents from the html into a dataframe
df = get_all_documents(
    root_dir="./mila-docs/docs/_build/",
    base_url="https://docs.mila.quebec/",
    parser_cls=SphinxParser,
    min_section_length=100,
    max_section_length=1000,
)

# Add the source column
df["source"] = "mila_docs"

# Save the .csv with chunks to disk
df.to_csv("mila_docs.csv")

# Initialize the vector store
dm = DeepLakeDocumentsManager(
    vector_store_path="deeplake_store",
    overwrite=True,
    required_columns=["url", "content", "source", "title"],
)

# Add all embeddings to the vector store
dm.batch_add(
    df=df,
    batch_size=3000,
    min_time_interval=60,
    num_workers=32,
    csv_filename="embeddings.csv",
    csv_overwrite=False,
)

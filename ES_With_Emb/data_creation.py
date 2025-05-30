import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def process_pdf(pdf_file):
    # Load and split PDF content into chunks
    loader = PyPDFLoader(pdf_file)
    document = loader.load()

    if not document:  # Check if the document is empty
        print(f'Error: Failed to load document {pdf_file}')
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)

    if not chunks:  # Check if there are no chunks
        print(f'Error: No content extracted from {pdf_file}')
        return []
    
    return chunks

# path="echapter-46-66.pdf"
# chunks=process_pdf(path)
# print(chunks[0].page_content)
data_dir="TATA_data"

pdf_files_gdp=[]
pdf_files_cpi=[]
pdf_files_iip=[]

for root,dirs,files in os.walk(data_dir):
    if "cpi_data" in root:
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files_cpi.append(os.path.join(root, file))
    
    if "gdp_data" in root:
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files_gdp.append(os.path.join(root, file))
    
    if "iip_data" in root:
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files_iip.append(os.path.join(root, file))   

print(f"Number of GDP files: {len(pdf_files_gdp)}") 
print(f"Number of CPI files: {len(pdf_files_cpi)}")   
print(f"Number of IIP files: {len(pdf_files_iip)}") 
   
index=[]

# Index creation for GDP
for file in pdf_files_gdp[:2]:
    filename = file
    chunks = process_pdf(filename)
    for chunk in chunks:
        if chunk.page_content.strip(): 
            index.append({"filename": file, "text": chunk.page_content})
            
# Index creation for CPI
for file in pdf_files_cpi[:2]:
    filename = file
    chunks = process_pdf(filename)
    for chunk in chunks:
        if chunk.page_content.strip(): 
            index.append({"filename": file, "text": chunk.page_content})
                        
# Index creation for IIP
for file in pdf_files_iip:
    filename = file
    chunks = process_pdf(filename)
    for chunk in chunks:
        if chunk.page_content.strip(): 
            index.append({"filename": file, "text": chunk.page_content}) 
             
with open("index.txt", "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=4)   

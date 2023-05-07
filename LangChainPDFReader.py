
# example code to query pdf documents using OpenAI

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import re

import platform
from tempfile import TemporaryDirectory
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# set OpenAI API key, get this from OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"

# initialise the variable that will hold the text loaded from pdfs
raw_text = ""

#
# Load product brochure file using PDF --> Image --> Text (OCR) 
#

# PyTesseract & Poppler. Install Tesseract-OCR separately (download Windows install from online)
#pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")
#path_to_poppler_exe = Path(r"C:\Program Files (x86)\poppler-0.68.0_x86\bin")

## Set input and output paths for OCR
#print("Loading product brochure using OCR conversion...")
#PDF_file = Path(r"C:/data/Projects/PythonLangChainPDFReader/fbau_pic0188_apeos_c5240-en.pdf")
#out_directory = Path(r"C:/data/Projects/PythonLangChainPDFReader").expanduser()

#image_file_list = []

#with TemporaryDirectory() as tempdir:
#    # read in PDF at 500 dpi
#    pdf_pages = convert_from_path(PDF_file, 500, poppler_path=path_to_poppler_exe)

#    # save one image file per page
#    print("Saving pages as images...")
#    for page_enumeration, page in enumerate(pdf_pages, start=1):
#        filename = f"{tempdir}\page_{page_enumeration:03}.jpg"
#        page.save(filename,"JPEG")
#        image_file_list.append(filename)

#    # convert image pages to text using OCR
#    print("Converting page images to text...")
#    raw_text += "<DOCUMENT>\n<NAME>fbau_pic0188_apeos_c5240-en.pdf</NAME>\n<CONTENT>\n"
#    for image_file in image_file_list:
#        text = str(((pytesseract.image_to_string(Image.open(image_file)))))
#        raw_text += text
#    raw_text += " \n</CONTENT>\n</DOCUMENT>\n"

#
# load documents using PDF --> Text conversion
#

print("Loading pdf documents using straight text conversion...")

# load TotoyaDisposalNotice.pdf
reader = PdfReader('C:/data/Projects/PythonLangChainPDFReader/ToyotaDisposalNotice.pdf')

raw_text += "<DOCUMENT>\n<NAME>ToyotaDisposalNotice.pdf</NAME>\n<CONTENT>\n"
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

raw_text += " \n</CONTENT>\n</DOCUMENT>\n"

# Load WilliamCovidCert.pdf
reader2 = PdfReader('C:/data/Projects/PythonLangChainPDFReader/WilliamCovidCert.pdf')

raw_text += "<DOCUMENT>\n<NAME>WilliamCovidCert.pdf</NAME>\n<CONTENT>\n"
for i, page in enumerate(reader2.pages):
    text = page.extract_text()
    if text:
        raw_text += text

raw_text += "\n</CONTENT>\n</DOCUMENT>\n\n"

# Load WilliamFlemingResume.pdf
reader3 = PdfReader('C:/data/Projects/PythonLangChainPDFReader/WilliamFlemingResume.pdf')

raw_text += "<DOCUMENT>\n<NAME>WilliamFlemingResume.pdf</NAME>\n<CONTENT>\n"
for i, page in enumerate(reader3.pages):
    text = page.extract_text()
    if text:
        raw_text += text

raw_text += "\n</CONTENT>\n</DOCUMENT>\n\n"


# load TotoyaDisposalNotice2.pdf
reader4 = PdfReader('C:/data/Projects/PythonLangChainPDFReader/ToyotaDisposalNotice2.pdf')

raw_text += "<DOCUMENT>\n<NAME>ToyotaDisposalNotice2.pdf</NAME>\n<CONTENT>\n"
for i, page in enumerate(reader4.pages):
    text = page.extract_text()
    if text:
        raw_text += text

raw_text += " \n</CONTENT>\n</DOCUMENT>\n"

# load Prince2Cert.pdf
reader5 = PdfReader('C:/data/Projects/PythonLangChainPDFReader/Prince2Cert.pdf')

raw_text += "<DOCUMENT>\n<NAME>Prince2Cert.pdf</NAME>\n<CONTENT>\n"
for i, page in enumerate(reader5.pages):
    text = page.extract_text()
    if text:
        raw_text += text

raw_text += " \n</CONTENT>\n</DOCUMENT>\n"

# Load product brochure again using simple PDF to text
#reader4 = PdfReader('C:/data/Projects/PythonLangChainPDFReader/fbau_pic0188_apeos_c5240-en.pdf')

#raw_text += "<DOCUMENT>\n<NAME>fbau_pic0188_apeos_c5240-en.pdf simple text version </NAME>\n<CONTENT>\n"
#for i, page in enumerate(reader4.pages):
#    text = page.extract_text()
#    if text:
#        raw_text += text

#raw_text += "\n</CONTENT>\n</DOCUMENT>\n\n"


# print the raw text we have loaded
# print(raw_text)

# Split raw text into 1000 character chunks and store in Vector DB

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 2000,
    chunk_overlap = 200,
    length_function = len,
    )

texts = text_splitter.split_text(raw_text)

print("Split documents into:" + str(len(texts)) + " chunks...")

# store embeddings for chunks in vector db
print("Connecting to OpenAI embedding API...");
embeddings = OpenAIEmbeddings()

print("Store embeddings in vector db...")
docsearch = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

print("Connecting to OpenAI ChatGPT API...")
chain = load_qa_chain(OpenAI(), chain_type="stuff")             

# hard coded queries to save time

#query = "\nDoes William Fleming have a COVID 19 vaccination certificate and if so when did he get it? Please quote any documents that support your answer."
#docs = docsearch.similarity_search(query)
#print(query)
#print(chain.run(input_documents=docs, question=query))

#query = "\nDoes William Fleming currently own a vehicle with rego number XTF919? Please quote any documents that support your answer."
#docs = docsearch.similarity_search(query)
#print(query)
#print(chain.run(input_documents=docs, question=query))

#query = "\nWhat is Williams date of birth, current address, phone number and drivers license? Please quote any documents that support your answer."
#docs = docsearch.similarity_search(query)
#print(query)
#print(chain.run(input_documents=docs, question=query))

query = "\nDoes Yuko Fleming have a COVID vaccination? Please quote any documents that support your answer."
docs = docsearch.similarity_search(query)
print(query)
print(chain.run(input_documents=docs, question=query))

query = "\nFor each document you have been provided, list the corporate division most likely to be interested in the document from the following options (HR, Finance, Legal, IT, Sales, Marketing), along with one word the best classifies the document"
docs = docsearch.similarity_search(query)
print(query)
print(chain.run(input_documents=docs, question=query))

query = "\nList any documents that have the same or very similar content"
docs = docsearch.similarity_search(query)
print(query)
print(chain.run(input_documents=docs, question=query))

query = "\n. For each document provided, list five keywords or phrases that could be used to categorise the document. Do not use any keywords that contain reference numbers, dates or ID's"
docs = docsearch.similarity_search(query)
print(query)
print(chain.run(input_documents=docs, question=query))


#query = "\n\nWhat countries did William work in before he moved to Australia and what positions did he hold? Please quote any documents that support your answer."
#docs = docsearch.similarity_search(query)
#print(query)
#print(chain.run(input_documents=docs, question=query))

#query = "\n\nHow does the scan delivery function work on the Apeos C5240. Please explain the process step by step and quote any documents that support your answer."
#docs = docsearch.similarity_search(query)
#print(query)
#print(chain.run(input_documents=docs, question=query))

print("\n\n")

# loop around answering questions until the user enters the text stop

while True:
    query = input("Enter question:")
    if query == "stop":
        break    
    docs = docsearch.similarity_search(query)
    print("docs length = " + str(len(docs)) + "\n")
    print(chain.run(input_documents=docs, question=query))
    print("\n\n")


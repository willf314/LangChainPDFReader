
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import re
import os
import pandas as pd 
import time
    
#Set "OPENAI_API_KEY" in "secrets.TOML" or as a windows environment variable, its needed to connect to OpenAI
#Max file size is controlled by the maxUploadSize parameter defined in config.toml. Currently set to 3MB

#maximum number of files we will allow a user to upload at one time
MAX_FILES = 5      

#setup webpage
st.set_page_config(page_title="Helpful Harry", page_icon=":fish:")

col1,col2 = st.columns(2)

with col1:
    st.header("Helpful Harry")
    st.markdown("*Give Harry documents to classify and then ask questions about the content*")

with col2:
    st.image(image="Harry.png", width=150)

st.markdown("**Uploads docs...**")



#allow user to choose files
uploaded_files = st.file_uploader(label="",type=['pdf'], accept_multiple_files = True , label_visibility='collapsed')

raw_text=""
file_names=[]

if uploaded_files:    
    if len(uploaded_files) < (MAX_FILES+1):
        for file in uploaded_files:
            if file:
                reader = PdfReader(file)
                file_names.append(file.name)
                raw_text += " [Document starts:" + file.name + "] "
                for j, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        raw_text += text
                raw_text += " [Document ends] "      
    
        # display the uploaded contents
        #{raw_text[:min(len(raw_text), 500)]}
        #st.markdown(
        #    f"""
        #    <div style='border: 1px solid #ccc; border-radius: 3px; padding: 10px;'>
        #        {raw_text}                
        #    </div>
        #    """,
        #    unsafe_allow_html=True,
        #)

        # chunk and load the content into vector database
        text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        )

        texts = text_splitter.split_text(raw_text)
    
        # load document into vector database    
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)    
        #st.markdown("*loaded " + str(len(texts)) + " data chunks into vector database*")    

        #setup langchain chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")        
           
        # setup progress bar
        progress_text = "Classifying documents..."
        my_bar = st.progress(0, text=progress_text)

        # classify each document with a call to the LLM and collate results into a table
        answer = ""    
        document_summaries=[]    
        num_files = len(file_names)
        file_count = 0
        #st.write(file_names)

        for file_name in file_names:
            # increment progress bar
            file_count += 1
            my_bar.progress(int((file_count/num_files)*100), text=progress_text)

            # form the classification prompt
            query = "Summarise what the document " + file_name + " is about, what type of document it is and people involved, in one or two sentences, "
            query += "list the corporate division most likely to be interested in the document out of (HR, Finance, Accounting, Marketing, Sales, IT, Legal, Engineering) "
            query += "and finally list three keywords that best classify the type of document by different categories."
            query += "Do not include any ID's or reference numbers in your response."
            query += "Return the response in the format: Document Name | Summary | Corporate Division | Keywords. You must always include the pipe delimiters. Do not include column headers."
        
            # retreive matching chunks from vector DB        
            docs = docsearch.similarity_search(query)
        
            # pass question and chunks to LLM
            answer = chain.run(input_documents=docs, question=query)
            # st.write(answer)

            #map answer to a list with four items, one item for each column in a classification table row
            #note that the LLM does not always return data in the requested format - need to handle unexpected results
        
            answer1 = answer.strip()
            answer2 = answer1.split('|')
            document_summary = [""] * 4
        
            if (len(answer2) == 4):
                document_summary[0] = file_name         # file_name
                document_summary[1] = answer2[1]        # summary
                document_summary[2] = answer2[2]        # most relevant division
                document_summary[3] = answer2[3]        # keywords
            else:
                document_summary[0] = file_name
                document_summary[1] = "oops I got confused trying to read this"    
                document_summary[2] = " "
                document_summary[3] = " "
                    
            #append document classification to list - each item maps to another row in the classification table
            document_summaries.append(document_summary)

        # remove progress bar
        my_bar.empty()
                
        # load classification table with results for display        
        st.markdown("**Harry's classification...**")    
        column_names = ['Document Name', 'Summary', 'Most Relevent Division', 'Keywords']
        df = pd.DataFrame(document_summaries,columns = column_names)
        df.index +=1
        st.table(df)
                
        # now that documents are ingested and classified, allow user to ask questions
        st.markdown("**Ask Harry...**")

        def get_text():
            input_text = st.text_area(label="",placeholder="Your question...",key="query", label_visibility='collapsed')
            return input_text 

    
        query = get_text()

        if query:                
            # add request for a source document to the end of every question
            updated_query = query + " Please quote a source document to support your answer."
        
            # retreive matching chunks from vector DB        
            docs = docsearch.similarity_search(updated_query)
        
            # pass question and chunks to LLM
            answer = chain.run(input_documents=docs, question=updated_query)

            # display answer              
            st.markdown("**Harry says...**")
            st.markdown(
            f"""
            <div style='border: 1px solid #ccc; border-radius: 3px; padding: 10px;'>
                {answer}
            </div>
            """,
            unsafe_allow_html=True,)
    else:
        st.write("oops I can't handle more than 5 docs, please remove some")
       










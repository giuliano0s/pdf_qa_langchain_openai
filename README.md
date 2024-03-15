# Intro
Simple PDF QA using RAG on documents from IFRS Canoas.

# Use
Put your OpenAI API key on 'OPENAI_KEY' variable, then put unprocessed documents(PDF) on ./data/unprocessed_data and run the code. It will process the document and save it in a vector store.
Documents only need to be processed once.

Put your question in the last cell, on 'msg' variable.
The model is aware of chat history.

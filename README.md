# Intro
Simple PDF QA using RAG on documents from IFRS Canoas.

# Use
### Default documents
Go to run_app.ipynb, select your venv's kernel and run the code.
Put your OpenAI API key in the sidebar variable and then validate your api.
Once validated, you can chat with the model.

### Your documents
Put unprocessed documents(PDF) on ./data/unprocessed_data. Documents only need to be processed once.
Go to run_app.ipynb, select your venv's kernel and run the code.
Put your OpenAI API key in the sidebar variable and then validate your api.
Your PDF will be processed for better understanding of the LLM. Documents only need to be processed once. (can take a few minutes depending on the document's size)
After that, you can talk to the model.
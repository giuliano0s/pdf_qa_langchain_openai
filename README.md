# Intro
Simple PDF QA using RAG on documents from IFRS Canoas. <br>

# Use
Install requirements.txt packages in your venv. <br>
If you don't have Jupyter extension installed, you can just activate your venv and run 'streamlit run {app.py path}' on cmd. <br>
### Default documents
Go to run_app.ipynb, select your venv's kernel and run the code. <br>
Put your OpenAI API key in the sidebar variable and then validate your api. <br>
Once validated, you can chat with the model. <br>

### Your documents
- Put unprocessed documents(PDF) on ./data/unprocessed_data. Documents only need to be processed once. <br>
- Put your OpenAI API key in the sidebar variable and then validate your api. <br>
- Your PDF will be processed for better understanding of the LLM. Documents only need to be processed once. (can take a few minutes depending on the document's size) <br>
- After that, you can talk to the model. <br>

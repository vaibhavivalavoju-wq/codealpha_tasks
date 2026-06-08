# codealpha_tasks
## Description
The FAQ Chatbot is a simple AI-based application that answers user questions by finding the most relevant answer from a set of predefined FAQs. It uses Natural Language Processing (NLP) techniques to understand the user’s input. The most similar question is selected, and its answer is displayed to the user. The project also includes a basic web interface using Streamlit, where users can type questions and get instant responses. It demonstrates how text processing and similarity matching can be used to build an intelligent question-answering system.
## Features
* Understands user queries using NLP
* Finds most relevant FAQ using similarity matching
* Fast response generation
* Confidence score for each answer
* Streamlit-based interactive UI(user interface)
* Suggests similar questions for better clarity
* Handles unkown queries with fallback response
* NLP preprocessing with SpaCy
* FAQ dataset stored in CSV
* TF-IDF vectorization
* Cosine similarity matching
* Confidence score
* Fallback response
* Chat history
* Logging
* Modular code structure
## Tech Stack
* Python
* Streamlit
* Scikit-learn
* Pandas
* SpaCy
* Pickle (model storage)
## project structure
faq-chatbot/
│
├── data/
│   └── faqs.csv
│
├── models/
│   └── faq_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── chatbot.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

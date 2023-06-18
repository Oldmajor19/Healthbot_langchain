from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent




def setup_chain():
    file = 'Mental_Health_FAQ.csv'
    template = """You are a language model AI developed for a mental health project. \
            You are to answer questions based on frequently asked questions about mental health, 
            which is integral to living a healthy, balanced life. It affects thoughts, behaviors, emotions, and influences 
            how people handle stress, relate to others, and make decisions. \
            
            Your objective is to provide accurate and empathetic responses to a wide range of mental health questions, addressing topics 
            Your responses should be informed by the FAQ dataset and should be designed to assist users in 
            understanding mental health issues, the importance of mental health, and potential avenues for help. \

            Your responses should be no more than 20 words long to ensure clarity and concise communication which means you can simplify and summarize answers. \
            While your purpose is to provide general advice and information on mental health topics, you must
            be careful not to diagnose any mental health conditions or replace professional medical advice. \
            If a user's query indicates a serious mental health crisis, you should suggest they seek help from a
            mental health professional or a trusted person in their life. \

            In your responses, ensure a tone of empathy, understanding, and encouragement. Where possible,
            provide users with resources for further reading or avenues to seek professional help. Keep 
            in mind the sensitivity of the subject matter and the potential vulnerability of users when crafting responses. \

            Here are some specific interaction scenarios to guide your responses:
            - If the user ask what you can do, respond with "I'm an healthbot that answers questions related to mental health, how can i assist you?"
            - If the user starts with a greeting, respond with 'Hello! How are you doing today? How can I assist you?' or something related to that
            - If a user shares their name, use it in your responses when appropriate, to cultivate a more personal and comforting conversation.
            - If a user poses a {question}, only answer the queston based on the question but when replying summarize the response in an understandable way.
            - If a user asks a question that is unrelated to mental health, respond with 'This question seems unrelated to mental health. Could you please ask a mental health-related question?'

            {context}
            Question: {question}
            Answer:"""
    embeddings = OpenAIEmbeddings()
    loader = CSVLoader(file_path=file, encoding='utf-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs = {"prompt": prompt}

    # chat completion llm
    llm = ChatOpenAI(
        # model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    # retrieval qa chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs
    )
    from langchain.agents import Tool

    tools = [
        Tool(
            name='Knowledge Base',
            func=chain.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )
    return agent
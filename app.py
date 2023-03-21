import streamlit as st
from PIL import Image
from haystack.document_stores import SQLDocumentStore, FAISSDocumentStore
from haystack.nodes import FARMReader, TfidfRetriever, DensePassageRetriever,Seq2SeqGenerator
from haystack.pipelines import ExtractiveQAPipeline,GenerativeQAPipeline

st.set_page_config(
    page_title="Haystack QA",
    page_icon="🛠",
    layout="centered",
    initial_sidebar_state="auto",
)

main_image = Image.open('main_banner.png')


@st.cache_data
def instantiate_model(question):
    params = {"Retriever": {"top_k": 5}, "Reader": {"top_k": 3}}
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    document_store = FAISSDocumentStore.load(index_path="my_faiss")
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_passage=256,
                                      max_seq_len_query=64,
                                      batch_size=2,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True)
    pipe = ExtractiveQAPipeline(reader, retriever)
    prediction = pipe.run(query=question, params=params)
    return prediction

@st.cache_data
def instantiate_genmodel(question):
    params = {"Retriever": {"top_k": 5}, "Generator": {"top_k": 3}}
    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
    document_store = FAISSDocumentStore.load(index_path="my_faiss")
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_passage=256,
                                      max_seq_len_query=64,
                                      batch_size=2,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True)
    pipe = GenerativeQAPipeline(generator, retriever)
    prediction = pipe.run(query=question, params=params)
    return prediction


st.image(main_image, use_column_width='auto')
st.title("✨Question Answering system 📑")
question = st.text_area("Please enter your question:")
col1, col2 = st.columns(2)
with col1:
    retriever_top_k = st.slider('Please select the top N relevant documents in document store to be searched for', 1,
                                10, 1)
with col2:
    reader_top_k = st.slider('Please select the top M answers searched in retrieved N documents', 1, 10, 1)
if st.button("Fetch me the Answers 🚀"):
    if (len(question) != 0):
        with st.spinner("Getting the right answers... 💫"):
            prediction = instantiate_model(question)
            for ans in prediction['answers']:
                st.write(ans.answer)  # main answer
                st.write(ans.context)  # context
                st.write('---')
    else:
        st.warning('⚠ Please enter the question to generate the response 😯')

if st.button("Would you like to generative answer by model for your question🚀"):
    if (len(question) != 0):
        with st.spinner("Getting the right answers... 💫"):
            prediction = instantiate_genmodel(question)
            for ans in prediction['answers']:
                st.write(ans.answer)  # main answer
                # st.write(ans.context)  # context
                st.write('---')
    else:
        st.warning('⚠ Please enter the question to generate the response 😯')

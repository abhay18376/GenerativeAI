# https://docs.streamlit.io/
## LLM - Summarized content, sentiment, key topics
## https://auth.openai.com/log-in
## create user and password
## sign in

import streamlit as st
from openai import OpenAI
# load environment variables
from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file and sets them in os.environ
#####

## title for web page
st.header("Summarization Application")
text = st.text_area("Text to Analyze")
print("text", text)
# can you give me a Prompt , summarization, sentiment, topics
# initailizing openai client

#
prompt ='''Analyze the following text. 
Your analysis should include topic, sentiment and summary. 
Provide the output in json format.
{
'topic': <A one-sentence description of the main subject.>,
'sentiment': <Classify the overall sentiment as either "positive," "negative," or "neutral." >,
'summary': <Provide a concise summary of the text, no more than 50 words.>
}'''
# RELATED QUESTIONS -> PE
custom_prompt = prompt + "/n" + "Text:" + str(text)

client = OpenAI()
response = client.responses.create(
  model="gpt-4.1",
  input=custom_prompt,
  temperature=0.00001 #0.000000000001
)
print(response.output_text)
json_out_json = response.output_text.replace('```json','').replace("```","")
import json
json_out= json.loads(json_out_json)
print(type(json_out))
st.write("Topic:")
st.write(json_out['topic'])
st.write("Sentiment:")
st.write(json_out['sentiment'])
st.write("Summary:")
st.write(json_out['summary'])

# generative ai - theory
# given a simple task -> able to call api - get result python
# given a complex task -> prompting strategies|llm settings -> able to call api - get result python
# Usecase -

## Langchain - search engine -> RAG -> 100 pdf documents -> context -> output
## text2sql
## deployment
## multimodal - image+text

## AI - DL, AGENTIC AI


## Webpage [query ->"India"-> search] -> API [India]-> output -> Webpage

# API -> text
# sentiment
#


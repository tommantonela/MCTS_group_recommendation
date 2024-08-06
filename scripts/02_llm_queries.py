from signal import signal, alarm, SIGALRM # Only for linux-based systems
import time

class Timeout:
    def __init__(self, seconds=1, message="Timed out"):
        self._seconds = seconds
        self._message = message

    @property
    def seconds(self):
        return self._seconds

    @property
    def message(self):
        return self._message
    
    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def handle_timeout(self, *_):
        raise TimeoutError(self.message)

    def __enter__(self):
        self.handler = signal(SIGALRM, self.handle_timeout)
        alarm(self.seconds)
        return self

    def __exit__(self, *_):
        alarm(0)
        signal(SIGALRM, self.handler)    
        
import pandas as pd
import tiktoken
import os

import pickle

import numpy as np
from tqdm import tqdm

import datetime

from io import StringIO

from langchain_community.chat_models import ChatOllama

import sys
import time

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def num_tokens_from_string(string: str, encoding_name: str ="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
import random
random.seed(42)

import ollama
   
def generate(llm,prompt,dict_query,base_results,wr,wc) -> str: 
    
    uh = [f'* User {str(k)}: ' + ','.join(transform_movies(v)) for k,v in dict_query['users_history'].items()]
    uh = '\n'.join(uh)
    
    if wr != 'group': 

        rr = [f'* User {k}: ' + ','.join(transform_movies(base_results[k])) for k in dict_query['users_history']]
        rr = '\n'.join(rr)
    else:
        rr = []
        for k in dict_query['users_history']: rr.extend(transform_movies(base_results[k]))
        rr = sorted(rr)
        rr = '\n'.join(rr)

    dict_format = {'users_history': uh,
                   'to_recommend':rr,
                   'what_to_consider':wc}
           
                                      
    if 'intersection' in dict_query and len(dict_query['intersection']) > 0:
        dict_format['intersection'] = 'This is the "group preferences": ' + ', '.join(transform_movies(list(dict_query['intersection'])))
    else:
        dict_format['intersection'] = 'There are no common "group preferences."'
    
    print(dict_format)  


    model_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE.format(which_recommendations=wr)), # system role
        ("human",  prompt) # human, the user text 
        ])

    message = model_prompt.format(**dict_format)
    
    # print(message)
    # print('-----------------------------------------------------')
    nn = num_tokens_from_string(message)
    print(num_tokens_from_string(message), "tokens (approx.)")
    
    if nn > 20000:
        print('Context too long!')
        return ''
    
    chain = model_prompt | llm 
    response = chain.invoke(dict_format)

    print("Reponse:", response)
    print('-----------------------------------------------------')

    return response
   
my_models = {}

# my_models["mistral:7b-instruct"] = ChatOllama(
#     model="mistral:7b-instruct", temperature=0.0, #format="str",
#     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

my_models["gemma:2b"] = ChatOllama(
    model="gemma:2b-instruct", temperature=0.0, #format="str",
    cache = False,
    cache_prompt=False,
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# my_models["llama3"] = ChatOllama(
#     model="llama3", temperature=0.0, #format="str",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

# os.environ["OPENAI_API_KEY"] = "API_KEY"
# my_models['gpt-3.5-turbo'] = ChatOpenAI(
#     model='gpt-3.5-turbo', temperature=0.0, #format="str",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

import os

#os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'API_KEY'
#os.environ["HF_TOKEN"] = 'API_TOKEN'
#my_models['mistral:7b-instruct-v0.2'] = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2', temperature=0.01)
#my_models['openai-gpt'] = HuggingFaceEndpoint(repo_id='openai-gpt', temperature=0.0)
#my_models['google:gemma-7b'] = HuggingFaceEndpoint(repo_id='google/gemma-7b', temperature=0.0000001)

# my_models['google:gemma-7b'] = HuggingFacePipeline.from_model_id(model_id='google/gemma-7b', task="text-generation", device=0, temperature=0.0000001)

SYSTEM_TEMPLATE = """You are a helpful movie recommender system. Your task is to recommend movies to a group of people based on their waching history.
You will receive:
    * The group preferences.
    * The individual user preferences.
    * The set of {which_recommendations}.

Your task is to use all the provided information to generate a list of movies to recommend to the group. You have access to all the information you need.
"""

PROMPT = """
{intersection}

These are the "individual user preferences":
{users_history}

These are the "recommended movies": 
{to_recommend}

Your task is:
1. Using the "group preferences" and the "individual user preferences", pick 10 movies from "recommended movies" and sort them based on how well they would satisfy the group as a whole. When ranking the movies, consider {what_to_consider}. Position 1 should be the movie that best satisfies the group. Please, use only the movies in "recommended movies", do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences" and the "recommended movies".
Use only "recommended movies". Do not add any extra movie. 

Your JSON answer:

"""

def matching(x,y):
    
    if y is None:
        return None
    
    if '(' in x:
        xs = x.split('(')
        xs = '('.join(xs[0:-1])
    else: 
        xs = x
    
    return y if y.startswith(xs) else None


def transform_movies_to_ids(movies): 
    # print("Transform movies:", movies)
    if isinstance(movies[0],int):
        return movies
    
    if isinstance(movies[0],np.int64):
        return movies
    
    if movies[0].isdigit(): #str representing numbers
        return movies

    recs_ids = [movie_titles_set[x] if x in movie_titles_set else movie_titles_set[next((y for y in movie_titles_set if matching(x, y)), None)] for x in movies]
    recs_ids = [x for x in recs_ids if x != -1]        

    # return to_recommend
    print(recs_ids)
    return recs_ids

def transform_movies(movies): 
    # print("Transform movies:", movies)
    if isinstance(movies[0],str):
        return movies

    recs_ids = ['"'+movie_titles_set[x]+'"' for x in movies if x in movie_titles_set]
 
    return recs_ids


def find_file(dir_path,pickle_filename,prefix_base_recs):

	pf = '__'.join(pickle_filename.replace('_ids.pickle','').split('__')[0:-1])
	print('====',pf)
	for f in os.listdir(dir_path):
		if not f.startswith('recommendations_' + prefix_base_recs):
			continue
		if pf in f:
			return f
		# print(f)
	return None


dir_path = './'

how_many = 75

prefix = 'implicit_restricted_training_all_users__groups' # can be changed

df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')
movie_titles_set = {df_movies['movieId'].values[i] : df_movies['title'].values[i] for i in range(0,len(df_movies))}
movie_titles_set[None] = -1
del df_movies

which_recs = {}
which_recs['per_user'] = 'movies recommended to each user'
which_recs['group'] = 'movies recommended to the group members'

what_consider = {}
what_consider['div'] = 'recommendation diversity'
what_consider['nov'] = 'recommendation novelty'
what_consider['fair'] = "recommendation fairness towards users' preferences"
what_consider['all'] = "recommendation diversity, novelty and fairness towards users' preferences"

n = -1
to_recc = True

for model, llm in my_models.items():

    for query_file in os.listdir(dir_path):
        
        if not query_file.startswith('groups_') or '__queries' not in query_file:
            continue
        
        print('---------------',query_file)
        
        group_query = pd.read_pickle(dir_path + query_file)

        base_results = find_file(dir_path,query_file,prefix)
        if base_results is None:
            print('-------------------- ERROR!! MISSING BASE RECOMMENDATIONS FILE')
            continue
        
        base_results = pd.read_pickle(dir_path + base_results)
        br = {}
        for kk,vv in base_results.items():
            vv = [z[0] for z in sorted(list([(k,v) for k,v in vv.items()]),key= lambda x:x[1],reverse=True)] 
            if to_recc is not True: 
                vv = vv[0:n]
            base_results[kk] = vv

        for wr,wrv in which_recs.items():
            for wc,wcv in what_consider.items():

                results = []
                results_file = dir_path + 'results___' + model.replace(':','') + '__' + wr + '__' + wc + '__' 

                if  to_recc:
                    results_file = results_file + 'only_to_recc' + '__'

                if n != 10: # 10 is la base
                    results_file = results_file + 'n' + str(n).replace('-','m') + '__'
                
                results_file += query_file
                print('__________',results_file)
                
                if os.path.exists(dir_path + results_file):
                    results = pd.read_pickle(results_file)
                                
                how_many = min(how_many,len(group_query))

                for i in tqdm(range(len(results),how_many)): # estos se corren de cero
                    
                    group = group_query[i]

                    bbrr = base_results
                    if to_recc:
                        bbrr = {}
                        for u in group['users_history'].keys():
                            bbrr[u] = [x for x in base_results[u] if x in group['to_recommend']]
                            if n > 0:
                                bbrr[u] = bbrr[u][0:n]

                    with Timeout(90):
                        try:
                            response = generate(llm,PROMPT,group,bbrr,wrv,wcv) 
                        except Exception as e:
                            print('Timeout error!!', e)
                            response = str({"movies":[]})
                        results.append(response)

                    if len(results) % 20 == 0:
                        with open(results_file,'wb') as file: 
                            pickle.dump(results,file)   
                
                with open(results_file,'wb') as file: 
                        pickle.dump(results,file)   
import pandas as pd
import implicit 
from tqdm import tqdm
from collections import defaultdict
import scipy as sp
import os
import pickle
from datetime import datetime
import json

def get_training_recommend_group(structure,train_user_only=5,test_user_only=5,nliked_user_only=5):
    
    # con users_training tengo que editar ratings
    users_training = defaultdict(set) # union of all possible trainings for that user
    to_recommend = set() # union of all possible to recommend sets

    group_mapping = {}
    
    for group in structure:
        sets_ = group[0]
        group_sets = group[1]
        users = set(group_sets.keys())

        to_recommend.update(sets_['test_intersect'])

        for u in users:
            
            group_mapping[u] = {u} # To make it easier the other part
            
            users_training[u].update(sets_['train_intersect'])
            users_training[u].update(group_sets[u]['train_only'][-train_user_only if train_user_only != -1 else 0:])

            to_recommend.update(group_sets[u]['test_only'][:test_user_only if test_user_only != -1 else len(group_sets[u]['test_only'])])
            to_recommend.update(group_sets[u]['nliked_only'][:nliked_user_only if nliked_user_only != -1 else len(group_sets[u]['nliked_only'])])
    
    return users_training, to_recommend, group_mapping


def train_predict(df_ratings_final,users_training,to_recommend,group_mappings,all_users=True,restrict_training=True):
    
    print('----------------',len(to_recommend))
    
    if all_users:         
        set__ = set(df_ratings_final['userId'].values).difference(group_mapping.keys()) 
        for v in group_mapping.values(): set__.update(v) 
        
        index_users = dict(zip(set__, range(len(set__))))
    else:
        index_users = dict(zip(users_training.keys(), range(len(users_training))))
        
    index_items = dict(zip(set(df_ratings_final['movieId'].values), range(len(set(df_ratings_final['movieId'].values)))))

    inverted_index_users = {v:k for k,v in index_users.items()}
    inverted_index_items = {v:k for k,v in index_items.items()}

    row = [] # user
    column = [] # item
    data = [] # rating

    for i in tqdm(range(0,len(df_ratings_final))):
        uu = df_ratings_final['userId'].values[i]

        if not all_users and uu not in users_training:
            continue

        mm = df_ratings_final['movieId'].values[i]
        rr = df_ratings_final['rating'].values[i]
        
        if uu in group_mappings: 
            for uug in group_mappings[uu]: 

                if uug in users_training and mm not in users_training[uug]: 
                    continue

                row.append(index_users[uug])
                column.append(index_items[mm])
                data.append(rr)
                
        elif all_users: 
            row.append(index_users[uu])
            column.append(index_items[mm])
            data.append(rr)
        
    user_items = sp.sparse.coo_matrix((data, (row, column)), shape=(len(index_users), len(index_items)))
    user_items = user_items.asformat('csr')
    model = implicit.als.AlternatingLeastSquares(factors=200,num_threads=1)
    model.fit(user_items)

    recs_users_filtered = {}
    recs_users_all = {}
    for u in tqdm(users_training):
        i = index_users[u]
        recs = model.recommend(i,user_items[i],N=len(index_items),filter_already_liked_items=True)
        recs_users_all[abs(u)] = {inverted_index_items[recs[0][j]] : recs[1][j] for j in range(0,len(recs[0]))}        
        recs_users_filtered[abs(u)] = {k:v for k,v in recs_users_all[abs(u)].items() if k in to_recommend}
        
        ll = [(k,v) for k,v in recs_users_all[abs(u)].items()]
        ll.sort(key=lambda tup: tup[1],reverse=True)
        recs_users_all[abs(u)] = {x[0]:x[1] for x in ll[0:min(len(ll),len(to_recommend)*5)]}

        recs_users_all[abs(u)].update(recs_users_filtered[abs(u)])
        
    return recs_users_all,recs_users_filtered


def get_training_recommend_group_whole(structure,train_user_only=5,test_user_only=5,nliked_user_only=5):
    
    # con users_training tengo que editar ratings
    users_training = {} # union of all possible trainings for that user
    to_recommend = set() # union of all possible to recommend sets
    
    group_mapping = defaultdict(set)
    
    for i in range(0,len(structure)):
        sets_ = structure[i][0]
        group_sets = structure[i][1]
        users = set(group_sets.keys())

        ss = set()
        to_recommend.update(sets_['test_intersect'])

        for u in users:
            ss.update(sets_['train_intersect'])
            ss.update(group_sets[u]['train_only'][-train_user_only if train_user_only != -1 else 0:])

            to_recommend.update(group_sets[u]['test_only'][:test_user_only if test_user_only != -1 else len(group_sets[u]['test_only'])])
            to_recommend.update(group_sets[u]['nliked_only'][:test_user_only if test_user_only != -1 else len(group_sets[u]['nliked_only'])])
            
            group_mapping[u].add(-i)
            
        users_training[-i] = ss
        
    return users_training, to_recommend, group_mapping 


def post_process_meta(recs): # a dict with a dict of {movie:id, score}
    print('Postprocessing...')
    final_recs = []
    
    for i in range(0,len(recs)):
        aggregation = [(k,v) for k,v in recs[i].items()]
        aggregation.sort(key=lambda tup: tup[1],reverse=True)
                
        final_recs.append([(('',''),json.dumps({'movies':[df_movies_titles[x[0]] for x in aggregation]}))]) 
    
    return final_recs


dir_path = './'

implicit_path = dir_path + 'implicit/'

min_ratings_movie = 25
min_ratings_user = 30
min_date = '2016-01-01'

df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')
df_movies

df_movies_titles = {df_movies['movieId'].values[i] : df_movies['title'].values[i] for i in range(0,len(df_movies))}

df_ratings = pd.read_csv(dir_path + 'ml-25m/ratings.csv')
df_ratings

df_ratings_min_date = df_ratings[df_ratings['timestamp'] >= datetime.strptime('2016-01-01', '%Y-%m-%d').timestamp()]
df_ratings_min_date

df_items_ratings = df_ratings_min_date.groupby(['movieId'])[['rating']].count().sort_values(by="rating", ascending=False)
df_items_ratings = df_items_ratings[df_items_ratings['rating'] > min_ratings_movie]
df_items_ratings

df_users_ratings = df_ratings_min_date[df_ratings_min_date['movieId'].isin(set(df_items_ratings.index))].groupby('userId')[['rating']].count().sort_values('rating',ascending=False)
df_users_ratings = df_users_ratings[df_users_ratings['rating'] > min_ratings_user]
df_users_ratings

df_ratings_final = df_ratings_min_date[df_ratings_min_date['userId'].isin(set(df_users_ratings.index))]

value_combinations = []
value_combinations.append({'train_user_only':5,'test_user_only':5,'nliked_user_only':5,'name': 'restricted_training'})
value_combinations.append({'train_user_only':-1,'test_user_only':-1,'nliked_user_only':-1, 'name': 'full_training'})

for file in os.listdir(dir_path):
    
    if not file.startswith('group'):
        continue
    
    print(file)
    structure = pd.read_pickle(dir_path + file)
    
    for x in value_combinations: 
        
        recs_name = 'recommendations_implicit_' + x['name'] 
        
        file_condition_all_user = not os.path.exists(implicit_path + recs_name + f'_all_users__{file}') or os.path.exists(implicit_path + recs_name + f'_all_users_all__{file}')
        file_condition_group_users = not os.path.exists(implicit_path + recs_name + f'_group_users__{file}') or not os.path.exists(implicit_path + recs_name + f'_group_users_all__{file}')
                               
        if file_condition_all_user or file_condition_group_users:
            print(implicit_path + recs_name + f'_all_users__{file}')
            
            print('Obtaining training information...')
            users_training, to_recommend, group_mapping = get_training_recommend_group(structure,train_user_only=x['train_user_only'],test_user_only=x['train_user_only'],nliked_user_only=x['train_user_only'])
            
            if file_condition_all_user:
                recs_all, recs_filtered = train_predict(df_ratings_final,users_training,to_recommend,group_mapping,all_users=True)
                if not os.path.exists(implicit_path + recs_name + f'_all_users__{file}'):
                    with open(implicit_path + recs_name + f'_all_users__{file}','wb') as ff:
                        pickle.dump(recs_filtered,ff)
            
                if not os.path.exists(implicit_path + recs_name + f'_all_users_all__{file}'):
                    with open(implicit_path + recs_name + f'_all_users_all__{file}','wb') as ff:
                        pickle.dump(recs_all,ff)
                 
            if file_condition_group_users:
                recs_all, recs_filtered = train_predict(df_ratings_final,users_training,to_recommend,group_mapping,all_users=False)
                if not os.path.exists(implicit_path + recs_name + f'_group_users__{file}'):
                    with open(implicit_path + recs_name + f'_group_users__{file}','wb') as ff:
                        pickle.dump(recs_filtered,ff)
                        
                if not os.path.exists(implicit_path + recs_name + f'_group_users_all__{file}'):
                    with open(implicit_path + recs_name + f'_group_users_all__{file}','wb') as ff:
                        pickle.dump(recs_all,ff)

        recs_name = 'recommendations_implicit_mg_' + x['name'] 
        
        file_condition_all_user = not os.path.exists(implicit_path + recs_name + f'_all_users__{file}') or os.path.exists(implicit_path + recs_name + f'_all_users_all__{file}')
        
        file_condition_group_users = not os.path.exists(implicit_path + recs_name + f'_group_users__{file}') or not os.path.exists(implicit_path + recs_name + f'_group_users_all__{file}')
        
        if file_condition_all_user or file_condition_group_users:
            
            print(recs_name)
            
            print('Obtaining training information...')
            users_training, to_recommend, group_mapping = get_training_recommend_group_whole(structure,train_user_only=x['train_user_only'],test_user_only=x['train_user_only'],nliked_user_only=x['train_user_only'])
            
            if file_condition_all_user:
            
                recs_all, recs_filtered = train_predict(df_ratings_final,users_training,to_recommend,group_mapping,all_users=True)
            
                if not os.path.exists(implicit_path + recs_name + f'_all_users__{file}'):
                    with open(implicit_path + recs_name + f'_all_users__{file}','wb') as ff:
                        pickle.dump(recs_filtered,ff)

                if not os.path.exists(implicit_path + recs_name + f'_all_users_all__{file}'):
                    with open(implicit_path + recs_name + f'_all_users_all__{file}','wb') as ff:
                        pickle.dump(recs_all,ff)
                
                recs_filtered = post_process_meta(recs_all)
                with open(implicit_path + recs_name.replace('recommendations_','results__') + f'_all_users__{file}','wb') as ff:
                    pickle.dump(recs_filtered,ff)
                
            if file_condition_group_users:
            
                recs_all, recs_filtered = train_predict(df_ratings_final,users_training,to_recommend,group_mapping,all_users=False)
            
                if not os.path.exists(implicit_path + recs_name + f'_group_users__{file}'):
                    with open(implicit_path + recs_name + f'_group_users__{file}','wb') as ff:
                        pickle.dump(recs_filtered,ff)

                if not os.path.exists(implicit_path + recs_name + f'_group_users_all__{file}'):
                    with open(implicit_path + recs_name + f'_group_users_all__{file}','wb') as ff:
                        pickle.dump(recs_all,ff)

                recs_filtered = post_process_meta(recs_all)
                with open(implicit_path + recs_name.replace('recommendations_','results__') + f'_group_users__{file}','wb') as ff:
                    pickle.dump(recs_filtered,ff)

print('Done!', str(datetime.now()))
import pandas as pd
from datetime import datetime

dir_path = './'

data_path = dir_path + 'ml-25m/'

min_ratings_movie = 25
min_ratings_user = 30 

min_date = '2016-01-01'

df_movies = pd.read_csv(data_path + 'movies.csv')
df_movies

df_ratings = pd.read_csv(data_path + 'ratings.csv')
df_ratings

print('Done loading...')

df_ratings_min_date = df_ratings[df_ratings['timestamp'] >= datetime.strptime('2016-01-01', '%Y-%m-%d').timestamp()]
df_ratings_min_date

df_items_ratings = df_ratings_min_date.groupby(['movieId'])[['rating']].count().sort_values(by="rating", ascending=False)
df_items_ratings = df_items_ratings[df_items_ratings['rating'] > min_ratings_movie]
df_items_ratings

df_users_ratings = df_ratings_min_date[df_ratings_min_date['movieId'].isin(set(df_items_ratings.index))].groupby('userId')[['rating']].count().sort_values('rating',ascending=False)
df_users_ratings = df_users_ratings[df_users_ratings['rating'] > min_ratings_user]
df_users_ratings

df_ratings_final = df_ratings_min_date[df_ratings_min_date['userId'].isin(set(df_users_ratings.index))]
df_ratings_final

print('Done preprocessing...')

def get_group_structures(group,threshold_liked=4, train_ratio=0.8):
    aa = df_ratings_final[df_ratings_final['userId'].isin(group)]
    aa = aa.sort_values('timestamp')

    aa['movieId_rating'] = [(aa['movieId'].values[i],aa['rating'].values[i]) for i in range(0,len(aa))]
    aa = aa.groupby('userId')[['movieId_rating']].agg(list)

    aa['movieId_liked'] = [[y[0] for y in x if y[1] >= threshold_liked] for x in aa['movieId_rating']]
    aa['movieId_not_liked'] = [[y[0] for y in x if y[1] < threshold_liked] for x in aa['movieId_rating']]
    aa['train_liked'] = [x[0:int(len(x)*train_ratio)] for x in aa['movieId_liked']]
    aa['test_liked'] = [x[int(len(x)*train_ratio):] for x in aa['movieId_liked']]

    sets_ = {}
    sets_['train_intersect'] = None
    sets_['test_intersect'] = None
    sets_['train_any'] = set()
    sets_['test_any'] = set()
    sets_['not_liked_any'] = set()
   
    for x in group:
        
        mm = set(aa[aa.index == x]['train_liked'].values[0])
        mm_t = set(aa[aa.index == x]['test_liked'].values[0])
        if sets_['train_intersect'] is None:
            sets_['train_intersect'] = mm
            sets_['test_intersect'] = mm_t
        else:
            sets_['train_intersect'] = sets_['train_intersect'].intersection(mm)
            sets_['test_intersect'] = sets_['test_intersect'].intersection(mm_t)

        sets_['train_any'].update(mm)
        sets_['test_any'].update(mm_t)

        aa_nl = aa[aa.index == x]['movieId_not_liked'].values[0]
        sets_['not_liked_any'].update(aa_nl)

    sets_['test_any'] = sets_['test_any'] - sets_['train_any'] 
    sets_

    group_sets = {}
    for x in group:

        group_sets[x] = {}

        mm = list(aa[aa.index == x]['train_liked'].values[0])
        mm_t = list(aa[aa.index == x]['test_liked'].values[0])

        un = set(mm) - (sets_['train_any'] - set(mm))
        
        group_sets[x]['train_user'] = mm 
        group_sets[x]['train_only'] = [x for x in mm if x in un] 

        un =  set(mm_t) - sets_['train_any']   
        group_sets[x]['test_user'] = [x for x in mm_t if x in un] 

        un = set(mm_t) - (sets_['test_any'] - set(mm_t)) 
        group_sets[x]['test_only'] = [x for x in mm_t if x in un] 

        group_sets[x]['nliked_only'] = list(set(aa_nl) - sets_['train_any'] - sets_['test_any']) 

    return sets_, group_sets
	
	
def dict_format(sets_, group_sets, train_user_only=4, test_user_only=5, nliked_user_only=5): 

    dict_format = {}

    if len(sets_['train_intersect']) > 0:
        dict_format['intersection'] = '"' + '", "'.join(df_movies[df_movies['movieId'].isin(sets_['train_intersect'])]['title'].values) +'"' 

    dict_format['users_history'] = []  

    to_recommend = set()
    for x in group_sets:

        dict_format['users_history'].append(f'* User {x}: "' + '", "'.join(list(df_movies[df_movies['movieId'].isin(group_sets[x]['train_only'][-train_user_only:])]['title'].values)) + '"')

        to_recommend.update(group_sets[x]['test_only'][:test_user_only])
        to_recommend.update(group_sets[x]['nliked_only'][:test_user_only])
    
    dict_format['users_history'] = '\n'.join(dict_format['users_history'])
    
    to_recommend = to_recommend.union(sets_['test_intersect'])
    to_recommend = df_movies[df_movies['movieId'].isin(to_recommend)]['title'].values

    dict_format['to_recommend'] = list(to_recommend) # print('Movies to recommend:')
                                            
    return dict_format
	
	
import itertools

stereotypes = {}
stereotypes['race'] = ['asian','white','afroamerican','']
stereotypes['gender'] = ['woman','man','non-binary','']

combinations = list(itertools.product(*stereotypes.values())) # todas las combinaciones posibles
print(len(combinations))
combinations

def get_combinations_one(users,combinations):
    stereo_combinations = []
    for x in users:
        for c in combinations:
            ll = ' '.join(list(c)).strip()
            if len(ll) == 0:
                continue
            comb = f'* User {x} is ' + (ll.strip() if ll.startswith('non') else 'an ' + ll if ll[0] == 'a' else 'a ' + ll) + '.'
            stereo_combinations.append((c,comb))
    stereo_combinations.append((('',''),None))
    return stereo_combinations
	
# ---------------------------------------------


import pickle
import os
from tqdm import tqdm

threshold_liked = 4
train_ratio = 0.8
train_user_only = 5
test_user_only = 5
nliked_user_only = 5

combinations_strategy = get_combinations_one

structure_name = f'__gs_threliked-{threshold_liked}_trainratio-{train_ratio}'
queries_name = f'__queries_trainonly-{train_user_only}_testonly-{test_user_only}_nlikedonly-{nliked_user_only}_1vs'

for group_file in os.listdir(dir_path):
    
    if not group_file.startswith('groups_'):
        continue
    
    if 'trainratio' in group_file:
        continue
    
    if os.path.exists(dir_path + group_file.replace('.pickle','') + structure_name + queries_name + '.pickle'):
        continue
    
    print('-------------------------------',group_file)
    
    groups = pd.read_pickle(dir_path + group_file)
    
    groups_queries = [] 
    sets_all = [] 

    if os.path.exists(dir_path + group_file.replace('.pickle','') + structure_name + queries_name + '.pickle'):
        continue

    for i in tqdm(range(0,len(groups))):
        group = groups[i]['group_members']

        sets_, group_sets = get_group_structures(group,threshold_liked=threshold_liked, train_ratio=train_ratio)

        sets_all.append((sets_, group_sets))

        dict_ = dict_format(sets_, group_sets, train_user_only=train_user_only, test_user_only=test_user_only, nliked_user_only=nliked_user_only)
        
        dict_['stereo_combinations'] = combinations_strategy(group_sets.keys(),combinations)
        groups_queries.append(dict_) 

    with open(dir_path + group_file.replace('.pickle','') + structure_name + '.pickle','wb') as file:
        pickle.dump(sets_all,file)

    with open(dir_path + group_file.replace('.pickle','') + structure_name + queries_name + '.pickle','wb') as file:
        pickle.dump(groups_queries,file)
		
print('Done!')
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm

from collections import Counter
import json
from collections import defaultdict
import pickle

def agg_additive(users,recs,to_recommend,**kwargs): 
    binary = kwargs.get('binary',False)
    recs_all = defaultdict(float)
    for u in users:
        for rr,vv in recs[u].items():
            recs_all[rr] += (vv if not binary else 1)
      
    recs_all = dict(sorted(recs_all.items(), key=lambda item: -item[1]))
    
    return [x for x in recs_all if to_recommend is None or x in to_recommend]

def agg_additive_bin(users,recs,to_recommend,**kwargs):
    return agg_additive(users,recs,to_recommend,binary=True)


import ranky
def borda(users,recs,to_recommend,**kwargs):
    recs_all = [v for k,v in recs.items() if k in users]
    recs_all = ranky.borda(pd.DataFrame(recs_all).T) 
    recs_all = recs_all.sort_values().index

    return [x for x in recs_all if to_recommend is None or x in to_recommend]


def least_misery(users,recs,to_recommend,**kwargs):
    recs_all = defaultdict(list)
    for u in users:
        for rr,v in recs[u].items():
            recs_all[rr].append(v)

    for rr,v in recs_all.items():
        recs_all[rr] = min(v) if kwargs.get('min_',True) else max(v)

    recs_all = dict(sorted(recs_all.items(), key=lambda item: -item[1]))

    return [x for x in recs_all if to_recommend is None or x in to_recommend]

def least_misery_max(users,recs,to_recommend,**kwargs):
    return least_misery(users,recs,to_recommend,min_=False)


def gfar(users,recs,to_recommend,**kwargs): 
    
    def get_p_relevant(users,scores,n):
        p_relevant = {}
        for u in users:
            
            if u not in scores: # shouldn't happen
                continue
                
            if n is None or n > 0: 
                ss = [(k,v) for k,v in scores[u].items()]
                ss.sort(key=lambda x:x[1],reverse=True)
                if to_recommend is not None:
                    n_ = max([i for i in range(0,len(ss)) if ss[i][0] in to_recommend])
                elif 'pos' in kwargs:
                    n_ = max([i for i in range(0,len(ss)) if ss[i][0] in kwargs['pos']])
                else:
                    n_ = n if n is not None else len(ss) 
            else: # n == -1, only to_recommend
                ss = [(k,v) for k,v in scores[u].items() if k in to_recommend]
                n_ = len(ss)
                
            p_relevant[u] = {ss[i][0]:(n_-(i+1))/n_ for i in range(0,min(n_,len(ss)))}
        return p_relevant
        

    def get_S(S,i,p_relevant):
        fs = 0
        for user in p_relevant:
            su = 1
            for it in S:
                su *= (1 - p_relevant[user].get(it,0))
            fs += p_relevant[user].get(i,0) * su
        return fs

    n = kwargs.get('n',None) 
    p_relevant = get_p_relevant(users,recs,n)

    all_items = set()
    for v in p_relevant.values(): all_items.update(v.keys())

    S = []
    for _ in tqdm(range(0,len(all_items))): 
        max_ = -1
        max_item = None
        for i in all_items:
            s = get_S(S,i,p_relevant)
            if max_item is None or s > max_:
                max_ = s
                max_item = i
        S.append(max_item)
        all_items.remove(max_item)

    return S


def epfuzzda(users,recs,to_recommend,**kwargs): # Fairness-preserving Group Recommendations With User Weighting
    
    def select_items(users,recs,to_recommend,n):
        scores_ = {}
        
        for u in users:
            if u not in recs:
                continue
                        
            if n is None or n > 0: # the same logic as in gfar
                ss = [(k,v) for k,v in recs[u].items()]
                ss.sort(key=lambda x:x[1],reverse=True)
                if to_recommend is not None:
                    n_ = max([i for i in range(0,len(ss)) if ss[i][0] in to_recommend])
                elif 'pos' in kwargs:
                    n_ = max([i for i in range(0,len(ss)) if ss[i][0] in kwargs['pos']])
                else:
                    n_ = n if n is not None else len(ss) 
                scores_[u] = {x[0]:x[1] for x in ss[0:n_]}
            else: # n == -1, only to_recommend
                scores_[u] = {k:v for k,v in recs[u].items() if k in to_recommend}
        
        return scores_
    
    n = kwargs.get('n',None)
    scores_ = select_items(users,recs,to_recommend,n)
    S = []
    TOT = 0
    r_u = {k:0 for k in scores_}
    
    all_items = set()
    for v in scores_.values(): all_items.update(v.keys())
    
    for _ in tqdm(range(0,len(all_items))):
        max_ = None
        max_item = None
        for c in all_items:
            TOTc = TOT + sum(v.get(c,0) for v in scores_.values())
            gainc = 0
            for k in r_u:
                eu = max(0,(TOTc/len(r_u)) - r_u[k])
                gainc += min(eu,scores_[k].get(c,0))
            if max_item is None or gainc > max_:
                max_ = gainc
                max_item = c
        S.append(max_item)
        all_items.remove(max_item)

        r_u = {k:v+scores_[k].get(max_item,0) for k,v in r_u.items()}
        TOT = sum(r_u.values())
    
    return S


def pareto(users,recs,to_recommend,**kwargs):
        
    def adjust_scores(users,recs,to_recommend,n): # {item : min_utility} # este no cambia
        scores_ = {}
        rel_total = {}
        for u in users:
            if u not in recs:
                continue
                        
            if n is None or n > 0: # the same logic as in gfar
                ss = [(k,v) for k,v in recs[u].items()]
                ss.sort(key=lambda x:x[1],reverse=True)
                if to_recommend is not None:
                    n_ = max([i for i in range(0,len(ss)) if ss[i][0] in to_recommend])
                elif 'pos' in kwargs:
                    n_ = max([i for i in range(0,len(ss)) if ss[i][0] in kwargs['pos']])
                else:
                    n_ = n if n is not None else len(ss) 
                scores_[u] = {x[0]:x[1] for x in ss[0:n_]}
            else: # n == -1, only to_recommend
                scores_[u] = {k:v for k,v in recs[u].items() if k in to_recommend}
            rel_total[u] = sum(v for v in scores_[u].values())
        
        return scores_,rel_total
        
    def social_welfare(scores_,prev_SW,rel_total,i): 
        sw = 0
        min_ = set()
        for u in scores_:
            uu = prev_SW[u] + scores_[u].get(i,0)
            uu = uu / rel_total[u]
            sw += uu
            min_.add(uu)

        return sw / len(scores_), min(min_)
    
    def update_SW(scores_,prev_SW,i):
        for u in scores_:
            prev_SW[u] += scores_[u].get(i,0)
        return prev_SW
    
    lambd = kwargs.get('lambda',0.8)
    
    n = kwargs.get('n',None)
    scores_,rel_total = adjust_scores(users,recs,to_recommend,n)
    S = []
    prev_SW = {k:0 for k in scores_}
    
    all_items = set()
    for v in scores_.values(): all_items.update(v.keys())
    
    for _ in tqdm(range(0,len(all_items))):
        max_ = -1
        max_item = None
        for i in all_items:
            SW_score, fairness = social_welfare(scores_,prev_SW,rel_total,i)

            cc = lambd * SW_score + (1-lambd) * fairness
            if max_item is None or cc > max_:
                max_ = cc
                max_item = i
        S.append(max_item)
        all_items.remove(max_item)
        prev_SW = update_SW(scores_,prev_SW,max_item)
    return S

dir_path = './'
implicit_path = dir_path + 'implicit/'

df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')
df_movies

df_movies_titles = {df_movies['movieId'].values[i] : df_movies['title'].values[i] for i in range(0,len(df_movies))}

aggregations = {
                'additive' : agg_additive,
                'additive-binary' : agg_additive_bin,
                'borda' : borda,
                'least-misery':least_misery,
                'least-misery-max':least_misery_max,
                 'gfar': gfar,
                'epfuzzda' : epfuzzda,
                'pareto' : pareto
                }

per_user = -1 # -1, 10 y 50

for ff in tqdm(os.listdir(implicit_path)):
    
    if not ff.startswith('recommendations_implicit'):
        continue
    
    if not ff.endswith('.pickle'):
        continue

    if not os.path.exists(dir_path + '__'.join(ff.split('__')[1:])):
        print('Missing structure file for: ',ff)
        continue
    
    if '_mg_' in ff:
        continue
    
    print(ff)
        
    recs = None
    structure = pd.read_pickle(dir_path + '__'.join(ff.split('__')[1:]))
   
    train_user_only = 5 if 'restricted_training' in ff else -1 # full_training
    test_user_only = 5 if 'restricted_training' in ff else -1
    nliked_user_only = 5 if 'restricted_training' in ff else -1

    for agg_name, agg in aggregations.items():
    
        path_out = 'results__' + agg_name + '_' + ff.replace('recommendations_','')
        if per_user is not None:
           path_out = path_out.replace('.pickle', '#pu' + str(per_user).replace('-','m') + '.pickle')
        
        final_recs = []
        if os.path.exists(implicit_path + path_out): 
            try:
                final_recs = pd.read_pickle(implicit_path + path_out)
                continue
            except EOFError as e:
                pass
        
        if len(final_recs) >= min(100,len(structure)): # avoids loading the other file
            continue
        
        final_recs_all = []
        if os.path.exists(implicit_path + path_out): 
            try:
                final_recs_all = pd.read_pickle(implicit_path + path_out.replace('.pickle','__all_recs.pickle'))
                continue
            except EOFError as e:
                pass
        
        print('------',path_out)
        if recs is None:
            try:
                recs = pd.read_pickle(implicit_path + ff) 
            except EOFError as e:
                print('Error with base recommendations!')
                break
        
        for i in tqdm(range(len(final_recs),min(100,len(structure)))):
                
            sets_ = structure[i][0]
            group_sets = structure[i][1]
            users = set(group_sets.keys())

            to_recommend = sets_['test_intersect'] 
            pos = sets_['test_intersect']
            for u in users:
                to_recommend.update(group_sets[u]['test_only'][:test_user_only if test_user_only != -1 else len(group_sets[u]['test_only'])])
                to_recommend.update(group_sets[u]['nliked_only'][:nliked_user_only if nliked_user_only != -1 else len(group_sets[u]['nliked_only'])])
                pos.update(group_sets[u]['nliked_only'][:nliked_user_only if nliked_user_only != -1 else len(group_sets[u]['nliked_only'])])

            
            if per_user is None or per_user > 0: 
                aggregation = agg(users,recs,to_recommend=None,n=per_user)
            else:
                aggregation = agg(users,recs,to_recommend=to_recommend,n=per_user)
                
            final_recs_all.append([(('',''),json.dumps({'movies':[df_movies_titles[x] for x in aggregation[0:250]]}))]) 
            
            aggregation = [x for x in aggregation if x in to_recommend] 

            final_recs.append([(('',''),json.dumps({'movies':[df_movies_titles[x] for x in aggregation]}))]) 
            
            with open(implicit_path + path_out,'wb') as file:
                pickle.dump(final_recs,file)
                
            with open(implicit_path + path_out.replace('.pickle','__all_recs.pickle'),'wb') as file:
                pickle.dump(final_recs_all,file)
        
        del recs
        del final_recs_all
        del final_recs
        
print('All done!')
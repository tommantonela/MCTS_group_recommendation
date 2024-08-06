import pandas as pd
from datetime import datetime
import os
from sklearn import metrics
from collections import deque
import pickle

class evaluation_metric():  

    def __init__(self):
        pass

    def get_scores(self, recs, **kwargs):
        pass

    def summarize(self, scores):
        summarized = {}
        summarized['min'] = min(scores, key=scores.get)
        summarized['min'] = (scores[summarized['min']],summarized['min'])
        summarized['avg'] = (np.mean(list(scores.values())),None) #sum(scores.values()) / np.std(list(scores.values()))
        summarized['avg_std'] = (summarized['avg'][0] - np.std(list(scores.values())),None)
        summarized['variance'] = (np.var(list(scores.values())),None)
        summarized['max'] = max(scores, key=scores.get)
        summarized['max'] = (scores[summarized['max']],summarized['max'])
        summarized['max-min'] = (summarized['max'][0] - summarized['min'][0],None)
        
        summarized['zrecall'] = sum(1 for x in scores.values() if x > 0) / float(len(scores)) # % of users with score != 0
        
        return summarized
    
class precision_k_user(evaluation_metric):

    def get_scores(self, recs, **kwargs): 
        ground_truth = kwargs['ground_truth']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        if len(recs[0:k]) > 0:
            scores = {u: sum(1 for x in recs[0:k] if x in ground_truth[u]) / k for u in ground_truth}
            summarized = self.summarize(scores)
            return scores,summarized
        
        return {}, {}

    
class ndcg_k_user(evaluation_metric):

    def get_scores(self, recs, **kwargs):

        ground_truth = kwargs['ground_truth']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        ndcgs = {}
        if len(recs[0:k]) <= 1:
            return ndcgs,{}
        
        for u in ground_truth:

            ground_truth_ = deque()
            recs_ = deque()

            for x in recs[0:k]:
                recs_.append(1)
                ground_truth_.append(1 if x in ground_truth[u] else 0)
            nd = metrics.ndcg_score([ground_truth_], [recs_])
            ndcgs[u] = nd
                   
        return ndcgs, self.summarize(ndcgs)

import numpy as np
import scipy
from collections import defaultdict

def average_correlations(coeffs,what='avg'):
    z_transformed = [0.5 * np.log((1 + (r[0] if not isinstance(r, float) else r)) / (1 - (r[0] if not isinstance(r, float) else r))) for r in coeffs if (r[0] if not isinstance(r, float) else r) != 1] # Apply Fisher's Z transformation
    if len(z_transformed) == 0: # all was 1.0
        return 1.0    
    if what == 'avg':
        average_z = np.mean(z_transformed)
    elif what == 'variance':
        average_z = np.var(z_transformed)
    elif what == 'median':
        average_z = np.percentile(z_transformed,50)

    average_correlation = (np.exp(2 * average_z) - 1) / (np.exp(2 * average_z) + 1)
    return average_correlation


class rank_correlation_k(evaluation_metric): 
   
    def __init__(self,df_ratings,corr_metric):
        self.df_ratings = df_ratings
        self.corr = scipy.stats.kendalltau if corr_metric == 'kendall' else scipy.stats.spearmanr

    def get_scores(self, recs, **kwargs): 
        ground_truth = kwargs['ground_truth']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        if len(recs[0:k]) > 0:

            scores = {}
            if 'all' not in ground_truth:
                for u in ground_truth:
                    gtu = df_ratings[(df_ratings['userId'] == u) & df_ratings['movieId'].isin(recs[0:k])] 
                    gtu = gtu.sort_values(by='timestamp') 
                    gtu = list(gtu.sort_values(by='timestamp')['movieId'].values) 

                    rr = list(set(x for x in recs[0:k] if x in gtu))
                    
                    if len(rr) > 1: 
                        scores[u] = self.corr(rr,gtu)[0] 
            else:
                gtu = df_ratings[(df_ratings['userId'].isin(kwargs['known'])) & df_ratings['movieId'].isin(recs[0:k])] 
                gtu = gtu.sort_values(by='timestamp') 
                gtu = list(gtu.drop_duplicates(subset=['movieId'],keep='first')['movieId'].values)
                
                rr = list(set(x for x in recs[0:k] if x in gtu))
                
                if len(rr) > 1:
                    scores['all'] = self.corr(rr,gtu)[0] # 

            summarized = self.summarize(scores)
            return scores,summarized
        
        return {}, {}
    
    def summarize(self, scores): 
        summarized = {}
        
        if len(scores) == 0:
            return summarized
        
        summarized['min'] = min(scores, key=scores.get)
        summarized['min'] = (scores[summarized['min']],summarized['min'])
        
        summarized['avg'] = (average_correlations(scores.values()),None) 
        summarized['avg-n'] = (np.mean(list(scores.values())),None) 
        
        summarized['variance'] = (average_correlations(scores.values(),'variance'),None)
        summarized['variance-n'] = (np.var(list(scores.values())),None)
        
        summarized['median'] = (average_correlations(scores.values(),'median'),None)
        summarized['median-n'] = (np.percentile(list(scores.values()),50), None)
        
        summarized['max'] = max(scores, key=scores.get)
        summarized['max'] = (scores[summarized['max']],summarized['max'])
        return summarized 


from sklearn.metrics.pairwise import cosine_distances

class distance__():

    def __init__(self, **kwargs): 
        self.similarities = None
        if 'similarity_path' in kwargs:
            self.similarities = pd.read_pickle(kwargs['similarity_path'])

    def set_items(self,items):
        pass
       
    def compute_distances_reduced(self, nodesA, nodesB):

        nodesA = nodesA.intersection(self.items.index)  
        nodesB = nodesB.intersection(self.items.index)

        if len(nodesA) == 0 or len(nodesB) == 0:
            return None, None, None

        if self.similarities is not None: 
            
            ss = self.similarities[list(nodesB)]
            ss = ss[ss.index.isin(nodesA)]
            
            return ss.values, {ke: v for v, ke in enumerate(ss.index)}, {ke: v for v, ke in enumerate(ss.columns)}

        matrixA = self.items.loc[sorted(list(nodesA))].astype(int)
        
        matrixB = self.items.loc[sorted(list(nodesB))].astype(int)
        
        node_indexes_row = {ke: v for v, ke in enumerate(matrixA.index)}
        node_indexes_columns = {ke: v for v, ke in enumerate(matrixB.index)}

        return cosine_distances(matrixA, matrixB), node_indexes_row, node_indexes_columns

class content_distance(distance__):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'items' in kwargs:
            self.items = kwargs['items']
        else:
            self.items_path = kwargs['items_path']
            self.items = self.load_representations(self.items_path, kwargs.get('euclidean', True))

    def set_items(self, items):
        print('Computing similarities...',len(items))
        self.similarities = cosine_distances(self.items[self.items.index.isin(items)],self.items[self.items.index.isin(items)])
        items = list(items)
        self.similarities = pd.DataFrame(self.similarities, index=items,columns=items)

    def load_representations(self, path, euclidean=True):
        print('Loading representations...', path)
        rep = pd.read_pickle(path)
        return rep
    
class individual_diversity(evaluation_metric):  

    def __init__(self, **kwargs):
        self.distance_ = kwargs['distance']

    def get_scores(self, recs, **kwargs):

        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        dists = {}
        
        if len(recs[0:k]) <= 1:
            return dists
        
        dd, node_indexes, _ = self.distance_.compute_distances_reduced(set(recs[0:k]), set(recs[0:k]))
        if dd is None:
            return dists
        
        dists['all'] = (dd.sum()).sum() / (len(dd) * (len(dd)-1)) 

        return dists


class individual_novelty(evaluation_metric):  

    def __init__(self, **kwargs):
        self.distance_ = kwargs['distance']

    def get_scores(self, recs, **kwargs):

        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        if len(recs[0:k]) > 0:
            known = kwargs['known']

            if 'all' in kwargs.get('ground_truth',{}):
                known = {'all':set().union(*[v for v in known.values()])}

            recs_set = set(recs[0:k])
            dists = {}
            for u, gt in known.items():  

                dd, _, _ = self.distance_.compute_distances_reduced(recs_set, set(gt))
                if dd is None:
                    continue

                dists[u] = dd.sum().sum() / (dd.shape[0] * dd.shape[1]) 
            
            return dists, self.summarize(dists)
        
        return {}, {}
    
    
from collections import Counter
import numpy as np
class coverage(evaluation_metric): 
        
    def get_scores(self, recs, **kwargs):
        
        ground_truth = kwargs['ground_truth']
        nliked_users = kwargs['nliked_users']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        anti = {}
        
        if len(recs[0:k]) < 1:
            return anti,anti
        
        # tres métricas de una
        anti['coverage_relevants'] = Counter()
        anti['coverage_not_relevants'] = Counter()
        anti['coverage_unknown'] = Counter()
        
        for u in ground_truth:

            for x in recs[0:k]:
                if x in ground_truth[u]:
                    anti['coverage_relevants'][u] += 1
                elif x in nliked_users[u]:
                    anti['coverage_not_relevants'][u] += 1
                else:
                    anti['coverage_unknown'][u] += 1
            
            anti['coverage_relevants'][u] = anti['coverage_relevants'][u] / k
            anti['coverage_not_relevants'][u] = anti['coverage_not_relevants'][u] / k
            anti['coverage_unknown'][u] = anti['coverage_unknown'][u] / k
            
        summarized = {}
        summarized['coverage_relevants'] = self.summarize(anti['coverage_relevants']) 
        summarized['coverage_not_relevants'] = self.summarize(anti['coverage_not_relevants']) 
        summarized['coverage_unknown'] = self.summarize(anti['coverage_unknown']) 
         
        return anti, summarized
    
    
def compute_metrics(ground_truth, train, nliked_users, recommendations_valid, recommendations,k=None): 
    results = {}
    summarized_results = {}

    for m,metric in metrics_.items():
        if m != 'coverage':
            results[m], summarized_results[m] = metric.get_scores(recommendations_valid,ground_truth=ground_truth,known=train,nliked_users=nliked_users,k=k)
        else:
            rr = metric.get_scores(recommendations_valid,ground_truth=ground_truth,known=train,nliked_users=nliked_users,k=k)
            results.update(rr[0])
            summarized_results.update(rr[1])
        
    all_ground = set().union(*[v for v in ground_truth.values()])
    all_nliked = {'all' : set().union(*[v for v in nliked_users.values()])}
    
    for x in results:
        if x != 'precision' and x != 'ndcg' and x != 'novelty' and not x.startswith('rank'):
            continue
        
        #results[x].update(metrics_[x].get_scores(recommendations,ground_truth={'good':recommendations_valid},k=k,known=train)[0])
        results[x].update(metrics_[x].get_scores(recommendations_valid,ground_truth={'all':all_ground},k=k,known=train)[0])
        if 'all' in results[x]:
            summarized_results[x]['all'] = (results[x]['all'],None)
        
    if 'coverage_relevants' in results:
        rr = metrics_['coverage'].get_scores(recommendations,ground_truth={'all':all_ground},nliked_users=all_nliked,k=k)[0]
        for mm in rr:

            results[mm]['all'] = rr[mm]['all']
            summarized_results[mm]['all'] = (rr[mm]['all'],None) 
            
    results['diversity'] = (diversity.get_scores(recommendations_valid,k=k), None) # no requiere nada más

    return results,summarized_results


from tqdm import tqdm
import json
import re

pp = re.compile("```(json|){(.|\n)*}```")
pp2 = re.compile("{([^}]*)\}")

re_missing_brace = re.compile('{.*]')

re_comments = re.compile('(\/\/|##).*\n')

re_mutiple_lines = re.compile('\n{2,}')
re_check_extra_comma = re.compile('(],* *){2,}')
re_check_ending = re.compile('(] *){2,}]{1,}')

re_inbetween_quotes = re.compile('"([^"]+)"')

def ground_truths(users, to_recommend, base_i):
    gt = {}
    gt_full = {}
    train = {}
    not_liked = {}
    for x in users:
        test_x = set(base_i[1][x]['test_user'])
        gt_full[x] = test_x
        gt[x] = test_x.intersection(to_recommend) 
        train[x] = set(base_i[1][x]['train_user'])
        not_liked[x] = set(base_i[1][x]['nliked_only'])
    return gt, gt_full, train, not_liked
    
def matching(x,y):
    
    if y is None:
        return None
    
    if '(' in x:
        xs = x.split('(')
        xs = '('.join(xs[0:-1])
    else: 
        xs = x
    
    return y if y.startswith(xs) else None

def parse_movies(rr):
            
    if 'unable to generate' in rr:
        return []
    
    if '//' in rr or '##' in rr:
        rr = re_comments.sub('',rr)
        
    aa = pp.search(rr)
    if aa is None:
        aa = pp2.search(rr)

    aa = aa[0] if aa is not None else aa

    if aa is None:

        xx = re_mutiple_lines.search(rr)
        if xx is not None: # TODO !!!!!
            rr = rr[0:xx.span()[0]]
            rr = rr.replace('\n',' ').strip()

            rr = re_check_extra_comma.sub(']]',rr)
            if re_check_ending.search(rr):
                aa = rr[0:-1] + '}'
    else:
        aa = aa[3:-3] if aa.startswith('```') else aa 
        aa = aa if not aa.startswith('json') else aa[4:]

    if aa is None: 
        rr = rr.replace('`','')

        if '{' in rr and '}' not in rr:
            aa = rr + '}'

    aa = aa[:aa.index('[')+1] + aa[aa.index('[')+1:].replace('[','').replace(']','')
    aa = aa[:aa.rindex('}')] + ']}'

    if aa == "{'movies': []}": 
        aa = aa.replace('\'','"') 

    try: 
        recs = json.loads(aa)['movies']
    except Exception as e: 
        recs = re_inbetween_quotes.findall(aa)
        
    # Check genres
    recs = [x['title'] if isinstance(x,dict) else x for x in recs if x != 'movies' and x != 'title' and x != 'rank' and not isinstance(x,int) and not isinstance(x,float) and 'rime' not in x and 'antasy' not in x] # este venía funcionando bien, pero ya no es el caso
    return recs


def compute_group_results(results,groups,base):

    group_results = []
    
    # all_movies = set() # acá le faltan las recomendaciones que se hicieron
    # for i in range(0,len(groups)): # esto es para poder ahorrar hacer los cálculos cada vez!
        # if isinstance(groups[i]['to_recommend'][0],str): # if they are names, we transform them
            # to_recommend_ids = set(df_movies[df_movies['title'].isin(groups[i]['to_recommend'])]['movieId'].values)
        # else: # if they are ids, we can user them directly
            # to_recommend_ids = groups[i]['to_recommend']
        # all_movies.update(to_recommend_ids)
    # for i in range(0,len(base)):
        # for x in base[i][1]:
            # all_movies.update(base[i][1][x]['test_user'])
            # all_movies.update(base[i][1][x]['train_user'])
    # cdd.set_items(all_movies)
    
    for i in tqdm(range(0,len(results))): 
        results_i = results[i] 
        base_i = base[i]
        group_i = groups[i]

        users = list(base_i[1].keys()) # users in group_i

        if isinstance(group_i['to_recommend'][0],str): # if they are names, we transform them
            to_recommend_ids = set(df_movies[df_movies['title'].isin(group_i['to_recommend'])]['movieId'].values)
        else: # if they are ids, we can user them directly
            to_recommend_ids = group_i['to_recommend']
            group_i['to_recommend'] = set(df_movies[df_movies['movieId'].isin(group_i['to_recommend'])]['title'].values)

        gt_users, gt_full, train_users, nliked_users = ground_truths(users, to_recommend_ids, base_i)
        
        if len(results_i) > 0:
            
            if isinstance(results_i[0],str):
                if not isinstance(results_i,list):
                    recs = parse_movies(results_i)
                else:
                    recs = results_i
            else: 
                recs = list(df_movies[df_movies['movieId'].isin(results_i)]['title'].values)
        else:
            recs = []

        st_res = {}
        st_res['recommended'] = recs 
        st_res['recommended_ids'] = [movie_titles_set[x] if x in movie_titles_set else movie_titles_set[next((y for y in movie_titles_set if matching(x, y)), None)] for x in recs]

        st_res['recs_valid'] = [match for x in recs for y in group_i['to_recommend'] if (match := matching(x, y)) is not None]
        st_res['recs_valid_ids'] = [movie_titles_set[x] for x in st_res['recs_valid']]
        st_res['extra'] = [st_res['recommended'][i] for i in range(0,len(st_res['recommended'])) if st_res['recommended_ids'][i] == -1] 

        st_res['metrics'] = {}
        st_res['summarized_metrics'] = {}
        st_res['metrics-all'] = {}
        st_res['summarized_metrics-all'] = {}

        st_res['metrics-full'] = {}
        st_res['summarized_metrics-full'] = {}
        st_res['metrics-full-all'] = {}
        st_res['summarized_metrics-full-all'] = {}

        #st_res['metrics'][-1], st_res['summarized_metrics'][-1] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=None)
        #st_res['metrics'][1], st_res['summarized_metrics'][1] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=1)
        st_res['metrics'][5], st_res['summarized_metrics'][5] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=5)
                
        #st_res['metrics-all'][1], st_res['summarized_metrics-all'][1] = compute_metrics(gt_users, train_users, nliked_users, st_res['recommended_ids'],st_res['recommended_ids'],k=1)
        st_res['metrics-all'][5], st_res['summarized_metrics-all'][5] = compute_metrics(gt_users, train_users, nliked_users, st_res['recommended_ids'],st_res['recommended_ids'],k=5)
        
        #st_res['metrics-full'][1], st_res['summarized_metrics-full'][1] = compute_metrics(gt_full, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=1)
        st_res['metrics-full'][5], st_res['summarized_metrics-full'][5] = compute_metrics(gt_full, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=5)
                
        #st_res['metrics-full-all'][1], st_res['summarized_metrics-full-all'][1] = compute_metrics(gt_full, train_users, nliked_users, st_res['recommended_ids'],st_res['recommended_ids'],k=1)
        st_res['metrics-full-all'][5], st_res['summarized_metrics-full-all'][5] = compute_metrics(gt_full, train_users, nliked_users, st_res['recommended_ids'],st_res['recommended_ids'],k=5)
                
        st_res['#users'] = len(users)

        group_results.append(st_res)
        
    return group_results


# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------

dir_path = './'

df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')
df_movies

df_ratings = pd.read_csv(dir_path + 'ml-25m/ratings.csv')

print('Done loading...')

min_ratings_movie = 25
min_ratings_user = 30 # esto puede ser 20 o 30
min_date = '2016-01-01'
df_ratings_min_date = df_ratings[df_ratings['timestamp'] >= datetime.strptime('2016-01-01', '%Y-%m-%d').timestamp()]

df_items_ratings = df_ratings_min_date.groupby(['movieId'])[['rating']].count().sort_values(by="rating", ascending=False)
df_items_ratings = df_items_ratings[df_items_ratings['rating'] > min_ratings_movie]

df_users_ratings = df_ratings_min_date[df_ratings_min_date['movieId'].isin(set(df_items_ratings.index))].groupby('userId')[['rating']].count().sort_values('rating',ascending=False)
df_users_ratings = df_users_ratings[df_users_ratings['rating'] > min_ratings_user]

df_ratings = df_ratings_min_date[df_ratings_min_date['userId'].isin(set(df_users_ratings.index))]

print('Done preprocessing...')

if not os.path.exists(dir_path + 'ml-25m/movies_genres.pickle'):
    df_movies['genres_'] = [x.split('|') for x in df_movies['genres']]
    df_movies__ = df_movies.explode('genres_')
    df_movies__['a'] = 1
    df_movies__ = df_movies__.reset_index(drop=True)
    df_movies__ = df_movies__.drop_duplicates()
    df_movies_genre = df_movies__.pivot(index='movieId',columns='genres_',values='a').fillna(0).astype(int)
    del df_movies__
    df_movies_genre.to_pickle(dir_path + 'ml-25m/movies_genres.pickle')

metrics_ = {}
metrics_['precision'] = precision_k_user()
metrics_['ndcg'] = ndcg_k_user()
# metrics_['coverage'] = coverage()
cdd = content_distance(items_path=dir_path + 'ml-25m/movies_genres.pickle')

diversity = individual_diversity(distance=cdd)
metrics_['novelty'] = individual_novelty(distance=cdd)

metrics_['rank_correlation-kendal'] = rank_correlation_k(df_ratings,corr_metric='kendall')
# metrics_['rank_correlation-spearman'] = rank_correlation_k(df_ratings,corr_metric='spearman')

movie_titles_set = {df_movies['title'].values[i] : df_movies['movieId'].values[i] for i in range(0,len(df_movies))}
movie_titles_set[None] = -1

ks = [1,5]

def search_file(dir_path,file):
    if '#' in file:
        ff_s = file.split('__')#[-1].split('.pickle')[0]
        if ff_s[-1].endswith('#.pickle') or ff_s[-1].startswith('groups#'):
            ff_s = ff_s[-2]
        else:
            ff_s = ff_s[-1].split('.pickle')[0]
            
    else:
        ff_s = '__'.join(file.split('__')[2:]).split('.pickle')[0]
    
    if '_it' in file:
        ff_s = ff_s.split('_it')[0]
    
    print('+++++++++++++++++++++++',ff_s)
    if not ff_s.endswith('u'):
        ff_s = ff_s[0:ff_s.index('u_')+1]
        
    print('----',ff_s)
    
    for f in os.listdir(dir_path):
        if not f.startswith(ff_s):
            continue
        if 'queries' in f:
            print(f)
            return f
    return None


import os

group_results_file = '__groupr__tree_results__.pickle' 

if os.path.exists(dir_path + group_results_file):
    group_results = pd.read_pickle(dir_path + group_results_file)
else:
    group_results = {}

already_computed = {}
for file in tqdm(os.listdir(dir_path)):
    
    if not file.startswith('__group__tree__'):
        continue
    
    rr = pd.read_pickle(dir_path + file)
    already_computed[file] = {k : len(v) for k,v in rr.items()}


for file in tqdm(os.listdir(dir_path)):

    if not file.startswith('results__'):
        continue
    
    if 'any' in file:
        continue
    
    print('--------------------------------',file)
    
    results = pd.read_pickle(dir_path + file) 
    groups = search_file(dir_path,file)
    print('=======',groups)
        
    base = '__'.join(groups.split('__')[:-1]) + '.pickle' 
    if not os.path.exists(dir_path + base):
        print('MISSING:',base)
        continue
        
    groups = pd.read_pickle(dir_path + groups)
    
   
    par = file.split('__')[1]
    print('____',par)
    group_results_file = f'__group__tree__{par}.pickle'
    if os.path.exists(dir_path + group_results_file):
        group_results = pd.read_pickle(dir_path + group_results_file)
    else:
        group_results = {}

    if len(results) != len(groups):
    
        if file in group_results:
            
            if len(results) <= len(group_results[file]):
                print('No new groups added')
                continue
    else:
        if file in group_results:
            continue
            
    if len(results) != len(groups):
    
        if group_results_file in already_computed and file in already_computed[group_results_file]:
            
            if len(results) <= already_computed[group_results_file][file]:
                print('No new groups added')
                continue
    else:
        if group_results_file in already_computed and file in already_computed[group_results_file]:
            continue
    
    if os.path.exists(dir_path + group_results_file): # solo levanta cuando haya que guardar algo nuevo
        group_results = pd.read_pickle(dir_path + group_results_file)
    else:
        group_results = {}
    
    base = pd.read_pickle(dir_path + base)
    print('Computing group results...',str(datetime.now())) 
    group_results[file] = compute_group_results(results,groups,base)
    
    with open(dir_path + group_results_file,'wb') as ff:
            pickle.dump(group_results,ff)

print('Done processing!',str(datetime.now()))




def summarize_results(results_raw,key='summarized_metrics',alt='avg',k=5):
    results = defaultdict(defaultdict(list).copy)
    for f in tqdm(results_raw):
        for i in range(0,len(results_raw[f])):
            
            if key not in results_raw[f][i]:
                continue
            for mm, vv in results_raw[f][i][key][k].items():
                if alt in vv: 
                    results[f][mm].append(vv[alt][0] if isinstance(vv[alt],tuple) else vv[alt]) 
                elif alt == 'zrecall' and 'orrelation' not in mm: 
                    kk = key.replace('summarized_','')

                    p = results_raw[f][i][kk][k][mm] 
                    if len(p) != 0:
                        p = [v for x,v in p.items() if x != 'all']
                        results[f][mm].append(sum(1 for x in p if x != 0)/len(p))
                elif alt == 'std' and 'orrelation' not in mm: 
                    kk = key.replace('summarized_','')

                    p = results_raw[f][i][kk][k][mm] 
                    if len(p) != 0:
                        p = [v for x,v in p.items() if x != 'all']
                        results[f][mm].append(np.std(p))
                elif alt == 'min-max' and 'orrelation' not in mm:
                    kk = key.replace('summarized_','')
#                    
                    p = results_raw[f][i][kk][k][mm] 
                    if len(p) != 0:
                        p = [v for x,v in p.items() if x != 'all']
                        results[f][mm].append(np.min(p) / np.max(p))
            
            if key.replace('summarized_','') in results_raw[f][i]:
                aa = results_raw[f][i][key.replace('summarized_','')][k]['diversity'][0]
                if 'all' in aa:
                    results[f]['diversity'].append(aa['all'])
        ll = list(results[f].keys())
        for mm in ll: 
            p = results[f][mm]
            results[f][mm] = np.nanmean(p)
            results[f][mm+'-std'] = np.nanstd(p)
            
    return  pd.DataFrame(results).T
    
ks = [5]
alts=['min','avg','max','all','median-n','avg-n','variance','zrecall','std','min-max']

all_results = defaultdict(defaultdict(defaultdict(dict).copy).copy)

for file in tqdm(os.listdir(dir_path)):

    if not file.startswith('__group__'):
        continue
    
    print('--------',file) 
    results = pd.read_pickle(dir_path + file)
    
    for k in ks:
        for alt in alts:
            print(k,alt)

            all_results[k]['summarized_metrics'][alt][file] = summarize_results(results,key='summarized_metrics',alt=alt,k=k)
            all_results[k]['summarized_metrics-all'][alt][file] = summarize_results(results,key='summarized_metrics-all',alt=alt,k=k)
            
            all_results[k]['summarized_metrics-full'][alt][file] = summarize_results(results,key='summarized_metrics-full',alt=alt,k=k)
            all_results[k]['summarized_metrics-full-all'][alt][file] = summarize_results(results,key='summarized_metrics-full-all',alt=alt,k=k)
            
import pickle
with open('__all_results___FULL.pickle','wb') as file:
    pickle.dump(all_results,file)
    
    
def summarize_boxplot(results_raw,key='summarized_metrics',alt='avg',k=5):
    results = defaultdict(defaultdict(list).copy)
    for f in tqdm(results_raw):
        for i in range(0,len(results_raw[f])):
            # print(results_raw[f][i][key])
            if key not in results_raw[f][i]:
                continue
            for mm, vv in results_raw[f][i][key][k].items():
                if alt in vv: # could be that they are not there
#                     print(mm,type(vv[alt]),vv[alt])
                    results[f][mm].append(vv[alt][0] if isinstance(vv[alt],tuple) else vv[alt]) 
                elif alt == 'zrecall' and 'orrelation' not in mm: # TODO: Here we can update the zrecall score that might be missing
                    kk = key.replace('summarized_','')
#                     print(kk)
                    p = results_raw[f][i][kk][k][mm] # esto deberia dejarme los de esa metrica sin summarizar
                    if len(p) != 0:
                        p = [v for x,v in p.items() if x != 'all']
                        results[f][mm].append(sum(1 for x in p if x != 0)/len(p))
                elif alt == 'std' and 'orrelation' not in mm: # TODO!!
                    kk = key.replace('summarized_','')
#                     print(kk)
                    p = results_raw[f][i][kk][k][mm] # esto deberia dejarme los de esa metrica sin summarizar
                    if len(p) != 0:
                        p = [v for x,v in p.items() if x != 'all']
                        results[f][mm].append(np.std(p))
                elif alt == 'min-max' and 'orrelation' not in mm:
                    kk = key.replace('summarized_','')
#                     print(kk)
                    p = results_raw[f][i][kk][k][mm] # esto deberia dejarme los de esa metrica sin summarizar
                    if len(p) != 0:
                        p = [v for x,v in p.items() if x != 'all']
                        results[f][mm].append(np.min(p) / np.max(p))
            
            # adds diversity to the analysis!
            if key.replace('summarized_','') in results_raw[f][i]:
                aa = results_raw[f][i][key.replace('summarized_','')][k]['diversity'][0]
                if 'all' in aa:
                    results[f]['diversity'].append(aa['all'])
       
    return  results
    
    
all_results = defaultdict(defaultdict(defaultdict(dict).copy).copy)

for file in tqdm(os.listdir(dir_path)):

    if not file.startswith('__group__'):
        continue
    
    print('--------',file) 
    results = pd.read_pickle(dir_path + file)
    
    for k in ks:
        for alt in alts:
            print(k,alt)

            all_results[k]['summarized_metrics'][alt][file] = summarize_boxplot(results,key='summarized_metrics',alt=alt,k=k)
            all_results[k]['summarized_metrics-all'][alt][file] = summarize_boxplot(results,key='summarized_metrics-all',alt=alt,k=k)
            
            all_results[k]['summarized_metrics-full'][alt][file] = summarize_boxplot(results,key='summarized_metrics-full',alt=alt,k=k)
            all_results[k]['summarized_metrics-full-all'][alt][file] = summarize_boxplot(results,key='summarized_metrics-full-all',alt=alt,k=k)
            
import pickle
with open('__all_results___BOXPLOT.pickle','wb') as file:
    pickle.dump(all_results,file)
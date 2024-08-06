import pandas as pd
import os
from sklearn import metrics
from collections import deque
import pickle
import numpy as np


class evaluation_metric():  

    def __init__(self):
        pass

    def get_scores(self, recs, **kwargs):
        pass

    def __str__(self):
        pass
# --------------------------------------------------------

from sklearn.metrics.pairwise import cosine_distances

class distance__():

    def __init__(self, **kwargs): 
        self.similarities = None
        if 'similarity_path' in kwargs:
            self.similarities = pd.read_pickle(kwargs['similarity_path'])

    def set_items(self,items):
        pass
       
    def compute_distances_reduced(self, nodesA, nodesB):

        nodesA = nodesA.intersection(self.items.index)  # en caso de que alguno de los que pase no esté
        nodesB = nodesB.intersection(self.items.index)

        if len(nodesA) == 0 or len(nodesB) == 0:
            return None, None, None

        if self.similarities is not None: # if similarities are pre-computed, we only select them from the df
            
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
        
        dd, _, _ = self.distance_.compute_distances_reduced(set(recs[0:k]), set(recs[0:k]))
        if dd is None:
            return dists
        
        dists['all'] = (dd.sum()).sum() / (len(dd) * (len(dd)-1)) 

        return dists

    def __str__(self):
        return 'diversity'

class individual_novelty(evaluation_metric):  

    def __init__(self, **kwargs):
        self.distance_ = kwargs['distance']

    def get_scores(self, recs, **kwargs):

        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        if len(recs[0:k]) > 0:
            known = kwargs['known']
            recs_set = set(recs[0:k])
            dists = {}
            
            for u, gt in known.items():  

                dd, _, _ = self.distance_.compute_distances_reduced(recs_set, set(gt))
                if dd is None:
                    continue

                dists[u] = dd.sum().sum() / (dd.shape[0] * dd.shape[1]) 
            
            return dists
        
        return {}
    
    def __str__(self):
        return 'novelty'

# --------------------------------------------------------

def matching(x,y):
    
    if y is None:
        return None
    
    if '(' in x:
        xs = x.split('(')
        xs = '('.join(xs[0:-1])
    else: 
        xs = x
    
    return y if y.startswith(xs) else None

class metric_summarizer():

    def __init__(self,df_movies):
        self.movie_titles_set = {df_movies['title'].values[i] : df_movies['movieId'].values[i] for i in range(0,len(df_movies))}
        self.movie_titles_set[None] = -1

    def summarize_metric(self,recs,metric,**kwargs):
        pass

    def transform_movies(self,movies,to_recommend=None): 
        
        if isinstance(movies[0],int):
            return movies
        
        if isinstance(movies[0],np.int64):
            return movies
        
        if movies[0].isdigit(): 
            return movies

        recs_ids = [self.movie_titles_set[x] if x in self.movie_titles_set else self.movie_titles_set[next((y for y in self.movie_titles_set if matching(x, y)), None)] for x in movies]
        recs_ids = [x for x in recs_ids if x != -1]        

        if to_recommend is not None:
            recs_ids = [x for x in recs_ids if x in to_recommend]

        return recs_ids

    def __str__(self):
        pass

class user_summarizer(metric_summarizer):
    
    def __init__(self,df_movies,how=None):
        super().__init__(df_movies)
        self.how = how
    
    def set_how(self,how):
        self.how = how

    def summarize_metric(self,recs,metric,**kwargs):
        recs = self.transform_movies(recs,kwargs.get('to_recommend',None))
        scores = metric.get_scores(recs,**kwargs)

        if len(scores) == 1 and 'all' in scores:
            return scores['all']

        how = self.how if self.how is not None else kwargs.get('how','avg')
        
        if how == 'min':
            return scores[min(scores, key=scores.get)]
        elif how == 'avg':
            return np.mean(list(scores.values()))
        elif how == 'max':
            return scores[max(scores, key=scores.get)]
        elif how == 'var':
            return np.var(list(scores.values()))
        elif how == 'median':
            return np.percentile(list(scores.values()),50) 

    def __str__(self):
        return 'user_summarizer#' + self.how

class group_summarizer(metric_summarizer):

    def __init__(self,df_movies,metric):
        super().__init__(df_movies,metric)
        
    def summarize_metric(self,recs,metric,**kwargs): 
        recs = self.transform_movies(recs,kwargs.get('to_recommend',None))

        known = kwargs['known'] 
        kwargs['known'] = {'all' : set().union(*[v for v in known.values()])}
        
        scores = metric.get_scores(recs,**kwargs)
        return scores['all']

    def __str__(self):
        return 'group_summarizer' 
    
# --------------------------------------------------------
# --------------------------------------------------------

# if __name__ == '__main__':

#     dir_path = './'
#     cdd = content_distance(items_path=dir_path + 'ml-25m/movies_genres.pickle')
#     # Una o la otra de las métricas
#     diversity = individual_diversity(distance=cdd)
#     novelty = individual_novelty(distance=cdd)

#     df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')

#     print('All loaded!')
    
#     group = {'user_ids': [146, 3, 4, 21],
#  'intersection': {91529, 134130},
#  'users_history': {146: [1246, 115713, 122906, 122920, 143355, 91529, 134130],
#   3: [4448, 8983, 31878, 102407, 176371, 91529, 134130],
#   4: [924, 1036, 6863, 7451, 152081, 91529, 134130],
#   21: [750, 62849, 84152, 87232, 88129, 91529, 134130]},
#  'to_recommend': [293,  1265,  1732,  2019,  2959,  3175,  4388,  6879,  27660,  59315,  60072,  63113,  68157,  68954,  82461,  99114,  109487,  122910,  122914,  130634,  139644,  143385,  168250, 182715,  187593]
#             }
    
#     recs = [294, 99114,  109487,  122910,  122914, 1265,  1732,  2019]

#     # calcula para cada usuario, y luego summariza con un mean, median, min...
#     user_summarizer = user_summarizer(df_movies)

#     print(user_summarizer.summarize_metric(recs,novelty,known=group['users_history']))
#     # print(user_summarizer.summarize_metric(recs,novelty,known=group['users_history'],how='var'))
#     # print(user_summarizer.summarize_metric(recs,novelty,known=group['users_history'],how='median'))
#     print(user_summarizer.summarize_metric(recs,diversity))

#     # junta todo el grupo en un único set y calcula con todo eso
#     # group_summarizer = group_summarizer(df_movies)

#     # print(group_summarizer.summarize_metric(recs,novelty,known=group['users_history']))
#     # print(group_summarizer.summarize_metric(recs,diversity,known=group['users_history']))

import pandas as pd
from datetime import datetime
import numpy as np
import random
import pickle
from tqdm import tqdm
import os 
from collections import defaultdict

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

import random
import numpy as np
import itertools

random.seed(42)

def sparse_corrcoef(A, B=None):

    if B is not None:
        A = sparse.vstack((A, B), format='csr')

    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))

    return coeffs
    
import scipy

class groups_generator():
    
    def __init__(self, ratings,threshold=None, how='dense', min_corated=-1, check_matching=False):
        self.ratings = ratings.pivot_table(columns='movieId', index='userId', values='rating').fillna(0)
        if how == 'dense':
            self.sim_matrix = np.corrcoef(self.ratings)
            self.sim_matrix = pd.DataFrame(self.sim_matrix,index=self.ratings.index,columns=self.ratings.index) 
        else:
            self.sim_matrix = scipy.sparse.csr_matrix(self.ratings.values)
            self.sim_matrix = sparse_corrcoef(self.sim_matrix)
            self.sim_matrix = pd.DataFrame(self.sim_matrix, index=self.ratings.index, columns=self.ratings.index)
        
        self.threshold = self.sim_matrix.median().median() if threshold is None else threshold
        
        self.min_corated = min_corated
        self.check_matching = check_matching
        self.last_group = None
        
        print('_______________________',min_corated)
        
        if self.min_corated > 0:
            print('Modifying according to corated!')
            # bb = self.ratings[self.ratings > 0]
            # dd = defaultdict(dict)

            # bb_index = list(bb.index) # users

            # for i in tqdm(range(0,len(bb_index))): # way too slow!!
                # for j in range(0,len(bb_index)):
                    # if i == j:
                        # continue

                    # if i in dd and j in dd[i]: cc = dd[i][j]
                    # elif j in dd and i in dd[j]: cc = dd[j][i]
                    # else:
                        # cc = bb.loc[[bb_index[i],bb_index[j]]].count()
                        # cc = len(cc[cc >= 2])
                        # dd[i][j] = cc
                    # if cc < min_corated:
                        # self.sim_matrix.at[bb_index[i],bb_index[j]]= -2 

            mm_path = dir_path + f'__AUX__modified_sim_matrix_min_corated_{min_corated}.pickle'
            if os.path.exists(mm_path):
                print('Loading matrix')
                self.sim_matrix = pd.read_pickle(mm_path)
            else:
                print('Computing matrix')
                bb = ratings.groupby('userId')[['movieId']].agg(set)
                #bb = bb[bb.index.isin(self.sim_matrix.index)] 
                bb = bb.reindex(self.sim_matrix.index)
                dd = defaultdict(dict)
                for i in tqdm(range(0,len(bb))):
                    for j in range(0,len(bb)):
                        if i == j:
                            continue
                            
                        if i in dd and j in dd[i]: cc = dd[i][j]
                        elif j in dd and i in dd[j]: cc = dd[j][i]
                        else:
                            cc = len(bb['movieId'].values[i].intersection(bb['movieId'].values[j]))
                            dd[i][j] = cc
                        if cc < min_corated:
                            #if bb.index[j] in gg.sim_matrix.columns:
                            self.sim_matrix[bb.index[j]].values[i] = -2
                with open(mm_path, 'wb') as file:
                    pickle.dump(self.sim_matrix,file)

                
    def get_groups_generator(type_):
        if type_ == "random": return random_group_generator
        if type_ == "similar": return similar_group_mix_combinatorial_generator
        if type_ == "similar_any": return similar_group_any_mix_generator
        if type_ == "divergent": return divergent_group_combinatorial_generator
        if type_ == "divergent_any": return divergent_group_any_combinatorial_generator
        return None
    
    def compute_average_similarity(self, group): 
        similarities = []
        aux = self.sim_matrix[self.sim_matrix.index.isin(group)][group]
        return ((aux.sum() - 1) / (len(aux) - 1)).mean()
    
    def generate_groups(self, group_sizes_to_create, group_number_to_create, **kwargs):
        groups_list = []
        
        if isinstance(group_sizes_to_create,int):
            group_sizes_to_create = [group_sizes_to_create]
        
        user_ids = list(self.ratings.index)
        for group_size in group_sizes_to_create:
            for i in tqdm(range(0,group_number_to_create)):
                group = self.create_group(user_ids, group_size)
                if group is None: # no need of continuing searching, it means that there's no group
                    break
                groups_list.append(
                    {
                        "group_size": group_size,
                        "group_similarity": self.__str__(), 
                        "group_members": group,
                        "avg_similarity": self.compute_average_similarity(group)
                    }
                )
            print(len(groups_list))
            self.reset() # once we get everything we need here, we reset
        return groups_list

    def check_matching_group(self,new_group):

        if not self.check_group:
            return True

        if self.last_group is None:
            return True
        
        return not new_group.intersection(self.last_group) > 0


    def __str__(self):
        if self.min_corated > 0:
            return f'_mcr{str(self.min_corated)}_'
        return ''

    def create_group(self, user_ids, group_size):
        pass
    
    def reset(self):
        return
    
    
class random_group_generator(groups_generator):

    def __str__(self):
        return 'random'
    
    def create_group(self, user_ids, group_size):
         return random.sample(user_ids, group_size)
    
            
class similar_group_combinatorial_generator(groups_generator):
    
    def __init__(self, ratings, threshold=None,min_corated=-1,check_group=False):
        
        super().__init__(ratings,threshold,min_corated=min_corated,check_gruop=check_group)
        self.reset()
    
    def __str__(self):
        return f'similar-group-combinatorial_{str(self.threshold)}{super().__str__()}'
    
    def reset(self):
        self.iterator__ = None
        self.group_size = 0
    
    def check_similarities(self,group): 
        
        if len(group) <= 1:
            return True
        
        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)][group]
        rest_sims = rest_sims[rest_sims >= self.threshold].dropna()
                
        return True if len(rest_sims) == len(group) else False
    
    def get_next_group(self):
        for x in self.iterator__:

            if not self.check_matching_group(x):
                return None

            passes_sims = self.check_similarities(list(x))

            return x if passes_sims else None
        return []
    
    def set_iterator(self,user_ids,group_size):
        if self.iterator__ is None or self.group_size != group_size:
            self.iterator__ = itertools.combinations(user_ids, group_size)
            self.group_size = group_size
    
    def create_group(self, user_ids, group_size):
        self.set_iterator(user_ids, group_size)
        
        group = None
        while group is None:
            group = self.get_next_group()
        
        if len(group) == 0:
            return None
        
        return list(group)


class similar_group_mix_combinatorial_generator(similar_group_combinatorial_generator):
    
    def __init__(self, ratings, threshold=None, max_combinations=4, min_corated=-1,check_maching=False):
        super().__init__(ratings,threshold,min_corated=min_corated,check_matching=check_maching)
        self.max_combinations = max_combinations
        self.groups = set()
    
    def __str__(self):
        return f'similar-group-combinatorial_{str(self.threshold)}{super().__str__()}'
        
    def set_iterator(self,user_ids, group_size):
        if self.iterator__ is None or self.group_size != group_size:
            print('iterator_size:', self.max_combinations if self.max_combinations < group_size else group_size)
            self.iterator__ = itertools.combinations(user_ids, self.max_combinations if self.max_combinations < group_size else group_size)
            self.group_size = group_size
            self.groups = set() 
        
    def get_missing_similar_users(self,group):
        user_ids = None
        
        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)]
        rest_sims = rest_sims.drop(columns=group)
        print('trying for:',group)
        for x in group:
            
            aa = rest_sims[(rest_sims.index == x)]
            aa = aa[aa >= self.threshold]
            aa = aa.dropna(axis=1).columns
            
            if len(aa) == 0:
                return None
#             print(aa)
            if user_ids is None:
                user_ids = set(aa)
            else:
                user_ids = user_ids.intersection(set(aa))
        
        if len(user_ids) < (self.group_size - self.max_combinations): # si no encontré los que quería
            print('No users in the intersection')
            return None
        
        it_rest = itertools.combinations(user_ids, self.group_size - self.max_combinations)
        user_ids = None
        for y in it_rest:
            if self.check_similarities(list(y)):
                user_ids = list(y)
                break
    
        return user_ids
    
    def get_next_group(self):
        for x in self.iterator__:
            
            if not self.check_matching_group(x):
                return None

            passes_sims = self.check_similarities(list(x))
            if not passes_sims:
                return None
            
            x = set(x)

            if len(x) == self.group_size: # there's no need of continuing the analysis as we already have the group
                return x
            
            complement_ids = self.get_missing_similar_users(x)

            if complement_ids is None or len(complement_ids) == 0:
                return None
        
            x.update(complement_ids)
            
            if x in self.groups:

                return None
            
            self.groups.add(frozenset(x))
            
            return x 
        return []
        
class similar_group_any_generator(similar_group_combinatorial_generator): 
    
    def __str__(self):
        return f'similar-group-any_{str(self.threshold)}{super().__str__()}'
    
    def check_similarities(self,group): 
        if len(group) <= 1:
            return True

        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)][group]
        rest_sims = rest_sims[rest_sims >= self.threshold]

        rest_sims = rest_sims.dropna(thresh=2) # diagonal + an additional value
        return True if len(rest_sims) == len(group) else False
    
    
class similar_group_any_mix_generator(similar_group_mix_combinatorial_generator): 
    
    def __str__(self):
        return f'similar-group-any_{str(self.threshold)}{super().__str__()}'
    
    def check_similarities(self,group):  
        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)][group]
        rest_sims = rest_sims[rest_sims >= self.threshold]
        rest_sims = rest_sims[list(rest_sims.index)] 
        rest_sims = rest_sims.values
        np.fill_diagonal(rest_sims,np.nan)
        rest_sims = pd.DataFrame(rest_sims)

        rest_sims = rest_sims.dropna(how='all') 
        return True if len(rest_sims) == len(group) else False
    
    
    def get_missing_similar_users(self,group):
        user_ids = set()
        
        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)]
        rest_sims = rest_sims.drop(columns=group)
        
        for x in group:
            
            if not self.check_matching_group(x):
                return None

            aa = rest_sims[(rest_sims.index == x)]
            
            aa = aa[aa >= self.threshold]
                        
            aa = aa.dropna(axis=1).columns
                        
            if len(aa) == 0:
                return None
            
  
            user_ids = user_ids.union(set(aa))
        
        if len(user_ids) < (self.group_size - self.max_combinations): # si no encontré los que quería
            print('No users in the union')
            return None
        
        
        user_ids = list(random.sample(list(user_ids), self.group_size - self.max_combinations))
            
        return user_ids
    
    
class divergent_group_combinatorial_generator(groups_generator): # minority
    
    def __init__(self, ratings, threshold=None, threshold_divergence=None, min_corated=-1, under_generator=similar_group_mix_combinatorial_generator,check_matching=False):
        super().__init__(ratings,threshold,min_corated=min_corated,check_matching=check_maching)
        self.reset()
        self.user_ids = list(self.sim_matrix.index)
        random.shuffle(self.user_ids)
        self.threshold_divergence = self.threshold if threshold_divergence is None else threshold_divergence
        
        self.similar_generator = under_generator(ratings,threshold)
        self.groups = set()
                
    def __str__(self):
        return f'divergent-group-combinatorial_{str(self.threshold)}_{str(self.threshold_divergence)}{super().__str__()}'
        
    def check_dissimilarities(self,group,extra): 
        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)][[extra]]

        return not rest_sims[rest_sims >= self.threshold_divergence].any().values[0]
            
    def create_group(self, user_ids, group_size):
        
        group = None
        while group is None:
            group = set(self.similar_generator.create_group(user_ids, group_size-1)) 
            for u in self.user_ids: 
                if u in group:
                    continue
                if self.check_dissimilarities(group,u): # habría que chequear que el grupo no esté repetido
                    group.add(u)
                    if group not in self.groups:
                        self.groups.add(frozenset(group))
                        print(group)
                        return list(group)
                    group.discard(u)
            group = None # none of the users was useful
        

        return None
    
    
class divergent_group_any_combinatorial_generator(divergent_group_combinatorial_generator):
                
    def __str__(self):
        return f'divergent-group-any-combinatorial_{str(self.threshold)}_{str(self.threshold_divergence)}{super().__str__()}'
        
    def check_dissimilarities(self,group,extra): # similarities between group is already checked
        rest_sims = self.sim_matrix[self.sim_matrix.index.isin(group)][[extra]]

        return rest_sims[rest_sims < self.threshold_divergence].any().values[0]
        

# --------------------------------------------------

strategy = ['similar_any','divergent','divergent_any','similar','random'] 
members_in_group = [2,4,5,8]
cant_groups = [100,500,1000]

min_corated = 5

for strat in strategy:

    print('-------------------------------- ',strat,str(datetime.now()))
    generator = groups_generator.get_groups_generator(strat)
    generator = generator(df_ratings_final,min_corated=min_corated)

    for mg in members_in_group:
        for cg in cant_groups: 
        
            file_name = dir_path + f'groups_min{str(min_ratings_user)}_{generator}_{cg}g_{mg}u.pickle'
            print(file_name)
            
            if os.path.exists(file_name):
                continue
            
            print('No existe')
            
            print(mg,cg,str(datetime.now()))
            groups = generator.generate_groups(mg,cg)
            with open(file_name,'wb') as file:
                pickle.dump(groups,file)
                
            
print('Done!',str(datetime.now()))
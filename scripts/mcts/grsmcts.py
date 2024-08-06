
# Implementation of a Monte Carlo Tree Search algorithm for a recommender system

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import pickle

import networkx as nx
import sys

from strategies import MCTSNode
from strategies import RandomExpansion, RandomSimulation, MCTSNode
from evaluation import user_summarizer, metric_summarizer, individual_diversity, individual_novelty


class GroupMovieRecommenderSystem:

    def __init__(self, group_id, store, initial_state=[], max_rollouts=0, max_expansions=None, c_param=1.4):
        
        self.store = store
        self.group_id = group_id
        self.movies = store.get_movies_to_recommend(group_id)        
        print("Group:", group_id, "- movies to recommend:", self.movies)
        
        self.max_rollouts = sys.maxsize if max_rollouts is None else max_rollouts
        self.max_expansions = sys.maxsize if max_expansions is None else max_expansions
        # print("max_rollouts:", self.max_rollouts)
        # print("max_expansions:", self.max_expansions)

        # c_param: This parameter directly controls the balance between exploration and exploitation:
        # High c_param Value: Increases the influence of the exploration term, leading to more exploration of less-visited nodes.
        # Low c_param Value: Reduces the influence of the exploration term, leading to more exploitation of nodes with higher average rewards.
        self.c_param = c_param

        self.initial_state = initial_state
        self.metric = "diversity"

        self.simulation_strategy = None # RandomSimulation()
        self.expansion_strategy = None # RandomExpansion()
        self.reward_strategy = None
    
    def set_expansion_strategy(self, strategy):
        self.expansion_strategy = strategy
    
    def set_reward_strategy(self, strategy):
        self.reward_strategy = strategy
    
    def set_simulation_strategy(self, strategy):
        self.simulation_strategy = strategy

    def recommend_movies(self, top_k=None, n_iterations=1000, random_seed=None,metric='diversity'):

        if random_seed is not None:
            random.seed(random_seed)
        print("max_rollouts:", self.max_rollouts)
        print("max_expansions:", self.max_expansions)                                         

        self.root = MCTSNode(self.initial_state, None, self.movies, self.c_param, ntype="root")
        self.root.set_expansion_strategy(self.expansion_strategy)
        self.root.set_top_k(top_k)
        self.metric = metric
   
        for _ in range(n_iterations):
            node = self.root

            expansions = 0
            # Selection and expansion
            while (not node.is_terminal()) and (expansions < self.max_expansions):
                if not node.is_fully_expanded():
                    node = node.expand()
                    expansions += 1
                else:
                    node = node.best_child()
            
            # Simulation
            state = node.state

            rollouts = 0
            while not self.is_terminal_state(state,top_k) and rollouts < self.max_rollouts:
                state, n = self.simulate_next_state(node, state)
                rollouts += n 
            if rollouts > 0:
                node.set_type("simulated")
                node.state = state                                                           

            # Backpropagation
            reward = self.calculate_reward(state)
            node.backpropagate(reward)
        
        # Select the best child node
        best_state = self.root.best_child().state
        print("Best state:", best_state)
        # return best_state
        return (self.root.best_path()[-1]).state
    
    def is_terminal_state(self, state, k=None):
        limit = k if k is not None else len(self.movies)
        return len(state) == limit
    
    # This can be a strategy (LLM for guided simulation)
    # e.g., use LLM to suggest the next movie based on the current state
    def simulate_next_state(self, node, state):
        return self.simulation_strategy.simulate_next_state(node, state, self.movies)
    
    # This should be a strategy or something domain-specific for movies (LLM for state evaluation)
    # e.g., use LLM to evaluate the current state and predict the group satisfaction
    # Alternatively, we need to take into account the group members, not only the movie rates
    def calculate_reward(self, state):
        # total_score = 0
        # count = 0
        # for movie_id in state:
        #     ratings = self.store.get_movie_ratings(movie_id)
        #     total_score += np.mean(ratings) if len(ratings) > 0 else 0
        #     count += 1
        # reward = total_score / count if count > 0 else 0
        
        metric = self.reward_strategy.compute_score(state, group_id=self.group_id)
        # print("Reward= ", reward, "Diversity=", metric)

        return metric # reward

    def to_networkx(self):
        """
        Convert an MCTS tree to a Networkx graph.
        
        Parameters:
        node (MCTSNode): The root node of the MCTS tree.
        
        Returns:
        G (networkx.DiGraph): The corresponding Networkx directed graph.
        """
        node = self.root

        g = nx.DiGraph()
        
        def add_edges(node):
            for child in node.children:
                g.add_edge(str(node.state), str(child.state))
                add_edges(child)
        
        g.add_node(str(node.state))
        add_edges(node)

        return g


class MoviesStore:

    def __init__(self, name, ratings, movies) -> None:
        self.name = name

        self.ratings_df = ratings
        self.movies_df = movies
        print("movies:", self.movies_df.shape)
        print("ratings:", self.ratings_df.shape)

        self.groups = []
        self.users = list(self.ratings_df['userId'].unique())
        self.movies = list(self.movies_df['movieId'].unique())
        print("#users:", len(self.users))
        print("#movies (from movies)", len(self.movies))
        print("#movies (from ratings)", len(self.ratings_df['movieId'].unique()))
        self.metric_summ = metric_summarizer(movies)

    def find_movie_ids(self, movies):
        return self.metric_summ.transform_movies(movies)

    def get_users(self):
        return self.users
    
    def get_movies(self):
        return self.movies

    def get_groups(self):
        return list(range(0, len(self.groups)))
    
    def get_movie_ratings(self, movie_id):
        if movie_id in self.movies:
            return self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating'].values
        return []

    def get_group_members(self, group_id):
        if group_id < len(self.groups):
            return self.groups[group_id]['user_ids']
        return []

    def get_rated_movies(self, user_id):
        if user_id in self.users:
            df = self.ratings_df[self.ratings_df['userId']==user_id]
            return list(df['movieId'].unique())
        return []
    
    def get_group_movies(self, group_id): 
        # members = self.get_group_members(group_id)
        # df = self.ratings_df[self.ratings_df['userId'].isin(members)] 
        # return list(df['movieId'].unique())
        return self.get_personal_rated_movies(group_id,None)
    
    def get_movies_to_recommend(self, group_id):
        if group_id < len(self.groups):
            return list(self.groups[group_id]['to_recommend'])
        return []

    def get_common_rated_movies(self, group_id):
        if group_id < len(self.groups):
            return list(self.groups[group_id]['intersection'])
        return []

    def get_personal_rated_movies(self, group_id, user_id=None):
        if (group_id < len(self.groups)):
            if (user_id is not None) and (user_id in self.users):
                return list(self.groups[group_id]['users_history'][user_id])
            elif user_id is None:
                return [x for l in list(self.groups[group_id]['users_history'].values()) for x in l]
        return []
    
    def get_movie_descriptions(self, movie_ids):
        return [f'"{x}"' for x in self.movies_df[self.movies_df['movieId'].isin(movie_ids)]['title'].tolist()]
    
    # @private
    # def _compute_movie_intersection(self, group_id):
    #     members = self.get_group_members(group_id)
    #     setlist = [set(self.get_rated_movies(x)) for x in members]
    #     return set.intersection(*setlist)

    # @private
    # def _compute_personal_movies(self, group_id, common_movies):
    #     members = self.get_group_members(group_id)
    #     personal_movies = {u: list(set(self.get_rated_movies(u))-common_movies) for u in members}
    #     return personal_movies

    # def create_groups(self, p=5, k=None):
    #     users = self.users.copy()
    #     random.shuffle(users)
    #     n_groups = int(len(self.users) / p)
    #     # print(n_groups)
    #     groups = np.array_split(users, n_groups)
    #     print(len(groups), "groups created ("+str(p)+" users each)")
    #     self.groups = dict()
    #     if k is not None:
    #         enum_groups = list(enumerate(groups))[0:k]
    #     else:
    #         enum_groups = list(enumerate(groups))
    #     for g,m in tqdm(enum_groups, "configuring profiles"):
    #         self.groups[g] = dict()
    #         self.groups[g]['members'] = list(m)
    #         common = self._compute_movie_intersection(g)
    #         if len(common) > 0:
    #             print(g, "common: ", len(common))
    #         self.groups[g]['common_movies'] = list(common)
    #         self.groups[g]['personal_movies'] = self._compute_personal_movies(g, common)
    #     return self.groups
    
    def save_profiles(self, filename):
        pass

    def load_profiles(self, filename,base_recs_file=None,n_recs=None):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        # dict_keys(['user_ids', 'intersection', 'users_history', 'to_recommend'])
        self.groups = dataset
        n = len(self.groups)
        p = 0
        if n > 0:
            p = len(self.groups[0]['user_ids'])
        print(n, "groups created ("+str(p)+" users each)")

        if base_recs_file is None:
            self.recommendations = [{u:[] for u in x['user_ids']} for x in self.groups] 
        else:
            print('Loading base-recommendations...',base_recs_file)
            recs = pd.read_pickle(base_recs_file)

            self.base_recommendations = []

            for i in range(0,len(self.groups)):
                all_items = set()
                uss = self.groups[i]['user_ids']
                uu = {}
                for u in uss:
                    rr = [(k,v) for k,v in recs[u].items()]
                    rr.sort(key=lambda tup: tup[1],reverse=True)
                    if n_recs is None:
                        uu[u] = [x[0] for x in rr if x[0] in self.groups[i]['to_recommend']]
                    else:
                        uu[u] = [x[0] for x in rr[0:n_recs if n_recs!=-1 else len(rr)]]
                    
                    all_items.update(uu[u])
                self.base_recommendations.append(uu)
              
                self.groups[i]['to_recommend'] = all_items #set()

    def get_base_recommendations(self,group_id,user=None): 
        
        if group_id >= len(self.base_recommendations):
            return {}
        
        if user is None:
            return self.base_recommendations[group_id]
    
        return self.base_recommendations[group_id].get(user,{})


# ----------- MAIN -----------

# dir_path = './'

# ratings_df = pd.read_csv(dir_path + 'ml-25m/ratings.csv')
# movies_df = pd.read_csv(dir_path + 'ml-25m/movies.csv')

# Initialize the group recommender system
# group_recommender_system = GroupMovieRecommenderSystem(ratings_df, movies_df, group_size=3)

# Get recommended movies for the group
# recommended_movie_ids = group_recommender_system.recommend_movies(n_iter=1000)

# Print recommended movie titles
# recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].tolist()
# print("Recommended movies for the group:")
# for movie in recommended_movies:
    # print(movie)


# store = MoviesStore(name="dataset", ratings=ratings_df, movies=movies_df)

# pickle_filename = 'groups_min30_similar-group-combinatorial_0.15036991144527295_1000g_2u__gs_threliked-4_trainratio-0.8__queries_trainonly-5_testonly-5_nlikedonly-5_1vs_ids.pickle'
# recs_base_filename = 'recommendations_implicit_restricted_training_all_users__groups_min30_similar-group-combinatorial_0.15036991144527295_1000g_2u__gs_threliked-4_trainratio-0.8.pickle'
# n = 2

# store.load_profiles(pickle_filename,recs_base_filename,n)
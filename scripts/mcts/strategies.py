from abc import ABC, abstractmethod

import random
import numpy as np

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

from evaluation import user_summarizer, metric_summarizer, individual_diversity, individual_novelty


class MCTSNode:

    def __init__(self, state, parent, movies, c_param, ntype="undefined"):
        self.state = state # It is the list of movies being actually recommended
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.movies = movies # Movies available to recommend to the group
        self.c_param = c_param
        self.type = ntype
        
        self.top_k = None # 5
        self.expansion_strategy = None # RandomExpansion()
    
    def set_expansion_strategy(self, strategy):
        self.expansion_strategy = strategy
    
    def set_type(self, ntype):
        self.type = ntype
    
    def set_top_k(self, k):
        self.top_k = k
    
    def is_terminal(self):
        limit = self.top_k if self.top_k is not None else len(self.movies)
        return len(self.state) == limit
    
    def get_available_actions(self, skip_used_actions=False):
        used_actions = set(self.state)
        if skip_used_actions:
            for child in self.children:
                used_actions.update(child.state)
        possible_actions = [m for m in self.movies if m not in used_actions]
        return possible_actions
    
    def is_fully_expanded(self):
        possible_actions = self.get_available_actions()
        return len(possible_actions) == 0
        
    # This can be a strategy (LLM for action selection)
    # e.g., use LLM to suggest the next movie based on the current state
    def expand(self):
        return self.expansion_strategy.expand_node(self)
    
    def best_child(self): # Best child according to the UCB1 formula
        choices_weights = [
            (child.value / child.visits) + self.c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def best_path(self):
        best_path = []
        current_node = self
        
        while current_node.children:
            # Select child with the highest value (reward)
            best_child = max(current_node.children, key=lambda child: child.value)
            best_path.append(best_child)
            current_node = best_child
        
        return best_path
    
    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


class ExpansionStrategy(ABC):
    
    @abstractmethod
    def expand_node(self, node):
        pass

    def __str__(self):
        pass

class RandomExpansion(ExpansionStrategy):
    
    def expand_node(self, node):
        if node.is_fully_expanded():
            return None
        untried_actions = node.get_available_actions()
        action = random.choice(untried_actions)
        new_state = node.state + [action]
        child_node = MCTSNode(new_state, node, node.movies, node.c_param, ntype="expansion")
        child_node.set_expansion_strategy(node.expansion_strategy)
        child_node.set_top_k(node.top_k)
        node.children.append(child_node)
        return child_node
    
    def __str__(self):
        return 'randomexpansion'
        
    def set_group(self,group_id):
        pass


class LLMBase(ABC):

    SYSTEM_PROMPT = """You are a helpful movie recommender system. Your task is to recommend movies to a group of people based on their waching history.
        You will receive:
            * The group preferences.
            * The individual user preferences.
            * The set of movies to recommend.

        Your task is to use all the provided information to generate a list of recommended movies. You have access to all the information you need.
        """
    
    PROMPT = """
        {intersection}

        These are the "individual user preferences":
        {users_history}

        Movies to recommend: {to_recommend}

        Your task is the following:
            1. Using the "group preferences" and the "individual user preferences", pick {k} movies from "movies to recommend" and sort them based on how well they would satisfy the group as a whole. Position 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
            2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

        All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences" and the "movies to recommend".
        Note that the order of movies in the "movies to recommend" list does not reflect the group preferences. Use only "movies to recommend" for your answer. Do not add any extra movie. 

        Your JSON answer:
        """

    def __init__(self, llm, store, group_id=None, k=1,llm_name=None):
        self.llm = llm
        self.store = store
        self.group_id = group_id
        self.top_k = k
        self.output_parser = SimpleJsonOutputParser()
        self.llm_name = llm_name
    
    def set_group(self, group_id):
        self.group_id = group_id
    
    def _format_intersection(self, intersection):
        if len(intersection) == 0:
            result = 'There are no "group preferences"'
        else:
            # print("Intersection:", intersection)
            descriptions = self.store.get_movie_descriptions(intersection)
            result = "These are the 'group preferences': " + ", ".join(descriptions)
        return result
    
    def _format_recommendations(self, to_recommend):
        recs = self.store.get_movie_descriptions(to_recommend)
        return ", ".join(recs)
    
    def _format_users_history(self, group_id):
        users = self.store.get_group_members(group_id)
        all_history = []
        for u in users:
            history = self.store.get_personal_rated_movies(group_id, u)
            descriptions = self.store.get_movie_descriptions(history)
            s = "- User "+str(u)+": " + ", ".join(descriptions)
            all_history.append(s)
        return "\n"+"\n".join(all_history)

    def _call_llm(self, node):
        
        model_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT), # system role
            ("human", self.PROMPT) # human, the user text   
        ])

        intersection = self.store.get_common_rated_movies(self.group_id)
        to_recommend = [m for m in node.movies if m not in node.state]
        params = {
            "k": self.top_k,
            "intersection": self._format_intersection(intersection),
            "users_history": self._format_users_history(self.group_id),
            "to_recommend": self._format_recommendations(to_recommend)
        }

        message = model_prompt.format(**params)
        # print("Prompting:", message)

        chain = model_prompt | self.llm | self.output_parser
        response = chain.invoke(params)
        print("Response:", response)

        return response['movies']
    
class LLMExpansion(LLMBase, ExpansionStrategy):

    def __init__(self, llm, store, group_id=None, k=1):
        super().__init__(llm, store)
        self.group_id = group_id
        self.top_k = k
    
    def set_group(self, group_id):
        self.group_id = group_id                                                    
    def expand_node(self, node):
        if node.is_fully_expanded():
            return None
        
        untried_actions = self._call_llm(node)
        # print("Before:", untried_actions)
        untried_actions = self.store.find_movie_ids(untried_actions)
        # print("After:", untried_actions)

        # untried_actions = node.get_available_actions()
        # action = untried_actions[0] # random.choice(untried_actions)
        action = random.choice(untried_actions)

        new_state = node.state + [action]
        child_node = MCTSNode(new_state, node, node.movies, node.c_param, ntype="expansion")
        child_node.set_expansion_strategy(node.expansion_strategy)
        child_node.set_top_k(node.top_k)
        node.children.append(child_node)
        return child_node

    def __str__(self):
        return 'LLMexpansion#' + self.llm_name

class SimulationStrategy(ABC):
    
    @abstractmethod
    def simulate_next_state(self, state, actions):
        pass
        
    def __str__(self):
        pass


class RandomSimulation(SimulationStrategy):

    def __init__(self, k):
        self.top_k = k
    
    def simulate_next_state(self, node, state, actions):
        untried_actions = [a for a in actions if a not in state]
        if not untried_actions:
            return state
        # actions = [random.choice(untried_actions)]
        actions = untried_actions
        if len(untried_actions) > self.top_k:
            actions = random.sample(untried_actions, self.top_k)
        return state + actions, len(actions)
        
    def __str__(self):
        return 'RandomSim#' + str(self.top_k)
     
    def set_group(self, group_id):
        pass


class LLMSimulation(LLMBase, SimulationStrategy):

    def __init__(self, llm, store, llm_name, k, group_id=None):
        super().__init__(llm, store)
        self.group_id = group_id
        self.top_k = k
        self.llm_name = llm_name
        
    def set_group(self, group_id):
        self.group_id = group_id
    
    def simulate_next_state(self, node, state, actions):
        # untried_actions = [a for a in actions if a not in state]
        untried_actions = super()._call_llm(node)
        # print("Before:", untried_actions)
        untried_actions = self.store.find_movie_ids(untried_actions)
        # print("After:", untried_actions)

        if not untried_actions:
            return state
        
        actions = untried_actions
        if len(untried_actions) > self.top_k:
            actions = untried_actions[0:self.top_k]
        return state + actions, len(actions)

    
    def __str__(self):
         return 'LLMSim#' + self.llm_name + '#' + str(self.top_k)
    
class RewardStrategy(ABC):
    
    @abstractmethod
    def compute_score(self, recommendations, group_id=None):
        pass

    def __str__(self):
        pass

class MetricReward(RewardStrategy):
    
    def __init__(self, store, movies, cdd, metric, how):
        self.store = store

        self.diversity = None
        self.novelty = None 
        if cdd is not None:
            if metric == 'diversity':
                self.metric = individual_diversity(distance=cdd)
            else:
                self.metric = individual_novelty(distance=cdd)
        self.summarizer = user_summarizer(movies,how=how)
    
    def compute_score(self, recommendations, group_id=None):
        users_history = dict()
        for u in self.store.get_group_members(group_id):
            users_history[u] = self.store.get_personal_rated_movies(group_id, u)
        return self.summarizer.summarize_metric(recommendations, self.metric, known=users_history)
        
    def __str__(self):
        return 'metricreward#' + self.summarizer.__str__() + '#' + self.metric.__str__()
    
class LLMReward(RewardStrategy):

    SYSTEM_PROMPT = """You are a helpful movie recommender system. Your task is to assess movies for a group of people based on their waching history.
        You will receive:
            * The group preferences.
            * The individual user preferences.
            * A list of movies recommended to the group.

        Your task is to use all the provided information to generate a score for the recommended movies. You have access to all the information you need.
        """
    

    def __init__(self, llm, store, group_id=None, llm_name=None):
        self.llm = llm
        self.llm_name = llm_name
        self.store = store
        self.group_id = group_id
        self.output_parser = SimpleJsonOutputParser()
        self.PROMPT = """
        {intersection}

        These are the "individual user preferences":
        {users_history}

        Recommended movies: {to_recommend}

        Your task is the following:
            1. Using the "group preferences" and the "individual user preferences", generate a numeric score for the "recommended movies" based on how well they satisfy the group as a whole. The score should range between 0 and 100, where 0 means no satisfaction and 100 means full satisfaction.
            2. In your assessment also consider the degrees of diversity and novelty of each of the "recommended movies" with respect to the "group preferences" and "individual group preferences". The higher the diversity or the novelty, the better.
            3. Return your answer in a JSON format including the key 'score' and a single integer value for your score.

        All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences" and the "recommended movies".
        Do not include anything else, except your numeric score, in your answer. 

        Your JSON answer:
        """
    
    def set_group(self, group_id):
        self.group_id = group_id
    
    def _format_intersection(self, intersection):
        if len(intersection) == 0:
            result = 'There are no "group preferences"'
        else:
            # print("Intersection:", intersection)
            descriptions = self.store.get_movie_descriptions(intersection)
            result = "These are the 'group preferences': " + ", ".join(descriptions)
        return result
    
    def _format_recommendations(self, to_recommend):
        recs = self.store.get_movie_descriptions(to_recommend)
        return ", ".join(recs)
    
    def _format_users_history(self, group_id,method):
        users = self.store.get_group_members(group_id)
        all_history = []
        for u in users:
            history = method(group_id, u)
            descriptions = self.store.get_movie_descriptions(history)
            s = "- User "+str(u)+": " + ", ".join(descriptions)
            all_history.append(s)
        return "\n"+"\n".join(all_history)
    
    def _call_llm(self, recommendations,group_id):
        
        print('=====',len(recommendations))
        
        model_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT), # system role
            ("human", self.PROMPT) # human, the user text   
        ])
        
        params = self.get_prompt_params(recommendations, group_id)

        # message = model_prompt.format(**params)
        # print("Prompting:", message)
        
        chain = model_prompt | self.llm | self.output_parser
        response = chain.invoke(params)
        print("Response:", response)

        return response

    def get_prompt_params(self, recommendations, group_id):

        rr = self._format_recommendations(recommendations)
        print(rr)
        
        intersection = self.store.get_common_rated_movies(group_id)
        params = {
            "intersection": self._format_intersection(intersection),
            "users_history": self._format_users_history(group_id,self.store.get_personal_rated_movies),
            "to_recommend": rr
        }
        
        return params
    
    def compute_score(self, recommendations, group_id=None, metric=None):
        response =  self._call_llm(recommendations,group_id)
        if 'final_score' in response:
            return response['final_score'] / 100.0
        
        if 'score' in response:
            return response['score'] / 100.0
        
        response = response['movies']
        xx = 0
        for x in response: 

            if 'score_diversity' in x and 'score_novelty' in x and 'score_fairness' in x:
                xx += (x['score_diversity'] + x['score_novelty'] + x['score_fairness']) / 3.0
            elif 'score_diversity' in x and 'score_novelty' in x:
                xx += (x['score_diversity'] + x['score_novelty']) / 2.0
            elif 'score_diversity' in x and 'score_fairness' in x:
                xx += (x['score_diversity'] + x['score_fairness']) / 2.0
            elif 'score_fairness' in x and 'score_novelty' in x:
                xx += (x['score_novelty'] + x['score_fairness']) / 2.0
            elif 'score_diversity' in x:
                xx += x['score_diversity']
            elif 'score_novelty' in x:
                xx += x['score_novelty']
            elif 'score_fairness' in x:
                xx += x['score_fairness']
            elif 'score' in x:
                xx += x['score']
        xx = xx / len(response)
        print('Score final:',xx)
        return xx / 100.0
    
    
    def __str__(self):
        return 'LLMreward#' + self.llm_name
        
        
class LLMRewardDN(LLMReward):

    SYSTEM_PROMPT = """You are a helpful movie recommender system. Your task is to assess movies for a group of people based on their waching history.
        You will receive:
            * The group preferences.
            * The individual user preferences.
            * A list of movies recommended to the group.

        Your task is to use all the provided information to generate a score for the recommended movies. You have access to all the information you need.
        """
    

    def __init__(self, llm, store, group_id=None, llm_name=None):
        super().__init__(llm,store,group_id,llm_name)
        self.PROMPT = """
        {intersection}

        These are the "individual user preferences":
        {users_history}

        Recommended movies: {to_recommend}

        Your task is the following:
            1. Using the "group preferences" and the "individual user preferences", generate a numeric score for each of the "recommended movies" based on how well they satisfy the group as a whole. The score should range between 0 and 100, where 0 means no satisfaction and 100 means full satisfaction. Generate a score for each movie in the list.
            2. The score should reflect the degrees of diversity and novelty of each of the "recommended movies" with respect to the "group preferences" and "individual group preferences". The higher the diversity or the novelty, the better. 
            3. The order of the recommended movies is important. With the scores computed by movie, compute a score for the whole list, considering the individual scores for each movie, their position in the ranking, and the relationship between a movie and the ones ranked before.
            4. Return your answer in a JSON format including the key 'movies' and, for each movie, the key 'name' with the name of the movie, and the key 'score_diversity', 'score_novelty' and 'score' with a numeric value for your diversity, novelty and  score.
            
        All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences" and the "recommended movies".
        Do not include anything else, except the name of all movies and your numeric score, in your answer. 

        Your JSON answer:
        """
           
    def compute_score(self, recommendations, group_id=None, metric=None):
        response =  self._call_llm(recommendations,group_id)['movies']
        xx = 0
        for x in response:
            if 'score_diversity' in x and 'score_novelty' in x:
                xx += (x['score_diversity'] + x['score_novelty']) / 2.0
            elif 'score_diversity' in x:
                xx += x['score_diversity']
            elif 'score_novelty' in x:
                xx += x['score_novelty']
            elif 'score' in x:
                xx.append(x['score'])
        xx = xx / len(response)
        print('Score final:',xx)
        return xx / 100.0
    
    def __str__(self):
        return 'LLMrewardDN#' + self.llm_name
        

class LLMRewardDNR(LLMReward):

    SYSTEM_PROMPT = """You are a helpful movie recommender system. Your task is to assess movies for a group of people based on their waching history.
        You will receive:
            * The group preferences.
            * The individual user preferences.
            * A list of movies recommended to the group.

        Your task is to use all the provided information to generate a score for the recommended movies. You have access to all the information you need.
        """
    

    def __init__(self, llm, store, group_id=None, llm_name=None):
        super().__init__(llm,store,group_id,llm_name)
        self.PROMPT = """
        {intersection}

        These are the "individual user preferences":
        {users_history}

        Recommended movies: {to_recommend}

        Your task is the following:
            1. Using the "group preferences" and the "individual user preferences", generate a numeric score for each of the "recommended movies" based on how well they satisfy the group as a whole. The score should range between 0 and 100, where 0 means no satisfaction and 100 means full satisfaction. Generate a score for each movie in the list.
            2. The score should reflect the degrees of diversity and novelty of each of the "recommended movies" with respect to the "group preferences" and "individual group preferences". The higher the diversity or the novelty, the better. 
            3. The order of the recommended movies is important. With the scores computed by movie, compute a score for the whole list, considering the individual scores for each movie, their position in the ranking, and the relationship between a movie and the ones ranked before.
            4. Return your answer in a JSON format including the key 'movies' and, for each movie, the key 'name' with the name of the movie, and the key 'score_diversity', 'score_novelty' and 'score' with an integer value for your diversity, novelty and overall score. Add the key 'final_score' with the score computed for the whole list.
        
        Do not include anything else, except the name of ALL movies and your numeric score, in your answer. Do not change the movie order.
            

        All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences" and the "recommended movies".
        Do not include anything else, except the name of ALL movies and your numeric score, in your answer. 

        Your JSON answer:
        """
       
    def __str__(self):
        return 'LLMrewardDNR#' + self.llm_name
        

class LLMRewardRerank(LLMReward): # the same as before, but states that the order is important (4).

    SYSTEM_PROMPT = """You are a helpful movie recommender system. Your task is to assess movies for a group of people based on their waching history.
        You will receive:
            * The group preferences.
            * The individual user preferences.
            * The list of movies recommended to each membem of the group.
            * A list of movies to recommend to the group.

        Your task is to use all the provided information to generate a score for the movies to recommend to the group. You have access to all the information you need.
        """
    

    def __init__(self, llm, store, group_id=None, llm_name=None,what_to_evaluate='diversity'):
        super().__init__(llm,store,group_id,llm_name)
        self.what_to_evaluate = what_to_evaluate
        self.PROMPT = """
        {intersection}

        These are the "individual user preferences":
        {users_history}

        These are the "recommendations for each group member":
        {user_recommendations}

        These are the "group recommended movies": {to_recommend}

        Your task is the following:
            1. Using the "group preferences", the "individual user preferences" and the "recommendations for each group member", generate a numeric score for each of the "group recommended movies" based on how well they satisfy the group as a whole. The score should range between 0 and 100, where 0 means no satisfaction and 100 means full satisfaction. Generate a score for each movie in the list.
            2. The score should reflect the degrees of {what_to_evaluate} of each of the "group recommended movies" with respect to the "group preferences", "individual group preferences" and "recommendations for each group member". The higher the diversity or the novelty, the better. 
            3. The order of the "recommendations for each group member" and "group recommended movies" is important. With the scores computed by movie, compute a score for the whole list, considering the individual scores for each movie, their position in the ranking, and the relationship between a movie and the ones ranked before.
            4. Return your answer in a JSON format including the key 'movies' and, for each movie, the key 'name' with the name of the movie, and the key 'score_diversity', 'score_novelty' and 'score' with an integer value for your diversity, novelty and overall score. Add the key 'final_score' with the score computed for the whole list.
        
        Do not include anything else, except the name of ALL movies and your numeric score, in your answer. Do not change the movie order.
            

        All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences", "recommendations for each group member" and the "group recommended movies".
        Do not include anything else, except the name of ALL movies and your numeric score, in your answer. 

        Your JSON answer:
        """
           
    def get_prompt_params(self, recommendations, group_id):
        
        rr = self._format_recommendations(recommendations)
        print(rr)
        
        intersection = self.store.get_common_rated_movies(group_id)
        params = {
            "intersection": self._format_intersection(intersection),
            "users_history": self._format_users_history(group_id,self.store.get_personal_rated_movies),
            "to_recommend": rr,
            "what_to_evaluate": self.what_to_evaluate,
            "user_recommendations": self._format_users_history(group_id,self.store.get_base_recommendations), 
        }
        
        return params

    def __str__(self):
        nn = 'D#' if self.what_to_evaluate == 'diversity' else 'N#' if self.what_to_evaluate == 'novelty' else 'F#' if self.what_to_evaluate == 'fairness' else 'DNF#'
        return 'LLMRewardRerank#' + nn + self.llm_name
        
        
class LLMRewardPos(LLMReward): # the same as before, but states that the order is important (4).

    SYSTEM_PROMPT = """You are a helpful movie recommender system. Your task is to assess movies for a group of people based on their waching history.
        You will receive:
            * The group preferences.
            * The individual user preferences.
            * The position in which each movie was recommended to each user. A -1 indicates that the movie was not recommended to the user.
            * A list of movies to recommend to the group.

        Your task is to use all the provided information to generate a score for the movies to recommend to the group. You have access to all the information you need.
        """
    

    def __init__(self, llm, store, group_id=None, llm_name=None,what_to_evaluate='diversity'):
        super().__init__(llm,store,group_id,llm_name)
        self.what_to_evaluate = what_to_evaluate
        self.PROMPT = """
        {intersection}

        These are the "individual user preferences":
        {users_history}

        These are the "positions in which each movie was recommended to the user":
        {user_recommendations}

        These are the "group recommended movies": {to_recommend}

        Your task is the following:
            1. Using the "group preferences", the "individual user preferences" and the "positions in which each movie was recommended to the user", generate a numeric score for each of the "group recommended movies" based on how well they satisfy the group as a whole. The score should range between 0 and 100, where 0 means no satisfaction and 100 means full satisfaction. Generate a score for each movie in the list.
            2. The score should reflect the degrees of {what_to_evaluate} of each of the "group recommended movies" with respect to the "group preferences", "individual group preferences" and "positions in which each movie was recommended to the user". The higher the diversity or the novelty, the better. 
            3. The order of the "group recommended movies" and in which positions they were recommended to the users is important. With the scores computed by movie, compute a score for the whole list, considering the individual scores for each movie, their position in the ranking, and the relationship between a movie and the ones ranked before.
            4. Return your answer in a JSON format including the key 'movies' and, for each movie, the key 'name' with the name of the movie, and the key 'score_diversity', 'score_novelty' and 'score' with an integer value for your diversity, novelty and overall score. Add the key 'final_score' with the score computed for the whole list.
        
        Do not include anything else, except the name of ALL movies and your numeric score, in your answer. Do not change the movie order.
            

        All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences", "positions in which each movie was recommended to the user" and the "group recommended movies".
        Do not include anything else, except the name of ALL movies and your numeric score, in your answer. 

        Your JSON answer:
        """
           
    def get_prompt_params(self, recommendations, group_id):
        
        rr = self._format_recommendations(recommendations)
        print(rr)
        
        intersection = self.store.get_common_rated_movies(group_id)
        
        recs = self.store.get_base_recommendations(group_id)
        ll = []
        for u in recs:
            ll.append(f"- User {str(u)}: [" + ','.join([str(recs[u].index(x)) if x in recs[u] else '-1' for x in recommendations]) + ']') 
        
        params = {
            "intersection": self._format_intersection(intersection),
            "users_history": self._format_users_history(group_id,self.store.get_personal_rated_movies),
            "to_recommend": rr,
            "what_to_evaluate": self.what_to_evaluate,
            "user_recommendations": "\n"+ '\n'.join(ll)
        }
               
        return params

    def __str__(self):
        nn = 'D#' if self.what_to_evaluate == 'diversity' else 'N#' if self.what_to_evaluate == 'novelty' else 'F#' if self.what_to_evaluate == 'fairness' else 'DNF#'
        return 'LLMRewardPos#' + nn + self.llm_name
    
class FairnessReward(RewardStrategy): # How many users were recommeded these items? # users * weight of position
    
    def __init__(self, store):
        self.store = store
    
    def compute_score(self, recommendations, group_id=None): 

        base = self.store.get_base_recommendations(group_id)
        cants = 0
        for i in range(0,len(recommendations)):
            cants += ((len(recommendations) - i)/len(recommendations)) * sum(1 for v in base.values() if recommendations[i] in v)
        
        return cants   
        
    def __str__(self):
        return 'fairreward'
    

class PositionReward(RewardStrategy): # What was the average position of this item in the rankings
    
    def __init__(self, store):
        self.store = store

        
    def compute_score(self, recommendations, group_id=None):
        cant = 0
        base = self.store.get_base_recommendations(group_id)
        for i in range(0,len(recommendations)): # if the item was not recommended to a user, it is penalized by setting index to len(v) 
            cc = ((sum(v.index(recommendations[i]) if recommendations[i] in v else len(v)+1 for v in base.values())/ len(base)) - i) 
            cant += 1 if cc == 0 else 1/cc
        
        return cant / len(recommendations)
        
    def __str__(self):
        return 'posrewardRR' 
    

class MRRReward(RewardStrategy): # linear combination, weights should add to 1
    
    def __init__(self, store, metrics):
        self.store = store
        self.metrics = metrics # {metric : weight}

    def compute_score(self, recommendations, group_id=None):
        score = sum(k.compute_score(recommendations,group_id)*v for k,v in self.metrics.items())
        return score

        
    def __str__(self):
        return 'MRRreward' # This should include the names of the combinations, but here we will only use one
    

class WeightMetricReward(RewardStrategy): # fairness acts as a weight of the other score
    
    def __init__(self, store, weight_metric, metric):
        self.store = store
        self.weight_metric = weight_metric
        self.metric = metric 

    def compute_score(self, recommendations, group_id=None):
        return self.weight_metric.compute_score(recommendations,group_id) * self.metric.compute_score(recommendations,group_id)

        
    def __str__(self):
        return 'Weightreward#' + self.weight_metric.__str__() + '#' + self.metric.__str__()
    

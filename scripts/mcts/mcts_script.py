
import pandas as pd
import pickle
from tqdm import tqdm

import os
import argparse

from langchain_community.chat_models import ChatOllama

from grsmcts import MoviesStore, GroupMovieRecommenderSystem
from strategies import LLMExpansion, RandomExpansion, MetricReward, LLMReward, LLMRewardDN, LLMRewardDNR, RandomSimulation, LLMSimulation, FairnessReward, MRRReward, PositionReward, WeightMetricReward, LLMRewardRerank, LLMRewardPos
from evaluation import content_distance


def find_file(dir_path,pickle_filename,prefix_base_recs):

	pf = '__'.join(pickle_filename.replace('_ids.pickle','').split('__')[0:-1])
	for f in os.listdir(dir_path):
		if not f.startswith('recommendations_' + prefix_base_recs):
			continue
		if pf in f:
			return f
	return None


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='MCTS')
	parser.add_argument('--model', type=str, default=None, metavar='N',help='which model to use: gemma:2b-instruct, mistral:7b-instruct, gpt-3.5-turbo, llama3')
	
	parser.add_argument('--path_dir', type=str, default='./', metavar='N', help='root path of everything')
	parser.add_argument('--results_dir', type=str, default='./', metavar='N', help='results path')
	
	parser.add_argument('--expansion', type=str, default='random', metavar='N', help='expansion strategy to use: ')
	
	parser.add_argument('--simulation', type=str, default=None, metavar='N', help='simulation strategy to use: ')
	
	parser.add_argument('--reward', type=str, default='metric', metavar='N', help='reward strategy to use: metric, llm')
	parser.add_argument('--metric', type=str, default='diversity', metavar='N', help='metric for the metric reward')
	parser.add_argument('--how_summ', type=str, default='avg', metavar='N', help='how to summarize metric (in case of novelty): avg, min, max, var, median')
	
	parser.add_argument('--num_iterations', type=int, default=200, metavar='N', help='num of iterations')
	parser.add_argument('--random_seed', type=int, default=4, metavar='N', help='random seed')
	parser.add_argument('--how_many', type=int, default=-1, metavar='N', help='random seed')
	
	parser.add_argument('--prefix_base_recs', type=str, default=None, metavar='N', help='base recommendations to select')
	parser.add_argument('--n', type=int, default=None, metavar='N', help='how many base recommendations per user to include in the analysis: None -> to_recommend, -1 -> all, >0 -> only those')
	parser.add_argument('--group', type=int, default=None, metavar='N', help='group building strategy to cosider')

	args = parser.parse_args()

	dir_path = args.path_dir
	dir_results = args.results_dir

	ratings_df = pd.read_csv(dir_path + 'ml-25m/ratings.csv')
	ratings_df['userId'] = ratings_df['userId'].astype(int)
	ratings_df['movieId'] = ratings_df['movieId'].astype(int)

	movies_df = pd.read_csv(dir_path + 'ml-25m/movies.csv')
	movies_df['movieId'] = movies_df['movieId'].astype(int)

	genres = content_distance(items_path=dir_path + 'ml-25m/movies_genres.pickle')

	# Create the store
	store = MoviesStore(name="dataset", ratings=ratings_df, movies=movies_df)
	
	max_rollouts = 2 #
	max_expansions = 3 #
	rs = args.random_seed
	
	if args.reward.startswith('llm') or args.expansion == 'llm' or args.simulation == 'llm':
		if args.model is None:
			print('------------ Error undefined LLM model!')
			exit()
		llm = ChatOllama(model=args.model, temperature=0.0, format="json")

	if args.reward == 'metric':
		reward_strategy = MetricReward(store, movies_df, cdd=genres,metric=args.metric, how=args.how_summ)
	elif args.reward == 'llm':
		reward_strategy = LLMReward(llm, store, llm_name=args.model)
	elif args.reward == 'llmDN':
		reward_strategy = LLMRewardDN(llm, store, llm_name=args.model)
	elif args.reward == 'llmDNR':
		reward_strategy = LLMRewardDNR(llm, store, llm_name=args.model)
	elif args.reward == 'llmRR':
		reward_strategy = LLMRewardRerank(llm,store,llm_name=args.model,what_to_evaluate=args.metric)
	elif args.reward == 'llmPos':
		reward_strategy = LLMRewardPos(llm,store,llm_name=args.model,what_to_evaluate=args.metric)
	elif args.reward == 'mrr':
		reward_strategy = MRRReward(store,metrics={MetricReward(store, movies_df, cdd=genres,metric='diversity', how=args.how_summ):0.5,
												   MetricReward(store, movies_df, cdd=genres,metric='novelty', how=args.how_summ):0.5})
	elif args.reward == 'fair':
		reward_strategy = FairnessReward(store)
	elif args.reward == 'pos':
		reward_strategy = PositionReward(store)
	elif args.reward == 'weight':
		reward_strategy = WeightMetricReward(store,weight_metric=FairnessReward(store),metric=MetricReward(store, movies_df, cdd=genres,metric='novelty', how=args.how_summ))
		
	if args.expansion == 'random':
		expansion_strategy = RandomExpansion()
	elif args.expansion == 'llm':
		expansion_strategy = LLMExpansion(llm, store,llm_name=args.model)

	if args.simulation is None:
		simulation_strategy = None
		max_rollouts = 0
		max_expansions = 5
	elif args.simulation == 'random':
		simulation_strategy = RandomSimulation(max_rollouts)
	elif args.simulation == 'llm':
		simulation_strategy = LLMSimulation(llm, store, args.model,max_rollouts)

	for pickle_filename in os.listdir(dir_path):
		
		if not pickle_filename.endswith('_ids.pickle'):
			continue
		
		if not pickle_filename.startswith('groups_min30'):
			continue

		if '1000g' not in pickle_filename:
			continue
		
		if args.group is not None and args.group not in pickle_filename:
			continue

		print(pickle_filename)
		
		if simulation_strategy is None:
			results_path = 'results__' + reward_strategy.__str__() + '-'+ expansion_strategy.__str__() + '__' + pickle_filename.split('__gs_')[0] + '.pickle'
		else:
			results_path = 'results__' + simulation_strategy.__str__() + '-' + reward_strategy.__str__() + '-'+ expansion_strategy.__str__() + '__' + pickle_filename.split('__gs_')[0] + '.pickle'
		
		if args.num_iterations != 200:
			results_path = results_path.replace('.pickle','_it' + str(args.num_iterations) + '.pickle')
			
		if rs != 4:
			results_path = results_path.replace('.pickle','_rs' + str(rs) + '.pickle')
		
		if args.prefix_base_recs is not None:
			num = '' if args.n is None else 'all' if args.n < 0 else str(args.n)
			results_path = results_path.replace('.pickle','_' + args.prefix_base_recs + '#' + num + '.pickle') 


		print(results_path)
		if os.path.exists(dir_results + results_path):
			results = pd.read_pickle(dir_results+ results_path)
		else:
			results = []

		if args.prefix_base_recs is None:
			store.load_profiles(dir_path + pickle_filename) 
		else:
			recs_base_filename = find_file(dir_path,pickle_filename,args.prefix_base_recs)
			if recs_base_filename is None:
				print('---------- Error recommendation file not found!!')
				exit()
			store.load_profiles(dir_path + pickle_filename,dir_path + recs_base_filename,args.n)

		if args.how_many != -1:
			how_many = min(args.how_many,len(store.get_groups())) 
		else:
			how_many = len(store.get_groups())
		
		all_movies = set()
		
		for group_id in tqdm(range(len(results),how_many), "Groups"):

			grs = GroupMovieRecommenderSystem(group_id, store,max_rollouts=max_rollouts, max_expansions=max_expansions) 
			grs.set_reward_strategy(reward_strategy)
			expansion_strategy.set_group(group_id)
			grs.set_expansion_strategy(expansion_strategy)
			if simulation_strategy is not None:
				simulation_strategy.set_group(group_id)
				grs.set_simulation_strategy(simulation_strategy)
			
			recommendations = grs.recommend_movies(n_iterations=args.num_iterations, top_k=5, random_seed=rs)
			print("Recommendations: ", recommendations)

			results.append(recommendations) 
				
			with open(dir_results + results_path,'wb') as file: 
				pickle.dump(results,file)   
		

print('All done!')
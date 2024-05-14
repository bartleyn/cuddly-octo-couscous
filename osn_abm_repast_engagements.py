import networkx as nx
from typing import Dict
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from abm_utils import *

from repast4py.network import write_network, read_network
from repast4py import context as ctx
from repast4py import core, random, schedule, logging, parameters
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.decomposition import NMF

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model as tfmodel
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_focal_crossentropy

import os
import sys
import math
import pandas as pd
import sklearn.preprocessing
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_RATING_COL as RATING_COL,
    DEFAULT_PREDICTION_COL as PREDICT_COL,
    DEFAULT_GENRE_COL as ITEM_FEAT_COL,
    SEED
)
from recommenders.utils import tf_utils, gpu_utils

import recommenders.evaluation.python_evaluation as evaluator
import recommenders.models.wide_deep.wide_deep_utils as wide_deep

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")


from collections import deque


import time




global map_user_id 
map_user_id = {}

global user_map
user_map = {}

#load full graph so that we can get the in and out degrees and labels
global full_graph

full_graph = nx.read_edgelist('twittergraph.edgelist_1_5m.txt', create_using=nx.DiGraph(), nodetype=int, data=(('seen', float),))






model = None



def predict_wide_deep(self):
   


    #data is a pandas dataframe with columns user_id, tweet_id, and like from each agents' liked tweets
    data = pd.DataFrame([{'user_id':self.su_uids.index(agent.id), 'tweet_id':self.map_user_id[tweet], 'like':1} for agent in self.context.agents() \
                                                                            for tweet in agent.liked_tweets])
    
    
    if len(data) == 0:
        print("No data for NCF in rank ", self.rank)
        data['user_id'] = []
        data['tweet_id'] = []
        data['like'] = []

    num_users = 5600
    num_tweets = self.total_nodes + 1
    # Creating a LIL sparse matrix
    data_matrix = lil_matrix((num_users, num_tweets), dtype=np.int8)

    # Populate the matrix with likes
    for _, row in data.iterrows():
        user_id = row['user_id']
        tweet_id = row['tweet_id']
        like = row['like']
        data_matrix[user_id, tweet_id] = like

    data_matrix = data_matrix.tocoo()
    
    users = data_matrix.row
    items = data_matrix.col
    ratings = data_matrix.data

    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    tweet_input = Input(shape=(1,), name='tweet_input')

    # Wide part uses linear relationships, hence simple embeddings
    wide_user_embedding = Embedding(num_users, 1, input_length=1, name='wide_user_embedding')(user_input)
    wide_tweet_embedding = Embedding(num_tweets, 1, input_length=1, name='wide_tweet_embedding')(tweet_input)
    wide = Concatenate()([Flatten()(wide_user_embedding), Flatten()(wide_tweet_embedding)])

    # Deep part with more complex interactions, larger embeddings
    deep_user_embedding = Embedding(num_users, 15, input_length=1, name='deep_user_embedding')(user_input)
    deep_tweet_embedding = Embedding(num_tweets, 15, input_length=1, name='deep_tweet_embedding')(tweet_input)
    deep = Concatenate()([Flatten()(deep_user_embedding), Flatten()(deep_tweet_embedding)])

    deep = Dense(128, activation='relu')(deep)
    deep = Dense(64, activation='relu')(deep)

    # Combine wide and deep parts
    combined = Concatenate()([wide, deep])
    output = Dense(1, activation='sigmoid')(combined)  # Assuming binary classification, use 'linear' for regression

    model = tfmodel(inputs=[user_input, tweet_input], outputs=output)

    model.compile(optimizer=Adam(0.001), loss=binary_focal_crossentropy, metrics=['accuracy'])
    if len(data) > 0:
        model.fit([users, items], ratings, epochs=10, batch_size=64, verbose=0)
    else:
        print("No data for NCF in rank ", self.rank)






    return model

class ExposureAgent(core.Agent):


    def __init__(self, nid: int, agent_type: int, rank: int, wealth=1, len_feed=30, last_nodes_seen=None, \
                  mdl=None, real_userid=None, activity="Constant",   lam=3, ranking=None, user_label=None):
            # Pass the parameters to the parent class.
            super().__init__(int(nid), agent_type, rank)

            # Create the agent's attribute and set the initial values.
            self.wealth = 1
            self.active = False
            self.lam = lam
            self.ranking = ranking
            self.activity = activity
            self.len_feed = len_feed
            self.last_nodes_seen = set() if last_nodes_seen is None else last_nodes_seen
            self.activity = activity
            self.user_label = user_label

            self.real_userid = real_userid
            self.liked_tweets = dict()        

            self.random = np.random.default_rng()

            if activity == "Poisson":
                self.mdl = self.random.poisson
            #if activity == 'Empirical':
            #    self.real_userid = real_userid
            #    #self.activity_map = {date: num_tweets}
            if activity == 'HeavyTail':
                self.mdl = self.random.lognormal#(mean=0.218)


    def _set_friends(self, friends):
        self.friends = friends
    
    def _set_len_feed(self, len_feed):
        self.len_feed = len_feed

    def _set_activity(self, activity):
        self.activity = activity
        if activity == "Poisson":
            self.mdl = self.random.poisson
        if activity == 'HeavyTail':
            self.mdl = self.random.lognormal#(mean=0.218)

    def _set_real_userid(self, uid):
        self.real_userid = int(uid[0])


    def _set_ranking(self, ranking):
        self.ranking = ranking


    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data (self.received_rumor).

        Returns:
            The agent's state
        """
        return (self.uid, self.wealth, self.len_feed,\
                 self.last_nodes_seen, self.mdl if self.activity == 'Poisson' or self.activity == 'HeavyTail' else None,\
                 self.real_userid if self.activity == 'Empirical' else None, self.activity, self.user_label)


    def update(self, wealth, len_feed, last_nodes_seen, mdl, real_userid, activity, user_label):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state (received_rumor)
        """
        if wealth > 0.0:
            self.wealth += wealth
        self.len_feed = len_feed
        self.last_nodes_seen = self.last_nodes_seen.union(last_nodes_seen)
        if activity == 'Poisson':
            self.mdl = mdl
        if activity == 'HeavyTail':
            self.mdl = mdl
        if activity == 'Empirical':
            real_userid = int(real_userid)
        self.user_label = user_label
        self.activity = activity
        

    def step(self):
        '''
        check to see if the user will be active at this timestep
        if so, then 
            add a number of tweets to the content pool
            pick a number of tweets that the user consumes from their timeline
            get the tweets that the user sees
            interact with each tweet with a certain probability

        
        if not, then do nothing

        '''
        if not self.active:
            if self.random.uniform() < 0.083:
                self.active = True
            else:
                return
        
        # pick a number of tweets that the user consumes from their timeline
        num_tweets_consumed = self.len_feed
        if self.activity == "Constant":
            num_tweets_produced = 10
        elif self.activity == 'Empirical':
            num_tweets_produced = self.activity_map[self.runner.tick()]
        elif self.activity == 'Poisson':
            num_tweets_produced = self.mdl(self.lam, size=1)[0]
        else:
            num_tweets_produced = int(self.mdl(size=1)[0])
        
        cur_time = model.runner.tick()
        model.content_pool.extend([[int(self.id), int(cur_time), "{}_{}_{}".format(self.id, cur_time, x), float(user_map[map_user_id[self.id]]) if map_user_id[self.id] in user_map else -1 ] for x in range(num_tweets_produced)]) #4807204

        if self.ranking == None:
            return
        
        

        t0 = time.time()
        # get the tweets that the user sees
        tweets_seen = model.serve_tweets(num_tweets_consumed, self.id, ranking=self.ranking) 
        num_friends = len(self.friends)

        t1 = time.time()
        if t1 - t0 > 0.9:

            print("time to serve tweets ", t1 - t0, " for agent ", self.id)
        

        try:
            map_user_id[self.id]
        except KeyError as e:
            raise KeyError("user {} not in map_user_id {}".format(self.id, map_user_id))
        try:
            
            cur_val = user_map[map_user_id[self.id]]
        except KeyError as e:
            print(e)
            print("map_user_id[self.id] not in user_map id {} map{} ".format(self.id, map_user_id))
            cur_val = model.random.binomial(n=1, p=model.true_prev, size=1)[0]
            user_map[map_user_id[self.id]] = cur_val
        #tweet[2] is each user tweet[0]'s value

        likes_20p = self.random.binomial(n=1, p=0.2, size=len(tweets_seen))


        # interact with each tweet with a certain probability

        likes = self.random.binomial(n=1, p=0.05, size=len(tweets_seen))

        if cur_val == 1:
            tweet_vals = np.array([float(x[3]) for x in tweets_seen])
            inv_tweet_vals = 1.0 - tweet_vals 

            likes_20p = tweet_vals * likes_20p
            likes = inv_tweet_vals * likes
        else:
            tweet_vals = np.array([float(x[3]) for x in tweets_seen])
            
    
            inv_tweet_vals = 1.0 - tweet_vals

            likes_20p = inv_tweet_vals * likes_20p
            likes = tweet_vals * likes
        
        likes = likes + likes_20p

        

        user_likes = list(zip([x for x in tweets_seen], likes))


        if self.real_userid is None:
            return
        
        model.update_likes(self.real_userid, [model.map_user_id[int(x[0][0])] for x in user_likes if int(x[1]) == 1])

        for ix, k in enumerate([10,20,30]):
            model.precision_tracker[model.su_uids.index(self.real_userid),ix] = np.sum([x[1] for x in user_likes[:k]]) / k

        for t in [x[0] for x in user_likes if x[1] == 1]:
            try:
                self.liked_tweets[int(t[0])].append(int(t[1]))
            except KeyError as e:
                self.liked_tweets[int(t[0])] = [int(t[1])]
        

        agent_dict = {agent.id: agent for agent in model.context.agents()}

        # Lookup current agent once, outside the loop
        cur_agent = agent_dict[self.id]

        # Initialize data structures if not already done
        if not hasattr(model, 'user_map_initialized'):
            model.user_map = {}
            model.user_map_initialized = True

        # Process tweets
        t0 = time.time()
        for tweet in tweets_seen:
            tweet_agent_id = int(tweet[0])
            if tweet_agent_id in agent_dict:
                agent = agent_dict[tweet_agent_id]
                user_id = model.map_user_id.get(agent.id)

                # Handle user map
                if user_id is not None:
                    user_value = model.user_map.get(user_id)
                    if user_value is None:
                        model.user_map[user_id] = self.random.binomial(n=1, p=model.true_prev, size=1)[0]
                
                # Check if agent is a friend and update graph if necessary
                if agent in model.map_friends[cur_agent]:
                    model.net.graph[agent][cur_agent]['seen'] = 1.0
                
                # Add to last nodes seen
                self.last_nodes_seen.add((tweet_agent_id, self.id, float(model.out_degree[cur_agent.id]) + 1, float(tweet[3]), float(model.out_degree.get(agent.id, 0)), float(model.map_deg_friends.get(agent, 0))))

        t1 = time.time()
        if t1 - t0 > 0.9:
            print(f"Time to process tweets_seen: {t1 - t0} seconds for {cur_agent}")



def create_exposure_agent(nid, agent_type, rank, **kwargs):
    return ExposureAgent(nid, agent_type, rank, **kwargs)



def restore_agent(agent_data):
    uid_ = agent_data[0]
    return ExposureAgent(uid_[0], uid_[1], uid_[2], *agent_data[1:])


@dataclass
class RumorCounts:
    total_rumor_spreaders: int
    new_rumor_spreaders: int

@dataclass
class ExposureMeasures:
    local_bias: float
    gini: float
    std_likes: float
    num_edges_seen: float
    mean_likes: float
    mean_tweets_per_user: float
    std_tweets_per_user: float
    curr_corr: float
    precision_at_10: float
    precision_at_30: float



class Model:    

    def __init__(self, comm, params):
        print("Initializing model")
        self.runner = schedule.init_schedule_runner(comm)

        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['network_file']
        self.context = ctx.SharedContext(comm)
        print("Reading network")
        read_network(fpath, self.context, create_exposure_agent, restore_agent)
        self.net = self.context.get_projection('twitter_graph')

        self.len_feed = params['len_feed']
        self.random = np.random.default_rng()
        self.max_iters = params['stop.at']
        self.activity = params['activity']
        self.network = params['network']

        if self.network == 'Empirical':
            seed_users_tmp = pd.read_csv(params['seed_users'], header=0)['user_id'].astype(int).values
            self.seed_users = []
            num_su = 5599
            with open(params['network_file'], 'r') as fp:
                for line in fp:
                    if line.startswith('twitter_graph'):
                        continue
                    elif line.startswith('EDGES'):
                        print("quit out of loop from EDGES")
                        break
                    else:
                        u = int(line.split(' ')[0])
                        if u in seed_users_tmp:
                            self.seed_users.append(tuple([int(x) for x in line.split(' ')]))
                            num_su -= 1
                    if num_su == 0:
                        break


        self.rank = comm.Get_rank()

        print("Num nodes: ", len(self.net.graph.nodes))
        print("Num seedusers ", len(self.seed_users))
        for prj in self.context.projections.values():
            print("Projection name: ", prj.name)
        
        friends_to_add = []
        self.su_uids = [x[0] for x in self.seed_users]
        for agent in self.context.agents():
            agent._set_real_userid(agent.uid)
            if self.network == 'Empirical' and agent.id in self.su_uids:
                friends_to_add.extend(list(self.net.graph.predecessors(agent)))
        for fr in friends_to_add:
            self.context.add(fr)
        
        
        self.tweets_seen_dict = {x.id:set([]) for x in self.context.agents()}

        ctr = 0
        num_in_if = 0

        for agent in self.context.agents():
            try:
                if self.network == 'Empirical' and agent.id in self.su_uids:
                    agent._set_ranking(params['ranking'])
                    agent._set_friends(list(self.net.graph.predecessors(agent)))
                    agent._set_len_feed(self.len_feed)
                    #agent._set_user_val()
                    num_in_if += 1
                else:
                    agent._set_ranking(None)
                agent._set_activity(params['activity'])
                ctr += 1
            except nx.exception.NetworkXError as e:
                print(e)
                print(agent.id)
                print(self.net.graph.nodes)
                continue

        print("Num agents: ", ctr, " with ", num_in_if, " in if")
        if params['network'] != 'Empirical':
            self.net.graph = adjust_assort(self.net.graph, params['assort'])
        self.out_degree = dict(full_graph.out_degree())
        self.follower_dist = {x:self.out_degree[x] if x in self.out_degree else 0.0 for x in full_graph.nodes()}
        self.in_degree = dict(full_graph.in_degree())
        
        self.total_nodes = len(full_graph.nodes)

        self.friend_dist = [self.in_degree[x] if x in self.in_degree else 0.0 for x in list(full_graph.nodes())]
        #self.mean_in_deg = np.mean(self.friend_dist)
        self.mean_in_deg = np.mean([x for x in list(self.in_degree.values())])
        self.sum_out_degree_obs = 0.0
        print("mean in degree: {}\t mean out_degree: {}".format(self.mean_in_deg, np.mean([x for x in dict(full_graph.out_degree()).values()])))
        self.content_pool = []
        self.edges_seen = set()
        self.num_edges_seen = 0
        self.true_prev = params['true_prev']
        self.likes_tracker = lil_matrix((5599, len(full_graph.nodes)), dtype=np.uint16)
        self.precision_tracker = lil_matrix((5599, 3))


        if params['network'] != 'Empirical':
            self.rewire_synth(params['kx_corr'])
        self.kx_corr = params['kx_corr']

        
        self.nodes = list(self.net.graph.nodes)
        self.ranking = params['ranking']
        self.strategy = params['strategy']
        self.epsilon = params['epsilon']
        self.switch_ranking = None if  params['switch_ranking'] == 'None' else params['switch_ranking']
        #map_user_id.update({user.id: self.nodes.index(user) for user in self.nodes})
        global user_map
        self.user_map = user_map
        global map_user_id 

        self.map_user_id = map_user_id
        self.map_friends = {user: [fr for fr in self.net.graph.predecessors(user)] for user in self.net.graph.nodes}
        self.map_deg_friends = {user: sum([self.follower_dist[fr] if fr in self.follower_dist else 0.0 for fr in self.map_friends[user]]) for user in self.map_friends}

    


        
        self.exp_measures = ExposureMeasures(-1.,-1., -1., -1, -1, -1, -1, -1, -1, -1)

        loggers = logging.create_loggers(self.exp_measures, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['counts_file'])
        #self.data_set.log(0)

        self.true_prev = params['true_prev']
        self.curr_corr = 0.0
   


    def update_likes(self, real_userid, likes):
        for like in likes:
            self.likes_tracker[self.su_uids.index(real_userid), like] += 1

        #self.likes_tracker[self.su_uids.index(real_userid), [x[0] for x in likes]] += 1

    def at_end(self):
        self.data_set.close()

    def cache_edge_attributes(self):
        self.edge_cache = {}
        for edge in self.net.graph.edges(data=True):
            uid = (edge[0], edge[1])
            self.edge_cache[uid] = {
                'seen': edge[2].get('seen', 0),
                'out_degree': self.out_degree[int(edge[0].id)]
            }

    def step(self):
        print("Rank: ", self.rank, "staring Tick: ", self.runner.schedule.tick)
        self.cache_edge_attributes()  # Update cache at the beginning of the step
        total_friends = [(fr.uid, fr.uid[2]) for x in self.context.agents() for fr in self.map_friends[x]] 

        self.context.request_agents(total_friends, restore_agent)

        if self.ranking == 'NMF':
            nmf_model = NMF(n_components=4, init='random', random_state=0)
            try:
                self.W = lil_matrix(nmf_model.fit_transform(self.likes_tracker))
                self.H = lil_matrix(nmf_model.components_)
                self.rec = self.W.dot(self.H)

            except ValueError as e:
                print(e)
                print(np.mean(self.likes_tracker.data), np.min(self.likes_tracker.data))
                print("continuing with random vecs")
                self.rec = self.random.uniform(size=(self.likes_tracker.shape[0], self.likes_tracker.shape[1]))
            

        
        cur_tick = self.runner.schedule.tick
        if self.ranking == 'WideDeep':
            if cur_tick == 1:
                self.rec = None#self.random.uniform(size=(self.total_nodes, self.total_nodes))
            else:
                print ("BEFORE THE FIRST OF THE THINGS")
                self.rec = predict_wide_deep(self).predict
                print ("AFTER THE FIRST")
        
        if self.ranking == 'NCF':
                

            #data is a pandas dataframe with columns user_id, tweet_id, and like from each agents' liked tweets
            data = pd.DataFrame([{'user_id':self.su_uids.index(agent.id), 'tweet_id':self.map_user_id[tweet], 'like':1} for agent in self.context.agents() \
                                                                                    for tweet in agent.liked_tweets])
            
            
            if len(data) == 0:
                print("No data for NCF in rank ", self.rank)
                data['user_id'] = []
                data['tweet_id'] = []
                data['like'] = []

            num_users = 5600
            num_tweets = self.total_nodes + 1
            # Creating a LIL sparse matrix
            data_matrix = lil_matrix((num_users, num_tweets), dtype=np.int8)

            # Populate the matrix with likes
            for _, row in data.iterrows():
                user_id = row['user_id']
                tweet_id = row['tweet_id']
                like = row['like']
                data_matrix[user_id, tweet_id] = like

            data_matrix = data_matrix.tocoo()
            
            users = data_matrix.row
            items = data_matrix.col
            ratings = data_matrix.data

           
            # Neural Collaborative Filtering (NCF) model

            # User Embedding Path
            input_users = Input(shape=(1,))
            embedding_users = Embedding(num_users, 15)(input_users)
            flatten_users = Flatten()(embedding_users)

            # Item Embedding Path
            input_items = Input(shape=(1,))
            embedding_items = Embedding(num_tweets, 15)(input_items)
            flatten_items = Flatten()(embedding_items)

            # Concatenate the flattened embeddings
            concatenated = Concatenate()([flatten_users, flatten_items])

            # Add one or more Dense layers for learning the interaction
            dense = Dense(128, activation='relu')(concatenated)
            output = Dense(1)(dense)

            if self.runner.schedule.tick >= 0:
                # Create and compile the model
                ncf_model = tfmodel(inputs=[input_users, input_items], outputs=output)
                ncf_model.compile(loss=binary_focal_crossentropy, optimizer=Adam(0.001))

                # Train the model
                if len(data) > 0:
                    ncf_model.fit([users, items], ratings, epochs=10, batch_size=64, verbose=0)
                else:
                    print("No data for NCF in rank ", self.rank)
            self.rec = ncf_model.predict

        
        for node in self.context.agents():
            node.step()

        
        '''
        if context is rank 0 
            then request all the seeduser agents
            update the edges seen
            compute local bias and gini
            log the data

        '''
        t0 = time.time()
        self.context.request_agents([(x, x[2]) for x in self.seed_users], restore_agent)

        self.context.synchronize(restore_agent)
        t1 = time.time()
        print("Rank: ", self.rank, "Tick: ", self.runner.schedule.tick, "done with step and sync stuff in ", t1 - t0)
        if self.ranking == 'MinimizeRho' and self.rank != 0:

            t0 = time.time()
            obs_edges = [uid for uid, attrs in self.edge_cache.items() if attrs['seen'] == 1.0]
            self.sum_out_degree_obs = sum(attrs['out_degree'] for uid, attrs in self.edge_cache.items() if uid in obs_edges)
            self.num_edges_seen = len(obs_edges)
            t1 = time.time()
            print("time to update all variables ", t1 - t0, " for model rank " , self.rank)
        agents = list(self.context.agents())
        if self.rank == 0:
            to_be_added = set()
            for agent in agents:  # Assuming a list of agents is processed here
                to_be_added.update(agent.last_nodes_seen)

            
            self.edges_seen.update(to_be_added)

            #agent.last_nodes_seen = set()

            self.num_edges_seen = len(self.edges_seen) if self.edges_seen is not None else 0
            self.exp_measures.local_bias = local_bias(self)
            self.exp_measures.gini = compute_gini(self)
            self.exp_measures.std_likes = compute_std_likes(self)
            self.exp_measures.num_edges_seen = self.num_edges_seen
            self.exp_measures.mean_likes = compute_mean_likes(self)
            self.exp_measures.mean_tweets_per_user = compute_mean_tweets_per_user(self)
            self.exp_measures.std_tweets_per_user = compute_std_tweets_per_user(self)
            self.exp_measures.precision_at_10 = compute_precision_at_k(self, 10)
            self.exp_measures.precision_at_30 = compute_precision_at_k(self, 30)
            cur_user_map = user_map
            list_user_map = list(user_map.keys())
            self.sum_out_degree_obs = np.mean([x[2] for x in self.edges_seen])
            
            
            delta = 100000

            #rev_user_map = {0:{x:0 for x in cur_user_map if cur_user_map[x] == 0}, 1:{x:0 for x in cur_user_map if cur_user_map[x] == 1}}
            pos_neg_bf = np.array([cur_user_map[x] for x in cur_user_map])
            posn_mapping = {user:ix for ix, user in enumerate(list_user_map)}
            rev_mapping = {ix: user for ix, user in enumerate(list_user_map)}

            positive_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]]#{x:0 for x in rev_user_map[1] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 1])
            negative_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf == 0)[0]]#{x:0 for x in rev_user_map[0] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 0])
            #cur_corr = corr(degree_dist, [*cur_user_map.values()], [*positive_nodes])
            cur_corr = self.corr(pos_neg_bf, positive_nodes)
            self.exp_measures.curr_corr = cur_corr
            

            
        self.data_set.log(self.runner.schedule.tick)
        self.data_set.write()
        num_tweets = len(self.content_pool)
        print("Rank: ", self.rank, " Tick: ", self.runner.schedule.tick, " with ", len(self.edges_seen), " edges seen ", " with ", num_tweets, " tweets in the content pool")


        for agent in agents:
            agent.last_nodes_seen = set()

        if self.kx_corr > 0:
            if self.runner.schedule.tick == 12:
                self.user_map = self.rewire_synth(self.kx_corr * 0.5)
            if self.runner.schedule.tick == 24:
                self.user_map = self.rewire_synth(self.kx_corr)

        if self.runner.schedule.tick % 24 == 0:
            if self.activity != 'Empirical' and int(num_tweets) > 0:
                self.content_pool = self.content_pool[-int(num_tweets)//2:]

            self.tweets_seen_dict = {x.id:set([]) for x in agents}
            self.edges_seen = set()
            nx.set_edge_attributes(self.net.graph, 0.0, 'seen')
            for agent in agents:
                agent.last_nodes_seen = set()

            

        

    def serve_tweets(self, num_tweets, user, ranking=None):
        
        try:
            preds = self.net.graph.predecessors
            for agent_it in self.context.agents():
                if agent_it.id == user:
                    agent = agent_it
                    break
            #agent = [agent for agent in self.context.agents() if agent.id == user][0]
            neighs = [x.id for x in preds(agent)]
            fof = set([x.id for fr in preds(agent) for x in preds(fr) ] + neighs)
        except nx.exception.NetworkXError as e:
            print(e)
            print(user, self.net.graph.nodes)
            return
        
        cur_tick = self.runner.schedule.tick
        #print("Rank: ", self.rank, "Tick: ", cur_tick, "User: ", user, "switch_ranking: ", self.switch_ranking, "Ranking: ", ranking)

        # Use deque to collect only the last 1000 items meeting the criteria
        tweets_seen_deque = deque(maxlen=1000)

        len_ctr = 0
        for tweet in reversed(self.content_pool):
            tweet_user_id = int(tweet[0])
            tweet_id = tweet[2]
            if tweet_user_id in fof and tweet_id not in self.tweets_seen_dict[user]:
                tweets_seen_deque.append(tweet)
                len_ctr += 1
                if len_ctr == 1000:
                    break

        # Convert deque back to list if needed
        tweets_seen = list(tweets_seen_deque)
        if ranking is not None:
            if cur_tick > self.max_iters/2:
                if self.switch_ranking is not None:
                    self.ranking = self.switch_ranking
                ranking = self.ranking
            if self.strategy == 'Epsilon':
                if self.random.uniform() < self.epsilon:
                    ranking = "Random"
            if ranking == "Popularity":
                tweets_seen = sorted(tweets_seen, key=lambda x: self.follower_dist[int(x[0])] if x[0] in self.follower_dist else 0.0, reverse=True)
            elif ranking == "Random":
                tweets_seen = self.random.choice(tweets_seen, size=min(len(tweets_seen), num_tweets), replace=False)
            elif ranking == "Wealth":
                wealths = {agent.id:agent.wealth for agent in self.schedule.agents}
                tweets_seen = sorted(tweets_seen, key = lambda x: wealths[x[0]], reverse=True)
            elif ranking == "NMF":
                try:
                    rec_comp = self.rec[self.su_uids.index(user), :].toarray().ravel()
                except AttributeError as e:
                    rec_comp = self.rec[self.su_uids.index(user), :]

                #print("REC COMP SIZE IS ", rec_comp.shape)
                try:
                    tweets_seen = sorted(tweets_seen, key = lambda x: rec_comp[model.map_user_id[int(x[0])]], reverse=True)
                except IndexError as e:
                    print(e)
                    print("user ", user, " not in rec comp with shape ", rec_comp.shape )
                    raise IndexError
            elif ranking == 'Chronological':
                tweets_seen = sorted(tweets_seen, key=lambda x: x[1], reverse=True)
            elif ranking == 'MinimizeRho':
                out_degree_cache_ix = {int(x[0]): ix for ix, x in enumerate(tweets_seen)}
                out_degree_cache = np.array([self.out_degree[int(x[0])] for x in tweets_seen])

                # Cache results to minimize repeated calculations
                active_indices = [out_degree_cache_ix[int(x[0])] for x in tweets_seen if x[3] == 1.0]
                inactive_indices = [out_degree_cache_ix[int(x[0])] for x in tweets_seen if x[3] == 0.0]

                t0 = time.time()

                # Use numpy advanced indexing for efficient computation
                average_active_in_degree = np.mean(out_degree_cache[active_indices])

                # Using numpy for summing to avoid recomputation and unnecessary indexing
                sum_active = np.sum(out_degree_cache[active_indices])
                sum_inactive = np.sum(out_degree_cache[inactive_indices])

                # Precompute lengths since these are used multiple times
                len_active = len(active_indices)
                len_inactive = len(inactive_indices)

                average_in_degree = (self.sum_out_degree_obs + sum_active + sum_inactive) / \
                                    (self.num_edges_seen + len_active + len_inactive)

                t1 = time.time()




                # Current active degrees
                active_degrees = out_degree_cache[active_indices]
                average_active_in_degree = np.mean(active_degrees)
                current_sum = np.sum(active_degrees)
                current_count = len(active_degrees)
                act_sum = current_sum
                act_count = current_count

                # List potential edges and their degrees
                potential_active_edges = [(index, out_degree_cache[index], 1) for index in active_indices]
                potential_inactive_edges = [(index, out_degree_cache[index], 0) for index in inactive_indices]
                total_edges = potential_active_edges + potential_inactive_edges
                # Sort potential edges by their impact on the average
                potential_active_edges.sort(key=lambda x: abs((current_sum + x[1]) / (current_count + 1) - average_active_in_degree))
                potential_inactive_edges.sort(key=lambda x: abs((current_sum + x[1]) / (current_count + 1) - average_active_in_degree))
                total_edges.sort(key=lambda x: abs((current_sum + x[1]) / (current_count + 1) - average_active_in_degree))
                
                # Initialize control variables
                min_iters = 0
                difference = 0.0
                sampled_edges = []#[x[0] for x in potential_active_edges]
                new_average = average_active_in_degree
                prob_diff = 0.0


                # Greedy alternation between adding and removing edges
                #while min_iters < 2000 and (potential_active_edges or potential_inactive_edges) and (abs(difference) < 10 or prob_diff < 0.05):
                while min_iters < 2000 and (total_edges) and abs(difference) < 10:
                    # Alternate between removing an active edge and adding an inactive edge
                    best_edge = None
                    if total_edges:
                        best_edge = total_edges.pop(0)
                        if best_edge[2] == 0:
                            current_sum += best_edge[1]
                            current_count += 1
                        else:
                            current_sum += best_edge[1]
                            current_count += 1
                            act_sum += best_edge[1]
                            act_count += 1
                            if act_count == 0:
                                average_active_in_degree = 0.0
                                continue
                            else:
                                average_active_in_degree = float(act_sum) / act_count
                            #best_edge = None
                    
                    #if min_iters % 2 == 0:
                    #potential_active_edges.sort(key=lambda x: abs((current_sum + x[1]) / (current_count + 1) - average_active_in_degree))
                    #potential_inactive_edges.sort(key=lambda x: abs((current_sum + x[1]) / (current_count + 1) - average_active_in_degree))
                    total_edges.sort(key=lambda x: abs((current_sum + x[1]) / (current_count + 1) - average_active_in_degree))

                    if best_edge is not None:
                        prob_diff = ((act_count + current_sum) / ( current_count)) - self.true_prev
                        sampled_edges.append(best_edge[0])
                    
                    new_average = current_sum / current_count
                    difference = abs(new_average - average_active_in_degree)
                    
                    if min_iters % 100 == 0:
                        print(f"Iteration {min_iters}: Current difference = {difference}")
                    
                    min_iters += 1

                #print("Final average in-degree after adjustment: {:.2f}".format(new_average))
                #print("Target average active in-degree: {:.2f}".format(average_active_in_degree))

                try:
                    if len_active == 0:
                        active_possible_edges_sub = [x for x in tweets_seen if x[3] == 1.0]
                    if len_inactive == 0:
                        inactive_possible_edges_sub = [x for x in tweets_seen if x[3] == 0.0]

                    try:
                        active_possible_edges_sub = [tweets_seen[ix] for ix in sampled_edges]
                    except TypeError as e:
                        print("ts ", tweets_seen, " si ", sampled_edges)
                        raise Exception
                except AttributeError:
                    try:
                        active_possible_edges_sub = np.concatenate([active_possible_edges_sub, inactive_possible_edges_sub])
                    except ValueError as e:
                        print(e)
                        print("ACTIVE P EDGES SUB ", active_possible_edges_sub, " INACTIVE P EDGES SUB ", inactive_possible_edges_sub)
                        raise Exception
                except ValueError as e:
                        print(e)
                        print("ACTIVE P EDGES SUB ", active_possible_edges_sub, " INACTIVE P EDGES SUB ", inactive_possible_edges_sub)
                        raise Exception
                tweets_seen = active_possible_edges_sub
                tweets_seen = self.random.choice(tweets_seen, size=len(tweets_seen), replace=False)

            elif ranking == 'NCF' or ranking == 'WideDeep':
                if cur_tick == 1 or len(tweets_seen) == 0:
                    tweets_seen = self.random.choice(tweets_seen, size=min(len(tweets_seen), num_tweets), replace=False)
                else:
                    probabilities = self.rec([np.array([self.su_uids.index(user) for ix in range(len(tweets_seen)) ]),\
                                        np.array([model.map_user_id[int(x[0])] for x in tweets_seen])]).flatten()
                    #print("probs ", probabilities)
                    tweets_seen = sorted(tweets_seen, key = lambda x: probabilities[tweets_seen.index(x)], reverse=True)

            

            


        tweets_seen = tweets_seen[:num_tweets]
        self.tweets_seen_dict[user].union(set([x[2] for x in tweets_seen]))
        return tweets_seen
    
    def start(self):
        self.runner.execute()

    def corr(self, y, um):
        avg_pos_degree = np.mean([self.in_degree[x] for x in um if x in self.in_degree ])
        avg_degree = np.mean(self.friend_dist)
        std_deg = np.std(self.friend_dist)
        std_vals = np.std(y)
        p_pos = np.sum(y) / float(len(y))

        return (p_pos / (std_deg * std_vals)) * np.abs(avg_pos_degree - avg_degree)
        

    def rewire_synth(self, goal_corr):
        cur_user_map = user_map
        list_user_map = list(user_map.keys())
        
        delta = 100000

        #rev_user_map = {0:{x:0 for x in cur_user_map if cur_user_map[x] == 0}, 1:{x:0 for x in cur_user_map if cur_user_map[x] == 1}}
        pos_neg_bf = np.array([cur_user_map[x] for x in cur_user_map])
        posn_mapping = {user:ix for ix, user in enumerate(list_user_map)}
        rev_mapping = {ix: user for ix, user in enumerate(list_user_map)}

        positive_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]]#{x:0 for x in rev_user_map[1] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 1])
        negative_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf == 0)[0]]#{x:0 for x in rev_user_map[0] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 0])
        #cur_corr = corr(degree_dist, [*cur_user_map.values()], [*positive_nodes])
        cur_corr = self.corr(pos_neg_bf, positive_nodes)

        #print("done with ", goal_corr, " with corr ", cur_corr)
        iters = 0

        np_rand_choice = np.random.choice
        list_user_map_index = list_user_map.index
        #print("LENGTH OF FRIEND DIST AND POSN MAPPING {} {} {}".format(len(self.friend_dist), len(list_user_map), len(map_user_id)))
        while (cur_corr < goal_corr) and (iters < 10000):

            #es. For example, to increase ρkx, we randomly
            #choose nodes v1 with x = 1 and v0 with x = 0 and swap their attributes if the degree of v0 is
            #greater than the degree of v1. W
            
            #rand_pos = np_rand_choice(list([x for x in positive_nodes]), size=1)[0]
            rand_pos = np_rand_choice([rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]], size=min(100000, int(len(list_user_map)/100)), replace=False)

            rand_neg = np_rand_choice([rev_mapping[x] for x in np.nonzero(pos_neg_bf == 0)[0]], size=min(100000, int(len(list_user_map)/100)), replace=False)
            
            degrees_pos = [self.friend_dist[posn_mapping[x]] for x in rand_pos]
            degrees_neg = [self.friend_dist[posn_mapping[x]] for x in rand_neg]
            neg_gt_pos = [neg >= pos for neg, pos in zip(degrees_neg, degrees_pos)]
            if any(neg_gt_pos):
                true_tups = [ix for ix, val in enumerate(neg_gt_pos) if val]
                for tup_ix in true_tups:
                    pos = rand_pos[tup_ix]
                    neg = rand_neg[tup_ix]
                    cur_user_map[pos] = 0
                    cur_user_map[neg] = 1
                    pos_neg_bf[posn_mapping[pos]] = 0
                    pos_neg_bf[posn_mapping[neg]] = 1

                new_corr = self.corr( pos_neg_bf, [rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]])
                delta = new_corr - cur_corr
                cur_corr = new_corr

            iters += 1
            if iters % 5000 == 0:
                print(cur_corr, delta, iters)
        return  cur_user_map   


def run(params: Dict):
    global model
    'lam=3, ranking=None, activity="Constant", len_feed=30'
    global vals
    global user_map
    global map_user_id
    vals = np.random.default_rng(42).binomial(n=1, p=params['true_prev'], size=len(list(full_graph.nodes)))
    
    user_map = {user:val for user, val in zip(range(len(list(full_graph.nodes))), vals)}
    map_user_id = {user:ix for ix, user in enumerate(list(full_graph.nodes))}
    print("STARTING LENGTH OF USER MAP AND MAP USER ID {} {}".format(len(user_map), len(map_user_id)))
    #split_network_file('twittergraph.edgelist.txt', 8, params['num_agents'], kwargs={x:params[x] for x in ['activity', 'ranking', 'len_feed']})
    #generate_network_file('network.txt', 2, params['num_agents'], kwargs={x:params[x] for x in ['activity', 'ranking', 'len_feed']})
    model = Model(MPI.COMM_WORLD, params)
    
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
    print("DONE")

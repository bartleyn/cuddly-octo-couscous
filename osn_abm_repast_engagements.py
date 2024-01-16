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


def generate_network_file(fname: str, n_ranks: int, n_agents: int, kwargs: Dict[str, str] = {}):
    """Generates a network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = gen_graph(n_agents, 3.0, kwargs)
    #print("Nodes: ", [x for x in g.nodes])
    try:
        import nxmetis
        write_network(g, 'exposure_network', fname, n_ranks, partition_method='metis')
    except ImportError:
        write_network(g, 'exposure_network', fname, n_ranks)

def split_network_file(fname: str, n_ranks: int, n_agents: int, kwargs: Dict[str, str] = {}):
    """Split existing network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = nx.read_edgelist(fname)
    #print("Nodes: ", [x for x in g.nodes])
    try:
        import nxmetis
        write_network(g, 'exposure_network', fname , n_ranks, partition_method='metis')
    except ImportError:
        write_network(g, 'exposure_network', fname, n_ranks)




model = None


class ExposureAgent(core.Agent):


    def __init__(self, nid: int, agent_type: int, rank: int, wealth=1, len_feed=30, last_nodes_seen=None, \
                  mdl=None, real_userid=None, activity="Constant",   lam=3, ranking=None,):
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

            self.real_userid = real_userid
            

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
                 self.real_userid if self.activity == 'Empirical' else None, self.activity)


    def update(self, wealth, len_feed, last_nodes_seen, mdl, real_userid, activity):
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
        model.content_pool.extend([(self.id, cur_time ) for x in range(num_tweets_produced)]) #4807204
        #for x in range(num_tweets_produced):
        #    self.model.content_pool.append((self.unique_id, self.model.schedule.time))

        if self.ranking == None:
            return
        # get the tweets that the user sees
        tweets_seen = model.serve_tweets(num_tweets_consumed, self.id, ranking=self.ranking) 
        num_friends = len(self.friends)
        # update the user's model of the network
        #self.model.update_network(self.unique_id, tweets_seen)

        # interact with each tweet with a certain probability

        likes = self.random.binomial(n=1, p=0.05, size=len(tweets_seen))
        
        user_likes = list(zip([x[0] for x in tweets_seen], likes))

        '''
        def attention_decay_function(position, decay_rate):
            """ Calculate the probability of liking a tweet based on its position in the feed. """
            return np.exp(-decay_rate * position)

        def decide_to_like_tweet(position, decay_rate):
            """ Decide whether to like a tweet based on the decay function. """
            probability_of_liking = attention_decay_function(position, decay_rate)
            return 1 if np.random.rand() < probability_of_liking else 0

        # Parameters
        decay_rate = 0.1  # Adjust this to change how quickly attention decays
        number_of_tweets = len(tweets_seen)  # Total number of tweets in the feed

        # Simulation
        user_likes = [(x[0], decide_to_like_tweet(position, decay_rate)) for position, x in enumerate(tweets_seen)]
        '''



        model.update_likes(self.real_userid, [model.map_user_id[x[0]] for x in user_likes if x[1] == 1])

             
        #for tweet in tweets_seen:
        #    #if self.wealth > 0:
        #    #    other_agent = [agent for agent in model.context.agents() if agent.id == tweet[0]][0]
        #    #    if other_agent is not None:
        #    #        other_agent.wealth += self.wealth / num_friends
        #    #        self.wealth -= self.wealth / num_friends
        #if self.wealth > 0 and self.wealth < 1:
        #    self.wealth = 1
        

        in_deg = model.net.graph.in_degree
        out_deg = model.net.graph.out_degree
        cur_agent = [agent for agent in model.context.agents() if agent.id == self.id][0]

        for tweet in tweets_seen:
            agent = [agent for agent in model.context.agents() if agent.id == tweet[0]][0]
            try:
                float(model.user_map[agent.id])
            except KeyError as e:
                model.user_map[agent.id] = self.random.binomial(n=1, p=model.true_prev, size=1)[0]
            finally:
                self.last_nodes_seen.add((tweet[0], self.id, float(out_deg(cur_agent)), float(model.user_map[agent.id]), float(out_deg(agent)), float(model.map_deg_friends[agent])))

        #self.last_nodes_seen = [tweet[0] for tweet in tweets_seen]



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
        #for x in self.seed_users:
        #    if len(x) < 3:
        #        print(x)
        for prj in self.context.projections.values():
            print("Projection name: ", prj.name)
        
        friends_to_add = []
        self.su_uids = [x[0] for x in self.seed_users]
        for agent in self.context.agents():
            agent._set_real_userid(agent.uid)
            if self.network == 'Empirical' and agent.real_userid in self.su_uids:
                friends_to_add.extend(list(self.net.graph.predecessors(agent)))
        for fr in friends_to_add:
            self.context.add(fr)
        
        ctr = 0
        num_in_if = 0

        for agent in self.context.agents():
            try:
                if self.network == 'Empirical' and agent.real_userid in self.su_uids:
                    agent._set_ranking(params['ranking'])
                    agent._set_friends(list(self.net.graph.predecessors(agent)))
                    agent._set_len_feed(self.len_feed)
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
        self.follower_dist = {x:self.net.graph.out_degree[x] if x in self.net.graph.out_degree else 0.0 for x in self.context.agents()}
        self.friend_dist = [self.net.graph.in_degree[x] for x in self.context.agents()]
        self.mean_in_deg = np.mean(self.friend_dist)
        self.content_pool = []
        self.edges_seen = set()
        self.true_prev = params['true_prev']
        self.likes_tracker = lil_matrix((5599, len(self.net.graph.nodes)), dtype=np.int8)
        if params['network'] == 'Empirical':
            self.vals = self.random.binomial(n=1, p=self.true_prev, size=len(self.net.graph.nodes))
            self.user_map = {user:val for user, val in zip(list(self.context.agents()), self.vals)}
        else:
            self.vals = self.random.binomial(n=1, p=self.true_prev, size=params['num_agents'])
            self.user_map = {user:val for user, val in zip(list(range(params['num_agents'])), self.vals)}
        if params['network'] != 'Empirical':
            self.rewire_synth(params['kx_corr'])
        self.kx_corr = params['kx_corr']

        
        self.nodes = list(self.net.graph.nodes)
        self.ranking = params['ranking']
        self.switch_ranking = params['switch_ranking']
        self.map_user_id = {user.id: self.nodes.index(user) for user in self.nodes}
        self.map_friends = {user: [fr for fr in self.net.graph.predecessors(user)] for user in self.net.graph.nodes}
        self.map_deg_friends = {user: sum([self.follower_dist[fr] if fr in self.follower_dist else 0.0 for fr in self.map_friends[user]]) for user in self.map_friends}

    


        
        self.exp_measures = ExposureMeasures(-1.,-1.)

        loggers = logging.create_loggers(self.exp_measures, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['counts_file'])
        #self.data_set.log(0)

        self.true_prev = params['true_prev']

    '''
    def _seed_rumor(self, init_rumor_count: int, comm):
        world_size = comm.Get_size()
        # np array of world size, the value of i'th element of the array
        # is the number of rumors to seed on rank i.
        rumor_counts = np.zeros(world_size, np.int32)
        if (self.rank == 0):
            for _ in range(init_rumor_count):
                idx = random.default_rng.integers(0, high=world_size)
                rumor_counts[idx] += 1

        rumor_count = np.empty(1, dtype=np.int32)
        comm.Scatter(rumor_counts, rumor_count, root=0)

        for agent in self.context.agents(count=rumor_count[0], shuffle=True):
            agent.received_rumor = True
            self.rumor_spreaders.append(agent)
    '''

    def update_likes(self, real_userid, likes):
        for like in likes:
            self.likes_tracker[self.su_uids.index(real_userid), like] += 1

        #self.likes_tracker[self.su_uids.index(real_userid), [x[0] for x in likes]] += 1

    def at_end(self):
        self.data_set.close()

    def step(self):
        print("Rank: ", self.rank, "staring Tick: ", self.runner.schedule.tick)
       
        total_friends = [(fr.uid, fr.uid[2]) for x in self.context.agents() for fr in self.map_friends[x]] 
        #print("done with gini")
        #self.context.synchronize(restore_agent)
        self.context.request_agents(total_friends, restore_agent)


        if self.ranking == 'NMF':
            nmf_model = NMF(n_components=4, init='random', random_state=0)
            #rt_tracker = lil_matrix((5599, len(who_follows_who['tid'].value_counts().index)), dtype=np.int16)
            try:
                self.W = csr_matrix(nmf_model.fit_transform(self.likes_tracker))
                self.H = csr_matrix(mf_model.components_)
                self.rec = self.W.dot(self.H)
            except ValueError as e:
                print(e)
                print("continuing with random vecs")
                self.rec = self.random.uniform(size=(5599, len(self.net.graph.nodes)))
            

        for node in self.context.agents():
            node.step()

        
        '''
        if context is rank 0 
            then request all the seeduser agents
            update the edges seen
            compute local bias and gini
            log the data

        '''
        self.context.request_agents([(x, x[2]) for x in self.seed_users], restore_agent)

        self.context.synchronize(restore_agent)

        
        if self.rank == 0:
            for agent in self.context.agents():
                self.edges_seen = self.edges_seen.union(agent.last_nodes_seen)
             

            self.exp_measures.local_bias = local_bias(self)
            self.exp_measures.gini = compute_gini(self)
        self.data_set.log(self.runner.schedule.tick)
        self.data_set.write()
        num_tweets = len(self.content_pool)
        print("Rank: ", self.rank, " Tick: ", self.runner.schedule.tick, " with ", len(self.edges_seen), " edges seen ", " with ", num_tweets, " tweets in the content pool")

        #self.context.synchronize(restore_agent)
        
        #print("done with local bias")
        

        

        #print("Rank: ", self.rank, "Tick: ", self.runner.schedule.tick, "Local Bias: ", self.exp_measures.local_bias, "Gini: ", self.exp_measures.gini)
        if self.runner.schedule.tick % 24 == 0:
            if self.activity != 'Empirical':# and int(num_tweets) > 0:
                self.content_pool = self.content_pool#[-int(num_tweets)/2:]
            self.edges_seen = set()
            for agent in self.context.agents():
                agent.last_nodes_seen = set()

            

        

    def serve_tweets(self, num_tweets, user, ranking=None):
        
        try:
            agent = [agent for agent in self.context.agents() if agent.id == user][0]
            neighs = [x.id for x in self.net.graph.predecessors(agent)]
        except nx.exception.NetworkXError as e:
            print(e)
            print(user, self.net.graph.nodes)
            return
        
        cur_tick = self.runner.schedule.tick
        #print("Rank: ", self.rank, "Tick: ", cur_tick, "User: ", user, "switch_ranking: ", self.switch_ranking, "Ranking: ", ranking)
        tweets_seen = [x for x in self.content_pool if x[0] in neighs]#[-int(len(self.content_pool)/2):] if x[0] in neighs]
        if ranking is not None:
            if cur_tick > self.max_iters/2:
                self.ranking = self.switch_ranking
                ranking = self.ranking
            if ranking == "Popularity":
                tweets_seen = sorted(tweets_seen, key=lambda x: self.follower_dist[x[0]] if x[0] in self.follower_dist else 0.0, reverse=True)
            elif ranking == "Random":
                tweets_seen = self.random.choice(tweets_seen, size=min(len(tweets_seen), num_tweets), replace=False)
            elif ranking == "Wealth":
                wealths = {agent.id:agent.wealth for agent in self.schedule.agents}
                tweets_seen = sorted(tweets_seen, key = lambda x: wealths[x[0]], reverse=True)
            elif ranking == "NMF":
                rec_comp = self.rec[self.su_uids.index(user), :]
                tweets_seen = sorted(tweets_seen, key = lambda x: rec_comp[model.map_user_id[x[0]]], reverse=True)
            elif ranking == 'Chronological':
                tweets_seen = sorted(tweets_seen, key=lambda x: x[1], reverse=True)
            


        tweets_seen = tweets_seen[:num_tweets]
        in_deg = self.net.graph.in_degree
        out_deg = self.net.graph.out_degree
        #for tweet in tweets_seen:
        #    #self.edges_seen["{}_{}".format(tweet[0], user)] = 0
        #    self.edges_seen.add((tweet[0], user, float(in_deg(agent)), float(self.user_map[user]), float(out_deg(agent)), float(self.map_deg_friends[agent])))
        #print("User {} asking for {} tweets saw {} tweets from a content pool of size {}".format(user, num_tweets, len(tweets_seen), len(self.content_pool)))
        return tweets_seen
    
    def start(self):
        self.runner.execute()

    def corr(self, y, um):
        avg_pos_degree = np.mean([self.net.graph.in_degree[x] for x in um if x in self.net.graph.in_degree ])
        avg_degree = np.mean(self.friend_dist)
        std_deg = np.std(self.friend_dist)
        std_vals = np.std(y)
        p_pos = np.sum(y) / float(len(y))

        return (p_pos / (std_deg * std_vals)) * np.abs(avg_pos_degree - avg_degree)
        

    def rewire_synth(self, goal_corr):
        cur_user_map = self.user_map
        list_user_map = list(self.user_map.keys())
        
        delta = 100000

        rev_user_map = {0:{x:0 for x in cur_user_map if cur_user_map[x] == 0}, 1:{x:0 for x in cur_user_map if cur_user_map[x] == 1}}
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

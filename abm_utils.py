import mesa
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import jaccard, pdist, cdist

def local_bias(model, attn_func = None):
    edges = np.array(list(model.edges_seen))#np.array([[int(x) for x in edge.split('_')] for i, edge in enumerate(model.edges_seen)])
    #np.array([(friend, user) for friend in model.edges_seen for user in model.edges_seen[friend]])

    user_map = model.user_map
    G_in = model.net.graph.in_degree
    mean_in_deg = model.mean_in_deg
    true_prev = model.true_prev
    in_deg = G_in
    num_edges = edges.shape[0]
    if num_edges <= 1:
        print(" num edges <= 1 but edge shape {}".format(edges.shape))
        return -true_prev
    exp_val = 0.0
    vals = edges#np.zeros((edges.shape[0], 2))
   
    #print("edges: ", edges)
    #vectorize
    try:
        print("num_edges before: ", num_edges)
        vals[:,2] = 1.0/(vals[:,2] + 1.0)
        vals[:,3] = (vals[:,3]*vals[:,2]) / num_edges
        print("after ", num_edges)
    except IndexError as e:
        print("edges: ", edges)
        print(edges[:,0])
        
        raise Exception
    '''
    if attn_func is None:
        def attn_func_int(edge):
            return (1.0/in_deg(edge[1])) if in_deg(edge[1]) > 0 else 0.0
        attn_func = attn_func_int
        #attn_func = lambda edge: (1.0/in_deg(edge[1])) if in_deg(edge[1]) > 0 else 0.0
    num_edges = len(edges)
    exp_val = 0.0
    #def map_func(edge):
    #   return user_map[edge[1]] * attn_func(edge) / num_edges
    #vals = map(map_func, edges)
    vals = [user_map[edge[1]]*attn_func(edge) / num_edges for edge in edges]
    exp_val = np.nansum(list(vals))
    '''
    #print(mean_in_deg, np.nansum(vals[:,3]), vals[:,2])
    return mean_in_deg * np.nansum(vals[:,3]) - true_prev

    return np.mean(list(dict(in_deg).values())) * exp_val - true_prev

'''
def compute_gini(model):
    uniq_vals, counts = np.unique([x[0] for x in model.edges_seen], return_counts=True)
    #num times seen each edge 
    #agent_wealths = [agent.wealth for agent in model.schedule.agents]
    agent_wealths = counts
    x = sorted(agent_wealths)
    N = model.num_agents
    try:
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    except ZeroDivisionError:
        return 0
    return 1 + (1 / N) - 2 * B
'''

def compute_jaccard(model, ):
    # homogenization being the avg jaccard index of a node and all that nodes friends
    # the attribute of the node at each of the X positions in the feed
    sims = []
    agents = model.schedule.agents
    len_feed = model.len_feed
    model_user_map = model.user_map
    for agent_id in agents:
        #if len(agent_id.last_nodes_seen) < 30 or any([len(agents[f_id].last_nodes_seen) < 30 for f_id in agent_id.friends]):
        #    continue
        
        #if not enough nodes seen, assume seen 0s
        #friend_items = np.array([[model_user_map[x] for x in agents[friend_id].last_nodes_seen] + [0 for x in range(len_feed - len(agents[friend_id].last_nodes_seen))]\
        #                        for friend_id in agent_id.friends])
        agent_friends = agent_id.friends
        friend_mat = np.zeros((len(agent_id.friends)+1, len_feed))
        friend_mat[0,:] = np.array([model_user_map[x] for x in agent_id.last_nodes_seen][:len_feed] + [0 for x in range(len_feed - len(agent_id.last_nodes_seen))])

        mask = [(agent_friends.index(friend_id)+1, agents[friend_id].last_nodes_seen.index(x))  for friend_id in agent_friends for x in agents[friend_id].last_nodes_seen\
                                ]
        #mask = [[x[0] for x in mask], [x[1] for x in mask]]
        #mask = [(friend_ix, fof_ix) for fof_ix in agents[friend_id].last_nodes_seen]
        #print("mask: ", mask)
        try:
            friend_mat[tuple(zip(*mask))] = 1
        except IndexError as e:
            print("num friends {} \t num last_nodes_seen {}".format(len(agent_id.friends), len(agent_id.last_nodes_seen)))
            print("mask: ", mask)
            raise Exception(e)

        #print(len(agent_items), len(agent_id.last_nodes_seen))

        if friend_mat.size == 0 or len(agent_friends) == 1:
            continue
        try:
            agent_dists = 1 - pdist(friend_mat, 'jaccard')[:len(agent_friends)]#[min(1, len(agent_friends)-1):min(1,len(agent_friends)-1)+1]
        except ValueError as e:
            #print(agent_items)
            print(friend_mat)
            raise Exception(e)
        #print(agent_dists, len(agent_friends))
        sims.append(np.nanmean(agent_dists))

    return np.mean(sims) if any([x >= 0 for x in sims]) else -1

def compute_gini(model, ):
    
    '''total_times_user_seen = [x for x in df_obs.sum(axis=1) if x > 0.]
    num_users = len(np.unique(df_obs.nonzero()[0]))
    gini = 0
    for x_i in total_times_user_seen:
        for x_j in total_times_user_seen:
            gini += np.abs(x_i - x_j)
    
    #print("gini total before norm: {} len {} sum {}".format(gini, num_users, sum(total_times_user_seen_leah)))
    
    gini /= 2 * num_users * sum(total_times_user_seen)
    
    return gini'''
    """Calculate the Gini coefficient of a numpy array."""
    # All values are treated equally, arrays must be 1d:
    uniq_vals, counts = np.unique([x[0] for x in model.edges_seen], return_counts=True)
    array = counts.astype('float64')
    array = array.flatten()
    try:
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
    except ValueError:
        pass
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    #print("n {} sum_array {}".format(n, np.sum(array)))
    if n == 0:
        return np.nan
    else:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def gen_graph(num_nodes, exp, params):
    seq = [int(x) for x in nx.utils.powerlaw_sequence(num_nodes, exp)]
    G = nx.directed_configuration_model(seq, seq)
    G=nx.DiGraph(G)
    for param in params:
        for node in G.nodes():
            G.nodes[node][param] = params[param]
    #G.remove_edges_from(G.selfloop_edges())
    return G

def adjust_assort(G, goal_assort):
    tries = 0
    while (nx.degree_assortativity_coefficient(G, x='in', y='in')) > goal_assort and not tries > 500:

        cur_asst = nx.degree_assortativity_coefficient(G, x='in', y='in')
        Gc = G.copy()
        nx.algorithms.swap.directed_edge_swap(Gc, nswap=50, max_tries=1e6)
        if nx.degree_assortativity_coefficient(Gc, x='in', y='in') < cur_asst:
            G = Gc
            tries = 0
        if tries % 50 == 0:
            print(tries, cur_asst)
        tries += 1
    return G


class OSNAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, lam=3, ranking=None, activity="Constant"):
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # Create the agent's attribute and set the initial values.
        self.wealth = 1
        self.lam = lam
        self.ranking = ranking
        self.friends = self.model.grid.get_neighbors(self.unique_id)
        self.activity = activity
        self.len_feed = model.len_feed
        self.last_nodes_seen = []
        if activity != "Constant":
            self.poisson = self.random.poisson
   
    def step(self):
        # pick a number of tweets that the user consumes from their timeline
        num_tweets_consumed = self.len_feed
        if self.activity == "Constant":
            num_tweets_produced = 10
        else:
            num_tweets_produced = self.poisson(self.lam, size=1)[0]
        cur_time = self.model.schedule.time
        uid = self.unique_id
        self.model.content_pool.extend([(uid, cur_time) for x in range(num_tweets_produced)])
        #for x in range(num_tweets_produced):
        #    self.model.content_pool.append((self.unique_id, self.model.schedule.time))

        tweets_seen = self.model.serve_tweets(num_tweets_consumed, self.unique_id, ranking=self.ranking) 
        num_friends = len(self.friends)
        # update the user's model of the network
        #self.model.update_network(self.unique_id, tweets_seen)

        
        for tweet in tweets_seen:
            if self.wealth > 0:
                other_agent = [agent for agent in self.model.schedule.agents if agent.unique_id == tweet[0]][0]
                if other_agent is not None:
                    other_agent.wealth += self.wealth / num_friends
                    self.wealth -= self.wealth / num_friends
        if self.wealth > 0 and self.wealth < 1:
            self.wealth = 1
        

        self.last_nodes_seen = [tweet[0] for tweet in tweets_seen]






class OSNModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, graph, assort, kx_corr=0.0, len_feed=30, lam=None, ranking=None, activity="Constant"):
        self.num_agents = N
        graph = adjust_assort(graph, assort)
        self.graph = graph
        self.follower_dist = graph.out_degree
        self.friend_dist = [graph.in_degree[x] for x in graph.nodes]
        self.in_degree = self.graph.in_degree
        self.grid = mesa.space.NetworkGrid(graph)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.random = np.random.default_rng()
        self.content_pool = []
        self.edges_seen = set()
        self.true_prev = 0.05
        self.vals = self.random.binomial(n=1, p=self.true_prev, size=self.num_agents)
        self.user_map = {user:val for user, val in zip(list(range(self.num_agents)), self.vals)}
        self.rewire_synth(kx_corr)
        self.kx_corr = kx_corr
        
        self.ctr = 0
        self.len_feed=len_feed
        self.mean_in_deg = np.mean(list(dict(self.graph.in_degree).values()))
        

        # Create agents
        for i in range(self.num_agents):
            a = OSNAgent(i, self, lam, ranking=ranking, activity="Not")
            self.schedule.add(a)
            # Add the agent to a random grid cell
            try:
                self.grid.place_agent(a, i)
            except KeyError as e:
                self.running = False
                print(e)
                return
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini, "B_local": local_bias, "Jaccard": compute_jaccard}, agent_reporters={"Wealth": "wealth"}
        )

    def serve_tweets(self, num_tweets, user, ranking=None):
        neighs = self.grid.get_neighbors(user)
        # TODO: use a higher lookback for content pool
        tweets_seen = [x for x in self.content_pool[-4*num_tweets:] if x[1] in neighs]
        if ranking:
            if ranking == "Popularity":
                tweets_seen = sorted(tweets_seen, key=lambda x: self.follower_dist[x[0]], reverse=True)
            elif ranking == "Random":
                tweets_seen = self.random.choice(tweets_seen, size=min(len(tweets_seen), num_tweets), replace=False)
            elif ranking == "Wealth":
                wealths = {agent.unique_id:agent.wealth for agent in self.schedule.agents}
                tweets_seen = sorted(tweets_seen, key = lambda x: wealths[x[0]], reverse=True)

        tweets_seen = tweets_seen[:num_tweets]
        in_deg = self.graph.in_degree
        for tweet in tweets_seen:
            #self.edges_seen["{}_{}".format(tweet[0], user)] = 0
            self.edges_seen.add((tweet[0], user, float(in_deg(user)), float(self.user_map[user])))
        return tweets_seen

    def step(self):
        try:
            self.datacollector.collect(self)
            self.schedule.step()
            self.ctr += 1
            #if self.ctr % 100 == 0:
            #    self.content_pool = self.content_pool[-self.len_feed*self.num_agents:]
            #    #self.edges_seen = set()
        except nx.NetworkXError as e:
            print(e)
            return
        except KeyError as e:
            print(e)
            return
        

    def corr(self, y, um):
        avg_pos_degree = np.mean([self.graph.in_degree[x] for x in um])
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

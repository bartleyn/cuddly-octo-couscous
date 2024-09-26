import networkx as nx
from typing import Dict
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
import mesa
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import jaccard, pdist, cdist



'''
    gini_positive: float
    gini_negative: float
    std_likes_positive: float
    std_likes_negative: float
    num_edges_seen_positive: float
    num_edges_seen_negative: float
    precision_at_30_positive: float
    precision_at_30_negative: float
    '''
def compute_std_likes(model, type='all'):
    if type == 'all':
        return np.std(model.likes_tracker.sum(axis=1))
    if type == 'positive':
        return np.std(model.likes_tracker[[model.su_uids_mapped.index(x) for x in model.positive_users],:].sum(axis=1))
    else:
        return np.std(model.likes_tracker[[model.su_uids_mapped.index(x) for x in model.negative_users],:].sum(axis=1))

def compute_mean_likes(model, type='all'):
    if type == 'all':
        return np.mean(model.likes_tracker.sum(axis=1))
    if type == 'positive':
        print("HELLO PS USERS ", list(model.positive_users)[:10])
        print("num map user id vals ", len(model.map_user_id))
        print("first tep of map user id ", list(model.su_uids_mapped)[:10], " and if 0 in it: ", 0 in model.map_user_id)
        return np.mean(model.likes_tracker[[model.su_uids_mapped.index(x) for x in model.positive_users],:].sum(axis=1))
    if type == 'negative':
        return np.mean(model.likes_tracker[[model.su_uids_mapped.index(x) for x in model.negative_users],:].sum(axis=1))


def compute_mean_tweets_per_user(model, type='all'):
    #compute the mean number of tweets per user in the content_pool
    #return np.mean(model.tweets_tracker.sum(axis=0))
    if type == 'all':
        return np.mean(model.likes_tracker.sum(axis=0))
    if type == 'positive':
        return np.mean(model.likes_tracker[[model.su_uids_mapped.index(x) for x in model.positive_users],:].sum(axis=0))
    if type == 'negative':
        return np.mean(model.likes_tracker[[model.su_uids_mapped.index(x) for x in model.negative_users],:].sum(axis=0))


def compute_std_tweets_per_user(model, type='all'):
    return np.std(model.likes_tracker.sum(axis=0))

def compute_precision_at_k(model,k, type='all'):
    if type == 'all':
        ix = [True] * 5599
    if type == 'positive':
        ix = [model.su_uids_mapped.index(x) for x in model.positive_users]
    if type == 'negative':
        ix = [model.su_uids_mapped.index(x) for x in model.negative_users]

    if k == 10:
        return np.mean(model.precision_tracker[ix,0])
    elif k == 20:
        return np.mean(model.precision_tracker[ix,1])
    elif k == 30:
        return np.mean(model.precision_tracker[ix,2])
    else:
        return -1
        #return np.mean(model.precision_tracker[:,2])

def local_bias(model, type='all', attn_func = None):
    edges = np.array(list(model.edges_seen))#np.array([[int(x) for x in edge.split('_')] for i, edge in enumerate(model.edges_seen)])
    #np.array([(friend, user) for friend in model.edges_seen for user in model.edges_seen[friend]])

    ###  each edge is (friend, cur_user, (tweet[0], self.id, float(in_deg(agent)), float(model.user_map[agent.id]), float(out_deg(agent)), float(model.map_deg_friends[agent])))

    #each edge is (friend id, 
    #              currrent user id, 
    #              current user out_degree, 
    #              friend 0/1, 
    #              friend out_degree, float(model.map_deg_friends.get(agent, 0))))
    user_map = model.user_map
    G_in = model.in_degree#model.net.graph.in_degree # fix the in degree
    mean_in_deg = model.mean_in_deg
    true_prev = model.true_prev
    in_deg = G_in
    num_edges = edges.shape[0]

    if num_edges <= 1:
        print(" num edges <= 1 but edge shape {}".format(edges.shape))
        return -true_prev
    exp_val = 0.0
    vals = edges#np.zeros((edges.shape[0], 2))
    current_user_labels = [user_map[x] if x in user_map else -1 for x in vals[:,1] ]
    #print("edges: ", edges)
    #vectorize
    try:
        print("num_edges before: ", num_edges)
        vals[:,2] = 1.0/(vals[:,2])
        vals[vals[:,2] == 1.0,2] = np.nan
        vals[vals[:,3] == -1, 3] = np.nan
        vals[:,3] = (vals[:,3]*vals[:,2]) / num_edges
        print("after ", num_edges)
    except IndexError as e:
        print("edges: ", edges)
        print(edges[:,0])
        
        raise Exception
 
    
    print("mean {}\t sum{}\td {}".format(mean_in_deg, np.nansum(vals[:,3]), vals[:,2]))
    if type == 'all':
        return mean_in_deg * np.nansum(vals[:,3]) - true_prev
    elif type == 'positive':
        return mean_in_deg * np.nansum(vals[[user_map[x] == 1 if x in user_map else False for x in vals[:,1]],3]) - true_prev
    else:
        return mean_in_deg * np.nansum(vals[[user_map[x] == 0 if x in user_map else False for x in vals[:,1]], 3]) - true_prev

    return np.mean(list(dict(in_deg).values())) * exp_val - true_prev



def compute_jaccard(model, ):
    # homogenization being the avg jaccard index of a node and all that nodes friends
    # the attribute of the node at each of the X positions in the feed
    sims = []
    agents = model.schedule.agents
    len_feed = model.len_feed
    model_user_map = model.user_map
    for agent_id in agents:

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

def compute_gini(model, type='all'):
    
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
    if type == 'all':

        uniq_vals, counts = np.unique([x[0] for x in model.edges_seen], return_counts=True)
    elif type == 'positive':
        uniq_vals, counts = np.unique([x[0] for x in model.edges_seen if model.user_map.get(x[1],-1) == 1], return_counts=True)
    else:
        uniq_vals, counts = np.unique([x[0] for x in model.edges_seen if model.user_map.get(x[1],-1) == 0], return_counts=True)

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




    def corr(in_degree, friend_dist, y, um):
        avg_pos_degree = np.mean([self.graph.in_degree[x] for x in um])
        avg_degree = np.mean(self.friend_dist)
        std_deg = np.std(self.friend_dist)
        std_vals = np.std(y)
        p_pos = np.sum(y) / float(len(y))

        return (p_pos / (std_deg * std_vals)) * np.abs(avg_pos_degree - avg_degree)
        
       

    def rewire_synth(user_map, in_degree, friend_dist, goal_corr):
        cur_user_map = user_map
        list_user_map = list(user_map.keys())
        
        delta = 100000

        rev_user_map = {0:{x:0 for x in cur_user_map if cur_user_map[x] == 0}, 1:{x:0 for x in cur_user_map if cur_user_map[x] == 1}}
        pos_neg_bf = np.array([cur_user_map[x] for x in cur_user_map])
        posn_mapping = {user:ix for ix, user in enumerate(list_user_map)}
        rev_mapping = {ix: user for ix, user in enumerate(list_user_map)}

        positive_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf)[0]]#{x:0 for x in rev_user_map[1] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 1])
        negative_nodes = [rev_mapping[x] for x in np.nonzero(pos_neg_bf == 0)[0]]#{x:0 for x in rev_user_map[0] if x in degrees}#set([node for node,assignment in zip(list_user_map, cur_assignments) if assignment == 0])
        #cur_corr = corr(degree_dist, [*cur_user_map.values()], [*positive_nodes])
        cur_corr = corr(in_degree, friend_dist,pos_neg_bf, positive_nodes)

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
    

def train_logit(model):
    global model_bk
        
    model.likes_tracker
    #I have collated tweet data and the actual retweets the users did 
    # current_time - original_time, original_user_id, if hashtag, retweet_yn
    # Sample DataFrame
    all_user_retweets = activity_df[activity_df['RT'] == 'RT']
    found_rts_users = rts[rts['re_tweet_id'].isin(all_user_retweets['tweet_id'])][['original_user_id', 'retweet_time', 'original_time', 'original_tweet_id']]

    found_rts_users['friend_deg'] = found_rts_users['original_user_id'].apply(lambda x: degrees_out[x] if x in degrees_out else -1)
    found_rts_users['friend_bin'] = found_rts_users['original_user_id'].apply(lambda x: 1 if user in map_friends and x in map_friends[user] else 0)
    #found_rts_users['retweet_time'] = pd.to_datetime(found_rts_users['retweet_time'])
    #found_rts_users['original_time'] = pd.to_datetime(found_rts_users['original_time'])
    found_rts_users['RT'] = [1 for x in found_rts_users.index]
    #found_rts_users['timedelta'] = (found_rts_users['retweet_time'] - found_rts_users['original_time']).dt.total_seconds()

    all_friend_tweets = collated_tweet_df[['user_id', 'date_created', 'tweet_id']]
    all_friend_tweets['tweet_id'] = all_friend_tweets['tweet_id'].astype(int)
    #all_friend_tweets['original_time'] = pd.to_datetime(all_friend_tweets['date_created'])
    
    frs_with_rts = all_friend_tweets[all_friend_tweets['tweet_id'].isin(found_rts_users['original_tweet_id'])].index
    all_friend_tweets['friend_deg'] = all_friend_tweets['user_id'].apply(lambda x: degrees_out[x] if x in degrees_out else -1)
    all_friend_tweets['friend_bin'] = all_friend_tweets['user_id'].apply(lambda x: 1 if x in map_friends[user] else 0)

    all_friend_tweets['RT'] = [0 for x in all_friend_tweets.index]
    all_friend_tweets.loc[frs_with_rts, 'RT'] = 1
    #print("frs with rts ", frs_with_rts, " and all frs ", all_friend_tweets['RT'].describe())
    
    df = pd.concat([found_rts_users, all_friend_tweets])[['friend_deg', 'friend_bin', 'RT']]


    #print("full df rt vc ", df['RT'].value_counts())
    features = ['friend_deg', 'friend_bin']  # Add other relevant features
    X = df[features]
    y = df['RT']

    # Splitting data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Training the logistic regression model
        model = LogisticRegression()

        model.fit(X_train, y_train)
    except ValueError as e:
        print("user {} had error {}".format(user, e))
        model = model_bk
    
    if model_bk is None:
        model_bk = model



    # Make predictions
    try:
        predictions = model.predict_proba(all_friend_tweets[['friend_deg', 'friend_bin']])[:,1]
    except ValueError as e:
        predictions = [1.0 for x in collated_tweet_df.index]
    #print(predictions)
    # The 'predictions' variable now contains the predicted probabilities of engagement for each tweet
    # You can add these probabilities to your DataFrame for further analysis or ranking
    collated_tweet_df['predicted_prob_engagement'] = predictions

    # Now you can rank tweets based on the predicted probabilities
    collated_tweet_df = collated_tweet_df.iloc[np.argsort(predictions)[::-1]]


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

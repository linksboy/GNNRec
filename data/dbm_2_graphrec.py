
"""
    dbm_meta:movie-type (movie_id, type_id, 1)
    dbm_ratings:(user_id, movie_id, trust)
    dbm_trust:(user_id, user_id,1 )

    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys


dbm_meta=pd.read_csv(sys.path[0] + "/data/dbm_meta.txt",sep='\s+',header=None,names=["movie",'meta',"rating"])
dbm_ratings=pd.read_csv(sys.path[0] + "/data/dbm_ratings.txt",sep='\s+',header=None,names=["user","movie","rating"])
dbm_trust=pd.read_csv(sys.path[0] + "/data/dbm_trust.txt",sep='\s+',header=None,names=["user1","user2","rating"])

users_set=set(dbm_ratings['user'])
movie_set=set(dbm_ratings["movie"])
meta_set=set(dbm_meta['meta'])

user_map={}
movie_map={}
meta_map={}

history_u_lists={}
history_ur_lists={}
history_v_lists={}
history_vr_lists={}
history_m_lists={}
history_mr_lists={}
train_u=[]
train_v=[]
train_r=[]
test_u=[]
test_v=[]
test_r=[]
social_adj_lists={}

num_user = 0
num_movie=0
num_meta = 0
for i in users_set:
    user_map[i]=num_user
    history_u_lists[num_user]=[]
    history_ur_lists[num_user]=[]
    num_user+=1
for i in movie_set:
    movie_map[i]=num_movie
    history_v_lists[num_movie]=[]
    history_vr_lists[num_movie]=[]
    history_m_lists[num_movie] = []
    history_mr_lists[num_movie] = []
    num_movie+=1

for i in meta_set:
    meta_map[i]=num_meta
    num_meta+=1

print(num_user,num_movie)
for i in tqdm(range(dbm_ratings.shape[0])):
    history_u_lists[user_map[dbm_ratings["user"][i]]].append(movie_map[dbm_ratings["movie"][i]])
    history_ur_lists[user_map[dbm_ratings["user"][i]]].append(dbm_ratings["rating"][i])
    history_v_lists[movie_map[dbm_ratings["movie"][i]]].append(user_map[dbm_ratings["user"][i]])
    history_vr_lists[movie_map[dbm_ratings["movie"][i]]].append(dbm_ratings["rating"][i])

print(num_meta)

for i in tqdm(range(dbm_meta.shape[0])):
    if dbm_meta['movie'][i] not in movie_map:
        continue
    history_m_lists[movie_map[dbm_meta['movie'][i]]].append(meta_map[dbm_meta['meta'][i]])
    history_mr_lists[movie_map[dbm_meta['movie'][i]]].append(dbm_meta['rating'][i])

from sklearn.model_selection import train_test_split

train, test = train_test_split(dbm_ratings.to_numpy())

train=np.array(train).T
test=np.array(test).T
train_u=[user_map[i] for i in train[0]]
train_v=[movie_map[i] for i in train[1]]
train_r=train[2]
test_u=[user_map[i] for i in test[0]]
test_v=[movie_map[i] for i in test[1]]
test_r=test[2]


for i in range(num_user):
    social_adj_lists[i]=set()



for i in range(dbm_trust.shape[0]):
    if dbm_trust['user1'][i] not in user_map or dbm_trust['user2'][i] not in user_map:
        continue
    social_adj_lists[user_map[dbm_trust['user1'][i]]].add(user_map[dbm_trust['user2'][i]])
    social_adj_lists[user_map[dbm_trust['user2'][i]]].add(user_map[dbm_trust['user1'][i]])

ratings_list={2.0: 0, 1.0: 1, 3.0: 2, 4.0: 3, 2.5: 4, 3.5: 5, 1.5: 6, 0.5: 7}

def export():
    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, history_m_lists, history_mr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list, num_meta
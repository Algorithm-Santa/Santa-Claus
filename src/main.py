import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
from os import getcwd
from os.path import join
import matplotlib.pyplot as plt


import random
import functools as ft
from collections import ChainMap

from sklearn.metrics.pairwise import haversine_distances
from math import radians

weight_limit = 1000
class Simulated_Anealing:
    def __init__(self):
        self.ITERATION_COUNT = 2
        self.DECREASE_STEP = 0.0001
        self.a = 0.95
        #// Pick an initial temperature to allow "mobility" 
        self.T = 125 #selectInitialTemperature()
    
    def simulated_annealing(self,tour):
        tour = tour.reset_index(inplace=False)
        #print("Tour Length: ", len(tour))
        ITERATION_COUNT = self.ITERATION_COUNT 
        DECREASE_STEP = self.DECREASE_STEP
        a = self.a 
        #// Pick an initial temperature to allow "mobility" 
        T = self.T  #selectInitialTemperature()

        #// Start with any tour, e.g., in input order 
        s = list(tour['Subtour'])[0] #0,1,...,n-1
        
        
        ## Record initial tour as best so far.
        
        weights = tour['Weight'].to_numpy()
        coordinates = tour[['Latitude','Longitude']].to_numpy()
                
        min_cost = self.weighted_distance(coordinates, weights,weight_limit=1000,sleigh_weight=10)
        minTour = s
        mintour_0 = s
  
        #// Iterate "long enough" 
        nothing_changed_counter = 0
        for i in range(ITERATION_COUNT):
            #// Randomly select a neighboring state.
            #print("Iteration i: ", i)
            s_ = self.randomNextState (s)
            
            #// If it's better, then jump to it. 
            #print("s_ :",type(s_)," ",s_)
            #print("s :",type(s)," ",s)
            new_tour_df = tour.set_index('GiftId').loc[s_].reset_index(inplace=False)
            weights_s_ = new_tour_df['Weight'].to_numpy()
            coordinates_s_ = new_tour_df[['Latitude','Longitude']].to_numpy()           
            cost_s_ = self.weighted_distance(coordinates_s_, weights_s_,weight_limit=1000,sleigh_weight=10)
            
            current_tour_df = tour.set_index('GiftId').loc[s].reset_index(inplace=False)
            weights_s = current_tour_df['Weight'].to_numpy()
            coordinates_s = current_tour_df[['Latitude','Longitude']].to_numpy()    
            cost_s = self.weighted_distance(coordinates_s, weights_s,weight_limit=1000,sleigh_weight=10)
            if cost_s_ < cost_s:
                nothing_changed_counter = 0
                s = s_
                ##// Record best so far: 
                if cost_s_ < min_cost:
                    min_cost = cost_s_
                    minTour = s_
            elif self.expCoinFlip(s, s_,cost_s,cost_s_,T):
               #// Jump to s' even if it's worse. 
               s = s_
               #// Else stay in current state.
               nothing_changed_counter +=1
               if nothing_changed_counter > 3000:
                   break
          
            #// Decrease temperature. 
            T = T*a
        #print(minTour)
        #print(minTours)

        return minTour,min_cost

    def randomNextState(self,s):
        s = s.copy()
        upper_bound = len(s)-1           
        n_1 = random.randint(0,upper_bound)  
        n_2 = random.randint(0,upper_bound) 
        #print("s before exchange: ", s)
        temp = s[n_1]
        s[n_1]=s[n_2]
        s[n_2] =temp
        
        #print("s after exchange: ", s)
        return s

    def expCoinFlip(self, s, s_,cost_s,cost_s_, T):
        #Input: two states s and s'
        found_smaller_number = False
        p = np.exp( -(cost_s_ - cost_s) / T)
        u = random.random()
        if u < p:
            found_smaller_number = True
        return found_smaller_number
    
    def weighted_distance(self, coordinates, weights,weight_limit=1000,sleigh_weight=10):
        startweight = sleigh_weight + np.sum(weights)
        if startweight > weight_limit:
            return -1

        north_pole = np.radians([90,0])
        coords = np.vstack((north_pole,np.radians(coordinates),north_pole))
        #print("coords (function)",coords)
        distances = []
        for i in range(len(coords)-1):
            distances.append(haversine_distances([coords[i],coords[i+1]])[0][1]*6371000/1000)
            
        distances = np.array(distances)
        #print("distances (function):",distances)
        #adj_matrix = haversine_distances(coords,np.roll(coords.copy(),-1,axis=0))
        #adj_matrix = adj_matrix * 6371 #6371000/1000
        #distances = np.diag(adj_matrix)[:-1]

        #weights +=sleigh_weight
        weights = np.append(weights,sleigh_weight)
        weights = np.cumsum(weights[::-1])[::-1] # flip, cummulative sum, flip again
        weighted_dist = np.sum(weights*distances)
        
        return weighted_dist


class Graph:
    def __init__(self, gifts,tourId_provided = False):
        self.numEdges = 0
        self.sort_gifts = False
        #data = self.init_tours(gifts)
        #self.tourgraph = pd.DataFrame(data)
        if tourId_provided:
            """
            Assuming that an initial list of gifts with 
            corresponding tourIds is available.
            """
            self.tourgraph = gifts
            self.sort_gifts = True
        else:
            """
            assuming that just the original df is given 
            --> Naive tourlist with n tours
            """
            data = self.init_tours(gifts)
            self.tourgraph = pd.DataFrame(data)
        
    def init_tours(self, gifts):
        tripIds = np.arange(len(gifts))
        gifts_copy = gifts.copy()
        gifts_copy['TripId'] = tripIds
        return gifts_copy
    
    def optimize_subtours(self,optimizer):
        simulated_annealing_alg = optimizer
        trips = self.tourgraph
        optimized_trips = trips.copy()
        grouped_trips = trips.groupby('TripId')
        fig = plt.figure()
        total_weariness = 0
        optimized_df = pd.DataFrame()
        for group_name, trip in grouped_trips:
            if self.sort_gifts:
                trip = trip.sort_values(['Latitude','Longitude'],ascending=False)
            
            #print(trip['GiftId']*len(trip))
            trip['Subtour'] = [list(trip['GiftId'])]*len(trip)
            plt.plot(trip['Latitude'],trip['Longitude'])
            optimized_trip,current_weariness = simulated_annealing_alg.simulated_annealing(trip)
            total_weariness += current_weariness
            subtours = [optimized_trip]*len(trip)
            trip['Subtour'] = subtours   
            optimized_df = pd.concat([optimized_df,trip],axis=0)            
        return optimized_df, total_weariness,fig
            
        
    def weighted_reindeer_weariness(self):
        weighted_weariness = 0
        tot_distance = 0
        trips = self.tourgraph
        grouped_trips = trips.groupby('TripId')
        fig = plt.figure()
        for group_name, trip in grouped_trips:
            if self.sort_gifts:
                trip = trip.sort_values(['Latitude','Longitude'],ascending=False)
            
            trip['Subtour'] = list(trip['GiftId'])*len(trip)
            plt.plot(trip['Latitude'],trip['Longitude'])
                
            weights = trip['Weight'].to_numpy()
            coordinates = trip[['Latitude','Longitude']].to_numpy()
            current_wearniess= self.weighted_distance(coordinates,weights)
            weighted_weariness += current_wearniess 

        print("weighted_weariness ",weighted_weariness)
        return weighted_weariness
    
    def weighted_distance(self, coordinates, weights,weight_limit=1000,sleigh_weight=10):
        startweight = sleigh_weight + np.sum(weights)
        if startweight > weight_limit:
            return -1

        north_pole = np.radians([90,0])
        coords = np.vstack((north_pole,np.radians(coordinates),north_pole))
        #print("coords (function)",coords)
        distances = []
        for i in range(len(coords)-1):
            distances.append(haversine_distances([coords[i],coords[i+1]])[0][1]*6371000/1000)
            
        distances = np.array(distances)
        #print("distances (function):",distances)
        #adj_matrix = haversine_distances(coords,np.roll(coords.copy(),-1,axis=0))
        #adj_matrix = adj_matrix * 6371 #6371000/1000
        #distances = np.diag(adj_matrix)[:-1]

        #weights +=sleigh_weight
        weights = np.append(weights,sleigh_weight)
        weights = np.cumsum(weights[::-1])[::-1] # flip, cummulative sum, flip again
        weighted_dist = np.sum(weights*distances)
        return weighted_dist


def create_sclices(df):
    slices = []
    offset=0.5
    for i in range(-180,181):
        j = i+offset
        slices.append(df[(df['Longitude']>=(j-offset)) & (df['Longitude']<(j+offset))])
    return slices

def convert_tour_dict_to_df(tours):
    res = {} 
    for dict in tours: 
        for list in dict: 
            if list in res: 
                res[list] += (dict[list]) 
            else: 
                res[list] = dict[list] 
    return pd.DataFrame(res)

def append_location(tour,row,tripId):
        tour['GiftId'].append(row['GiftId'])
        tour['Latitude'].append(row['Latitude'])
        tour['Longitude'].append(row['Longitude'])
        tour['Weight'].append(row['Weight']) 
        tour['TripId'].append(tripId) 
        
def create_tours_from_slices(slices_list):
    slices = slices_list
    tours = []
    tripId = 0
    
    tour = {"GiftId" : [],"Latitude" : [],"Longitude" :[],"Weight" : [],"TripId":[]}

    for s in slices:
        for index,row in s.iterrows():
            sum_current_tour = ft.reduce(lambda x,y:x+y,tour['Weight'],0) 
            if (sum_current_tour+row['Weight'])<=weight_limit:
                append_location(tour,row,tripId)
            else:
                tripId +=1
                sum_current_tour = 0
                tours.append(tour.copy())
                tour = {"GiftId" : [],"Latitude" : [],"Longitude" :[],"Weight" : [],"TripId":[]}
                append_location(tour,row,tripId)
                
    tours.append(tour.copy())
    tours = convert_tour_dict_to_df(tours)
    return tours


if __name__ == "__main__":
    file_path = "../data/"
    df = pd.read_csv('../data/gifts.csv')
    solutions=pd.read_csv('../data/solutions.csv')
    searchGrid = {
        "T": [10,100,1000,10000,1000000],
        "alpha":[0.9,0.95,0.98,0.99],
        "Iterations":[10,100,1000,10000,100000,1000000]
    }
    sa = Simulated_Anealing()
    for iterations in searchGrid['Iterations']:
        for T in searchGrid['T']:
            for alpha in searchGrid['alpha']:
                tours = pd.read_csv("../data/tours.csv")
                graph = Graph(tours,tourId_provided = True)  
                sa.T = T
                sa.a = alpha
                sa.ITERATION_COUNT = iterations
                print("ITERATION_COUNT: ",sa.ITERATION_COUNT)
                print("alpha : ",sa.a)
                print("Temperature :",sa.T)
                optimized_df, total_weariness,fig = graph.optimize_subtours(sa)
                print("total_weariness :" ,total_weariness)
                TEMPERATURE = str(T)
                ALPHA = str(alpha)
                ITERATIONS = str(iterations)
                
                optimized_df_name = file_path+ITERATIONS + "_"+TEMPERATURE + "_"+ALPHA+".csv"
                figure_name = file_path+ITERATIONS + "_"+TEMPERATURE + "_"+ALPHA+".png"
                data ={
                        'solution_dataframes':[optimized_df_name],
                        'total_weariness':[total_weariness],
                        'plots':[figure_name],
                        "Ts":[T],
                        "alphas":[alpha],
                        "Iterations":[iterations]
                    }
                solutions = pd.concat((solutions,pd.DataFrame(data)),axis=0,ignore_index=True)
                solutions.reset_index(inplace=True, drop=True) 
                solutions.to_csv('../data/solutions.csv')
                optimized_df.to_csv(optimized_df_name)
                fig.savefig(figure_name)



    
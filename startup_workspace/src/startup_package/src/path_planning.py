import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

class PathPlanner:
    def __init__(self):
        # load the files we need
        
        # 1. node locations
        self.nodes_df = pd.read_csv("/simulator/startup_workspace/src/startup_package/src/node_data/nodes.csv")
        print("loaded nodes df: ",type(self.nodes_df))
        # 2. path of nodes to follow
        self.planned_path = list(pd.read_csv("/simulator/startup_workspace/src/startup_package/src/node_data/path_1.csv"))
        print("loaded planned path: ",type(self.planned_path))
        self.path_progress_counter = 0
        self.node_current=0
        # 3. intersection actions   
        
    def getCurrentNode(self,x,y):
        # return the node ID of the current GPS location
        coord = np.array([[x,y]])
        #calculate the distance from "coord" to each node
        dm = distance_matrix(coord,self.nodes_df.values)
        #figure ouyt the index of the closest node
        #by finding the minimum distance
        node_index = np.argmin(dm)
        #convert the node index to a node ID
        #by looking it up in the original dataframe
        node_id = self.nodes_df.iloc[node_index].name
        return node_id

    def updatePathProgress(self,current_x,current_y):
        node_prev = self.node_current
        self.node_current = self.getCurrentNode(current_x,current_y)
        
        #we're progressing through the path
        if self.node_current == self.node_next:
            print("progressed to node %d"%self.node_current)
            self.path_progress_counter+=1
            self.node_next = self.planned_path[self.path_progress_counter+1]
        #we're still at the same point in the path
        elif self.node_current == node_prev:
            print("still at node %d"%self.node_current)
        #something went wrong and we've drifted from the path
        else:
            print("expected to move from node %d to %d, but currently at %d"%(node_prev,self.node_next,self.node_current))

    def getNextIntersectionAction(self,x,y):
        current_loc = getCurrentNode(x,y)

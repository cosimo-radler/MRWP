import networkx as nx
import random

def graphModelRun(G,tMax,groupA,groupB,p):
    #output data list
    data=[]
    #the three states in our system u=undecided, a=groupA, and b=groupB
    u=0
    a=1
    b=2
    initState={}
    #add the state attribute the the nodes of the graph
    for n in G:
        initState.update({n:{'state':[]}})
    nx.set_node_attributes(G,initState)
    #initialise the dynamics of the graph
    nodesInState={u:0,a:0,b:0}    
    for n in G:
      if n in groupA:
           G.nodes[n]['state'].append(a)
           nodesInState[a]+=1   
      elif n in groupB:
          G.nodes[n]['state'].append(b)
          nodesInState[b]+=1                    
      else:
          G.nodes[n]['state'].append(u)
          nodesInState[u]+=1  
    data.append([nodesInState[u],nodesInState[a],nodesInState[b]])   
    #computes the state of each node at each step of the simulation    
    for t in range(tMax):
            for n in G:
                done=False
                if G.nodes[n]['state'][t]==u:
                    for m in G[n]:
                        if done:
                            break
                        if G.nodes[m]['state'][t]==a:
                            if random.random()<p:
                                G.nodes[n]['state'].append(a)
                                nodesInState[a]+=1
                                done=True
                if not done:
                    G.nodes[n]['state'].append(G.nodes[n]['state'][t])
                else:
                    if G.nodes[n]['state'][t]==u:
                        nodesInState[u]-=1
            data.append([nodesInState[u],nodesInState[a],nodesInState[b]])
    return data
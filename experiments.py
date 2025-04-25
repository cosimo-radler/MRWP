import engine as en
import powerlaw
import random 
import networkx as nx 

#generate a random list of sampleSize nodes from G ignoring the nodes in the list exclude
def generateRandomSample(G,sampleSize,exclude):
    sample=[]
    nodelist=[n for n in G if n not in exclude]
    if len(nodelist)<sampleSize:
        return None
    while len(sample)<sampleSize: 
        newExtraction=nodelist[random.randint(0,len(nodelist)-1)] 
        if newExtraction not in sample:
            sample.append(newExtraction)   
    return sample

#generate a discrete random sample of the given size with powerlaw distribution of parameter gamma
def generatePowerLawSample(size,gamma):
    td=powerlaw.Power_Law(xmin=1,parameters=[gamma],discrete=True)
    sample=[1]
    while sum(sample)%2:
        sample=[int(n) for n in td.generate_random(size)]
    return sample

#generates a graph of the desired size with powerlaw distribution og parameter gamma 
def generatePowerLawGraph(size,gamma):
    degreeDistribution=generatePowerLawSample(size,gamma)
    return nx.configuration_model(degreeDistribution)
    
#runs the model on G with the specified probability  starting with randomly selected nodes for groupA and groupB 
def fullyRandomExperiment(G,tMax,probability,sizeGroupA,sizeGroupB):
    groupA=generateRandomSample(G,sizeGroupA,[])
    groupB=generateRandomSample(G,sizeGroupB,groupA)   
    return en.graphModelRun(G, tMax, groupA, groupB,probability)
    
#runs the model on G with the specified probability starting with sizeGroupB randomly selected nodes for groupB and with the specified nodes in groupA
def halfRandomExperimentA(G,tMax,probability,groupA,sizeGroupB):
    groupB=generateRandomSample(G,sizeGroupB,groupA)   
    return en.graphModelRun(G, tMax, groupA, groupB,probability)

#runs the model on G with the specified probability starting with sizeGroupA randomly selected nodes for groupA and with the specified nodes in groupB
def halfRandomExperimentB(G,tMax,probability,groupB,sizeGroupA):
    groupA=generateRandomSample(G,sizeGroupA,groupB)   
    return en.graphModelRun(G,tMax, groupA, groupB,probability)

#runs the function experiment with parameters G, tMax, probability, forthArgument, and fifthArgument for experimentsNumber times and returns a dictionary of the results
def repeatedExperiments(experiment,G,tMax,probability,forthArgument,fifthArgument,experimentsNumber):
    data={}
    #run the experiments
    for n in range(experimentsNumber):
        data[n]=experiment(G,tMax,probability,forthArgument,fifthArgument)
        print("Experiment:", n, "done.")
    return data

#takes a dictionary experimentDataDict as the one produced by repeatedExperiments and computes a list which contains the average number of nodes in state u, a, and b at each time step 
def averageExperiment(experimentDataDict,tMax):
    numberofStates=3
    experimentsNumber=len(experimentDataDict)
    return [[sum([experimentDataDict[n][t][i] for n in range(experimentsNumber)])/experimentsNumber for i in range(numberofStates)] for t in range(tMax)]

import networkx as nx 
import visualizer as viz
import experiments as exp
import numpy as np
import matplotlib.pyplot as plt


N=1000
tMax=50
nExp=20  # Reduced for faster execution
infected=5  # Initial infected nodes
vaccinated=5  # Initial vaccinated nodes
gamma=2.5
probabilityOfTransmission=0.5


# Generate network structures
print("Generating networks...")
# Random graph with 1000 nodes
rG=nx.erdos_renyi_graph(N,1.5/N)
# Power-law (scale-free) graph with 1000 nodes 
plG=exp.generatePowerLawGraph(N,gamma)
# Convert to simple graph (no parallel edges or self-loops)
plG = nx.Graph(plG)
print(f"Scale-free network: {plG.number_of_nodes()} nodes, {plG.number_of_edges()} edges")


# Define selective vaccination functions for different centrality measures

# Betweenness centrality vaccination
def betweennessVaccinationExperiment(G, tMax, beta, initialInfected, numVaccinated):
    # Calculate betweenness centrality for all nodes
    print("Calculating betweenness centrality (this may take a moment)...")
    betweenness_dict = nx.betweenness_centrality(G)
    
    # Get nodes sorted by betweenness centrality (highest first)
    sorted_nodes = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)
    high_betweenness_nodes = [node for node, _ in sorted_nodes[:numVaccinated]]
    
    # Use the halfRandomExperimentB function to run experiment with predefined vaccination nodes
    return exp.halfRandomExperimentB(G, tMax, beta, high_betweenness_nodes, initialInfected)

# Degree centrality vaccination
def degreeVaccinationExperiment(G, tMax, beta, initialInfected, numVaccinated):
    # Get nodes sorted by degree (highest first)
    degree_dict = dict(G.degree())
    sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    high_degree_nodes = [node for node, _ in sorted_nodes[:numVaccinated]]
    
    # Use the halfRandomExperimentB function to run experiment with predefined vaccination nodes
    return exp.halfRandomExperimentB(G, tMax, beta, high_degree_nodes, initialInfected)


# Run comparative experiments
print("\n=== VACCINATION STRATEGY COMPARISON EXPERIMENT ===")
print("Running experiments with 5 randomly vaccinated individuals vs. selectively vaccinated individuals")
print("(Testing random, degree, and betweenness centrality vaccination strategies)")

# RANDOM GRAPH EXPERIMENTS
print("\n--- Random Network Results ---")
# Random vaccination on random graph
print("Running random vaccination on random graph...")
random_graph_random_vax = exp.repeatedExperiments(exp.fullyRandomExperiment, 
                                       rG, 
                                       tMax, 
                                       probabilityOfTransmission, 
                                       infected, 
                                       vaccinated, 
                                       nExp)

# Degree centrality vaccination on random graph
print("Running degree centrality vaccination on random graph...")
random_graph_degree_vax = exp.repeatedExperiments(degreeVaccinationExperiment, 
                                       rG, 
                                       tMax, 
                                       probabilityOfTransmission, 
                                       infected, 
                                       vaccinated, 
                                       nExp)

# Betweenness vaccination on random graph
print("Running betweenness centrality vaccination on random graph...")
random_graph_betweenness_vax = exp.repeatedExperiments(betweennessVaccinationExperiment, 
                                          rG, 
                                          tMax, 
                                          probabilityOfTransmission, 
                                          infected, 
                                          vaccinated, 
                                          nExp)

# SCALE-FREE GRAPH EXPERIMENTS
print("\n--- Scale-Free Network Results ---")
# Random vaccination on scale-free graph
print("Running random vaccination on scale-free graph...")
sf_graph_random_vax = exp.repeatedExperiments(exp.fullyRandomExperiment, 
                                     plG, 
                                     tMax, 
                                     probabilityOfTransmission, 
                                     infected, 
                                     vaccinated, 
                                     nExp)

# Degree centrality vaccination on scale-free graph
print("Running degree centrality vaccination on scale-free graph...")
sf_graph_degree_vax = exp.repeatedExperiments(degreeVaccinationExperiment, 
                                     plG, 
                                     tMax, 
                                     probabilityOfTransmission, 
                                     infected, 
                                     vaccinated, 
                                     nExp)

# Betweenness vaccination on scale-free graph
print("Running betweenness centrality vaccination on scale-free graph...")
sf_graph_betweenness_vax = exp.repeatedExperiments(betweennessVaccinationExperiment, 
                                        plG, 
                                        tMax, 
                                        probabilityOfTransmission, 
                                        infected, 
                                        vaccinated, 
                                        nExp)

# Average the results
print("\nAveraging results...")
random_random_avg = exp.averageExperiment(random_graph_random_vax, tMax)
random_degree_avg = exp.averageExperiment(random_graph_degree_vax, tMax)
random_betweenness_avg = exp.averageExperiment(random_graph_betweenness_vax, tMax)

sf_random_avg = exp.averageExperiment(sf_graph_random_vax, tMax)
sf_degree_avg = exp.averageExperiment(sf_graph_degree_vax, tMax)
sf_betweenness_avg = exp.averageExperiment(sf_graph_betweenness_vax, tMax)

# Print numerical snapshot data
def print_snapshot(data, label, timesteps=[0, 25, 49]):
    print(f"\n{label} - Network States at Key Timepoints:")
    for t in timesteps:
        susceptible = data[t][0]
        infected = data[t][1]
        vaccinated = data[t][2]
        print(f"Time {t}: S={susceptible:.1f}, I={infected:.1f}, V={vaccinated:.1f}")

print("\n=== NUMERICAL RESULTS ===")
print_snapshot(random_random_avg, "Random Network - Random Vaccination")
print_snapshot(random_degree_avg, "Random Network - Degree Centrality Vaccination")
print_snapshot(random_betweenness_avg, "Random Network - Betweenness Centrality Vaccination")

print_snapshot(sf_random_avg, "Scale-Free Network - Random Vaccination")
print_snapshot(sf_degree_avg, "Scale-Free Network - Degree Centrality Vaccination")
print_snapshot(sf_betweenness_avg, "Scale-Free Network - Betweenness Centrality Vaccination")

# Create a comparative figure for all strategies
print("\nCreating comparative figure of all vaccination strategies...")

# Setup figure with 2 rows (one for each network type)
plt.figure(figsize=(14, 10))

# Define time array
time = list(range(tMax))

# Colors for each strategy
colors = {
    'random': 'lightgray',
    'degree': 'orange',
    'betweenness': 'green'
}

# Row 1: Random Network - Infected Population
plt.subplot(2, 2, 1)
plt.plot(time, [d[1] for d in random_random_avg], color=colors['random'], label="Random Vaccination")
plt.plot(time, [d[1] for d in random_degree_avg], color=colors['degree'], label="Degree Centrality")
plt.plot(time, [d[1] for d in random_betweenness_avg], color=colors['betweenness'], label="Betweenness Centrality")
plt.title("Random Network: Infected Population")
plt.xlabel("Time Steps")
plt.ylabel("Number of Infected Nodes")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Row 1: Random Network - Susceptible Population
plt.subplot(2, 2, 2)
plt.plot(time, [d[0] for d in random_random_avg], color=colors['random'], label="Random Vaccination")
plt.plot(time, [d[0] for d in random_degree_avg], color=colors['degree'], label="Degree Centrality")
plt.plot(time, [d[0] for d in random_betweenness_avg], color=colors['betweenness'], label="Betweenness Centrality")
plt.title("Random Network: Susceptible Population")
plt.xlabel("Time Steps")
plt.ylabel("Number of Susceptible Nodes")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Row 2: Scale-Free Network - Infected Population
plt.subplot(2, 2, 3)
plt.plot(time, [d[1] for d in sf_random_avg], color=colors['random'], label="Random Vaccination")
plt.plot(time, [d[1] for d in sf_degree_avg], color=colors['degree'], label="Degree Centrality")
plt.plot(time, [d[1] for d in sf_betweenness_avg], color=colors['betweenness'], label="Betweenness Centrality")
plt.title("Scale-Free Network: Infected Population")
plt.xlabel("Time Steps")
plt.ylabel("Number of Infected Nodes")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Row 2: Scale-Free Network - Susceptible Population
plt.subplot(2, 2, 4)
plt.plot(time, [d[0] for d in sf_random_avg], color=colors['random'], label="Random Vaccination")
plt.plot(time, [d[0] for d in sf_degree_avg], color=colors['degree'], label="Degree Centrality")
plt.plot(time, [d[0] for d in sf_betweenness_avg], color=colors['betweenness'], label="Betweenness Centrality")
plt.title("Scale-Free Network: Susceptible Population")
plt.xlabel("Time Steps")
plt.ylabel("Number of Susceptible Nodes")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Vaccination_Strategies_Comparison.png", dpi=300)
plt.show()

# Additionally, create a bar chart of final infection rates
print("Creating bar chart of final infection rates...")
plt.figure(figsize=(12, 7))

# Extract final infection counts
final_time = 49
networks = ["Random Network", "Scale-Free Network"]
strategies = ["Random", "Degree", "Betweenness"]

random_net_infections = [
    random_random_avg[final_time][1],
    random_degree_avg[final_time][1],
    random_betweenness_avg[final_time][1]
]

sf_net_infections = [
    sf_random_avg[final_time][1],
    sf_degree_avg[final_time][1],
    sf_betweenness_avg[final_time][1]
]

# Position the bars
x = np.arange(len(strategies))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, random_net_infections, width, label='Random Network', color='steelblue')
rects2 = ax.bar(x + width/2, sf_net_infections, width, label='Scale-Free Network', color='firebrick')

# Add labels and title
ax.set_ylabel('Number of Infected Nodes')
ax.set_title('Final Infection Count by Network Type and Vaccination Strategy')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.savefig("Final_Infection_Comparison.png", dpi=300)
plt.show()

# Visualize the data for each strategy separately
print("\nVisualizing individual results...")
viz.showData(random_random_avg, "Random_Network_Random_Vaccination")
viz.showData(random_degree_avg, "Random_Network_Degree_Vaccination")
viz.showData(random_betweenness_avg, "Random_Network_Betweenness_Vaccination")

viz.showData(sf_random_avg, "ScaleFree_Network_Random_Vaccination")
viz.showData(sf_degree_avg, "ScaleFree_Network_Degree_Vaccination")
viz.showData(sf_betweenness_avg, "ScaleFree_Network_Betweenness_Vaccination")





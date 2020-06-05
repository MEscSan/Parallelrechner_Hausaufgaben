# Theoretical Questions to MPI
## Data-Distribution Topologies

### Ring topology
Every node in a network is connected only to two other nodes, forming a network consisting of a single continuous path. Data can travel between nodes either clockwise or anticlockwise (unidirectional ring) or in both directions (dual-ring network) 

#### Advantages
* No central node for connectivity management required
* Faults can be easy identified due to point-to-point communication
* Works better than a bus-topology under heavy load

#### Disadvantage
* One failing node can create problems for the entire network
* Adding or removing devices can affect the network
* Delay directly proportional to the number of nodes

### Star topology
Every node (host) in the network is connected only to a central-node (hub), managing the network.

#### Advantages
* One failing node (or its connection) does not affect to the rest of the network
* Adding or removing devices does not disturb the network
* Works well under heavy load

#### Disadvantages
* Central node required, single point of failure for the network
* Higher material costs required to connect every host with the hub

### Bus topology
Every node (station) is connected to a common link (bus). Every station receives all the traffic and all stations have the same transmision priority. A medium access control technology is used to manage the communication through the bus, such a media access control protocol or a bus master.
#### Advantages 
* One failing node does not affect the rest of the network
* Easy to extend without network disruption  
* Lower material costs than a star-network
#### Disadvantages
* Faulst are difficult to identify and isolate
* Performance degrades with many nodes on the network since bandwidth shared by all nodes
* A failure in the bus may affect the whole network

### Switched fabric topology
Nodes interconnect via one or more switches and routers instead of being directly connected to each other as in a point-to-point network
#### Advantages
* Redundant communication and therefor robust towards network failure
* High bandwith
* High throughput 
#### Disadvanteg
* Higher complexity (and material costs) as in a simpler topology

Source: Wikipedia

## Infiniband vs Ethernet in high performance computing
Comparison of Interconnection-Families Infiniband and Ethernet (Gigabit-Ethernet)

#### Top500 System Share:
* Infininband 24.6%
* Gigabit-Ethernet: 54.4% 

#### Top500 Performance Share:
* Infiniband:   37.9%
* Gigabit-Ethernet: 25.4%

#### Throughput (Gb/s)
* Infiniband(1 Link, HDR):  50 Gb/s
* Infiniband(8 Link, HDR; used in clusters): 4000 Gb/s
* Gigabit-Ethernet:  100Gb/s (100GbE)   

#### Cable-Lenght 
* Infiniband: Up to 10m (copper) or up to 10km (optical fiber)
* Gigabit-Ethernet: Up to 5m (100 GbE, 2nd Gen., 100GBASE-CR4, twinaxial balanced cable), over 4km (100GBASE
-SR2-BiDirectional, OM4 connection )

# Sources:
* Wikipedia
* Top500

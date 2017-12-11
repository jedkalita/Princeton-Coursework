#ifndef TRAV_H_
#define COLORING_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "minisat/core/Solver.h"

using namespace std;

// ***************************************************************************
// A graph class.
// Requires nodes, edges, distances between the edges.
// Note that when adding an edge (n1,n2) n1 must be less or                    
// equal to n2. This is only done for simplicity and a more compact            
// implementation.
// ***************************************************************************
class Graph {
 public:
 Graph(int nNumberOfNodes) : m_nNumberOfNodes(nNumberOfNodes)
  {
    m_graph.resize(nNumberOfNodes);
  }

  int getNumberOfNodes() const { return m_nNumberOfNodes; }

  vector<int> getEdgesForNode(int node) const
    {
      assert (node < m_nNumberOfNodes);
      assert (node < m_graph.size());
      return m_graph[node];
    }

  void addEdge (int n1, int n2)
  {
    assert (n1 < m_nNumberOfNodes &&
	    n2 < m_nNumberOfNodes);
    assert (n1 <= n2);
    
    // Make sure that the vector can contain the first node                 
    if (m_graph.size() <= n1)
      {m_graph.resize(n1+1);}
    //now add the edge wrt n1
    m_graph[n1].push_back(n2);
    // Make sure that the vector can contain the second node                 
    if (m_graph.size() <= n2)
      {m_graph.resize(n2+1);}
    //now add the edge wrt n2                                                  
    m_graph[n2].push_back(n1);
  }

 private:
  int m_nNumberOfNodes;
  // A vector of vectors to represent the adjacency list                      
  // The outer vector is mapping a node (by index) to its                     
  // vector which represents a container of adjacent nodes.
  vector<vector<int> > m_graph;
};

// *************************************************************************** 
// A class modeling the traveling salesman problem.                           
// ***************************************************************************
class Traveling {
 private:
  int startNode;
  const Graph& m_graph;
  Minisat::Solver m_solver;

 public:
 Traveling(const Graph& g, int startingNode) :
  m_graph(g), startNode(startingNode), m_solver()
    {
      //prepare the solver with the needed variables
      int nodes = m_graph.getNumberOfNodes();
      for (int i = 0; i <= nodes; i++) //basically one number from 0 to n-1 assigned to each node indicating the number at which it is visited. An extra one added since the starting node will also be visited at the very end since it ends there.
	{
	  for (int j = 0; j < nodes; j++)
	    {
	      Minisat::Var v = m_solver.newVar(); //Row -- which hop no., Col -- which node no. Element(i,j) - At the ith hop, node j is being visited
	      //printf("%i\n", v);
	    }
	}
    }
 
  //condition that the starting node has to be the one specified
  void addStartingConstraint() {
    assert (startNode < m_graph.getNumberOfNodes());
    Minisat::vec<Minisat::Lit> clause;
    Minisat::Var v = 0 * m_graph.getNumberOfNodes() + startNode; //the first hop has to be the starting node
    //printf("starting node constraint : %i\n", v);
    clause.push(Minisat::mkLit(v, false)); //has to be true
    m_solver.addClause(clause);
  }

  //condition that the ending node has to be the one specified
  void addEndingConstraint() {
    assert (startNode < m_graph.getNumberOfNodes());
    Minisat::vec<Minisat::Lit> clause;
    Minisat::Var v = m_graph.getNumberOfNodes() * m_graph.getNumberOfNodes() + startNode; //the last hop has to be the starting node 
    //printf("ending node constraint : %i\n", v);
    clause.push(Minisat::mkLit(v, false)); //has to be true                    
    m_solver.addClause(clause);
  }

  //condition that says that except for the first node (starting node), all the other nodes must have a color and all the nodes must be visited exactly once
  void addVisitConstraints(int node) {
    assert (node < m_graph.getNumberOfNodes());
    Minisat::vec<Minisat::Lit> clause;
    int nodes = m_graph.getNumberOfNodes();
    for (int i = 0; i <= nodes; i++)
      {
	Minisat::Var v = i * m_graph.getNumberOfNodes() + node;
	//printf("visit constraints 1 : %i\n", v);
	clause.push(Minisat::mkLit(v, false));
      }
    m_solver.addClause(clause);
    //for each node, now make sure only one hop happens there, except for the last hop and the starting node
    for (int i = 0; i < nodes; i++)
      {
	for (int k = i + 1; k <= nodes; k++)
	  {
	    if (i == 0 && k == nodes && node == startNode)
	      {
		continue; //the only time when two visits to the same node allowed
	      }
	    else
	      {
		//printf("Visit constraints 2 clause : i -  %i k -  %i \n", i * m_graph.getNumberOfNodes() + node, k * m_graph.getNumberOfNodes() + node);
		m_solver.addClause(Minisat::mkLit(i * m_graph.getNumberOfNodes() + node, true), Minisat::mkLit(k * m_graph.getNumberOfNodes() + node, true));
	      }
	  }
      }
  }

  void addPositionConstraints(int position) {
    assert (position <= m_graph.getNumberOfNodes());
    Minisat::vec<Minisat::Lit> clause;
    int nodes = m_graph.getNumberOfNodes();
    for (int i = 0; i <= nodes; i++)
      {
	Minisat::Var v = position * m_graph.getNumberOfNodes() + i;
        clause.push(Minisat::mkLit(v, false));
      }
    m_solver.addClause(clause);
    //for each node, now make sure only one hop happens there, except for the last hop and the starting node                    
    for (int i = 0; i < nodes - 1; i++)
      {
        for (int k = i + 1; k < nodes; k++)
	  {  
	    m_solver.addClause(Minisat::mkLit(position * m_graph.getNumberOfNodes() + i, true), Minisat::mkLit(position * m_graph.getNumberOfNodes() + k, true));
	  }    
      }
  }


  void noAdjacentNodesConstraint() {
    int nodes = m_graph.getNumberOfNodes();
    // generate a neighbor matrix where neigbs[i][j] = 1 if nodes i and j are neighbors, and 0 if not
    //int neighbs[nodes][nodes] = {0};
    int **neighbs = new int* [nodes];
    for (int i = 0; i < nodes; i++)
      {
	neighbs[i] = new int[nodes];
      }
    for (int i = 0; i < nodes; i++)
      {
	for (int j = 0; j < nodes; j++)
	  {
	    neighbs[i][j] = 0;
	  }
      }
    for (int i = 0; i < nodes; i++)
      {
	vector<int> edges = m_graph.getEdgesForNode(i);
	for (int adjacent = 0; adjacent < edges.size(); adjacent++){
	  neighbs[i][edges[adjacent]] = 1;
	}
      }
    // go through each level of the path, and in that level if two nodes j and are not neighbors, then if j  occurs in the ith hop, k cannot occur in the i + 1th hop -- thus not i_j or not i+1_k must be true.
    for (int i = 0; i <= nodes - 1; i++)
      {
	for (int j = 0; j < nodes; j++)
	  {
	    for (int k = 0; k < nodes; k++)
	      {
                    
		if (j != k)
		  {
		    // if j and k are not neighbors, then they cannot occur in subsequent positions of hte graph
		    if (neighbs[j][k] == 0){
		      m_solver.addClause(Minisat::mkLit(i * m_graph.getNumberOfNodes() + j, true), Minisat::mkLit((i + 1) * m_graph.getNumberOfNodes() + k, true));
		    }
		  }
	      }
	  }
      }
  }

  bool isTravelable()
  {
    // Go over all nodes                                                    
    for (int n = 0; n < m_graph.getNumberOfNodes(); n++)
      {
	// Add the constraints for the node                                 
	addVisitConstraints(n);
      }

    // Go over all hops                                                                                                        
    for (int n = 0; n <= m_graph.getNumberOfNodes(); n++)
      {
        // Add the constraints for the hope                                                                                     
        addPositionConstraints(n);
      }
    addStartingConstraint();
    addEndingConstraint();

    noAdjacentNodesConstraint();

    bool bResult = m_solver.solve();

    //try and write all clauses to a file                                   
    //ofstream myFile;                                                      
    //myFile.open("clauses.txt");                                           
    /*const char* myFile;                                                   
        myFile = "clauses.txt";                                                 
        m_solver.toDimacs(myFile);*/
    //myFile.close();                                                       

    return bResult;
  }

  void givemeAllPaths(vector<vector<Minisat::lbool> >& allPaths) {
    // Go over all nodes
    for (int n = 0; n < m_graph.getNumberOfNodes(); n++)
      {
        // Add the constraints for the node                                                      
        addVisitConstraints(n);
      }

    // Go over all hops                                                                                                                                                                    
    for (int n = 0; n <= m_graph.getNumberOfNodes(); n++)
      {
        // Add the constraints for the hope                                                     
	addPositionConstraints(n);
      }
    addStartingConstraint();
    addEndingConstraint();

    noAdjacentNodesConstraint();

    bool result = m_solver.solve();

    while (result)
      {
	vector<Minisat::lbool> v;
	Minisat::vec<Minisat::Lit> clause;
	for (int i = 0; i < m_solver.model.size(); i++)
	  {
	    v.push_back(m_solver.model[i]);
	    if(m_solver.modelValue(i) == Minisat::l_True)
	      {
		clause.push(Minisat::mkLit(i, true));
	      }
	    else
	      {
		clause.push(Minisat::mkLit(i, false)); //yakir told me not to worry about undef variables in minisat                                       
	      }
	  }
	allPaths.push_back(v);
	m_solver.addClause(clause);
	result = m_solver.solve();
      }
  }

};

#endif


  

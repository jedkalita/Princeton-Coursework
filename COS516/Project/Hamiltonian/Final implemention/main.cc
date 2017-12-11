#include "trav.h"

// ***************************************************************************  
// A main file that uses the Traveling and Graph classes.                        
// Note that you can create your own graphs and test their Hamiltonian prop.                                                       
// ***************************************************************************  
int main()
{
  printf("\n====================================================\n\n");
  printf("The Hamiltonian Cycle problem\n");
  printf("\n====================================================\n\n\n");

  //int k = 5; // The starting node                                        
  int k = 3;
  // Create a graph with 3 nodes. 
  //Graph g(11);
  //Graph g(6);                                                               
  Graph g(5);
  // Add edges to the graph - this is a simple triangle.                      
  /*                                                                         
  g.addEdge(0,3);
  g.addEdge(0,4);
  g.addEdge(1,3);
  g.addEdge(1,2);
  g.addEdge(2,4);
  g.addEdge(0,4);
  g.addEdge(3,5);
  g.addEdge(2, 5);
  g.addEdge(1,4);
  */
  /*
  g.addEdge(0,1);
  g.addEdge(1,2);
  g.addEdge(2,3);
  g.addEdge(3,4);
  g.addEdge(4,5);
  g.addEdge(5,6);
  g.addEdge(6,7);
  g.addEdge(0,7);
  g.addEdge(0,8);
  g.addEdge(6,8);
  g.addEdge(4,8);
  g.addEdge(7,9);
  g.addEdge(1,9);
  g.addEdge(3,9);
  g.addEdge(5,9);
  g.addEdge(0,10);
  g.addEdge(2,10);
  g.addEdge(4,10);
  */
 
  g.addEdge(0,1);
  g.addEdge(1,2);
  g.addEdge(2,3);
  g.addEdge(3,4);
  g.addEdge(0,4);
  g.addEdge(0,2);
  g.addEdge(1,3);
  g.addEdge(1,4);
  
  // Model the problem for k.                                                 
  Traveling t(g, k);

  // Now ask if it is colorable                                               
  bool bResult = t.isTravelable();

  if (bResult) {
    printf("\n\tThe graph has a Hamiltonian Cycle from the starting node.\n\n");
  }
  else {
    printf("\n\tThe graph has no Hamiltonian Cycle from the starting node.\n\n");
  }


  //debugging for the last function                                           
  cout<<"\n\n\n\nNOW EXPLICITLY DEBUGGING FOR LAST FUNCTION\n\n\n";
  vector<vector<Minisat::lbool> > allPaths;
  t.givemeAllPaths(allPaths);
  cout<<"Size of allPaths vector within main: "<< allPaths.size();
  cout<<endl;
  /*for (int i = 0; i < allColoring.size(); i++)                              
      {                                                                         
        for (int j = 0; j < allColoring[i].size(); j++)                         
          {                                                                     
	  cout<<allColoring[i][j];                                            
          }                                                                     
	  cout<<endl;                                                             
	  }*/


  for(int i = 0; i <allPaths.size(); i++)
    {
      for (int j = 0;j < allPaths[i].size(); j++)
	{
	  //cout<<allColoring[i][j];                                         \
                                                                                
	  printf("%d", allPaths[i][j]);
	}
      cout<<endl;
      cout<<i<<" size = "<<allPaths[i].size();
      cout<<endl;
    }




  return 0;
}

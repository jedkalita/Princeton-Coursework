#include "coloring.h"

// ***************************************************************************
// A main file that uses the Coloring and Graph classes.
// Note that you can create your own graphs and test their k-colorability for
// different values of k.
// ***************************************************************************
int main()
{
    printf("\n====================================================\n\n");
    printf("The k-coloring problem\n");
    printf("\n====================================================\n\n\n");

    //int k = 2; // The number of colors
    int k = 4;
    // Create a graph with 3 nodes.
    //Graph g(3);
    Graph g(5);
    // Add edges to the graph - this is a simple triangle.
    /*
    g.addEdge(0,1);
    g.addEdge(0,2);
    g.addEdge(1,2);
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
    Coloring c(g, k);

    // Now ask if it is colorable
    bool bResult = c.isColorable();

    if (bResult) {
        printf("\n\tThe graph is %d-colorable.\n\n", k);
    }
    else {
        printf("\n\tNo %d-coloring is found!\n\n", k);
    }

    //debugging for the last function
    //cout<<"\n\n\n\nNOW EXPLICITLY DEBUGGING FOR LAST FUNCTION\n\n\n";
    vector<vector<Minisat::lbool> > allColoring;
    c.givemeAllColoring(allColoring);
    //cout<<"Size of allColoring vector within main: "<< allColoring.size();
    //cout<<endl;
    /*for (int i = 0; i < allColoring.size(); i++)
      {
	for (int j = 0; j < allColoring[i].size(); j++)
	  {
	    cout<<allColoring[i][j];
	  }
	cout<<endl;
      }*/

    /*
    for(int i = 0; i <allColoring.size(); i++)
      {
	for (int j = 0;j < allColoring[i].size(); j++)
	  {
	    //cout<<allColoring[i][j];                                                
	    printf("%d", allColoring[i][j]);
	  }
	cout<<endl;
	cout<<i<<" size = "<<allColoring[i].size();
	cout<<endl;
	}*/


    return 0;
}

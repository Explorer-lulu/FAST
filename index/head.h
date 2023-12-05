#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <boost/thread/thread.hpp>
#include <functional>
#include <bits/stdc++.h>
#include <utility>


#define INF 999999999
using namespace std;

struct Nei{            
	int nid;
	int w;
	int c;
};

struct tri{
	int u;
	int v;
	int w;
};

struct Node{ //Tree Decomposition Node
	vector<pair<int,pair<int,int>>> vert;
	vector<pair<int,Nei>> neighInf;
	vector<int> pos, pos2;
	vector<int> dis, cnt;
	set<int> changedPos;
	vector<bool> FN;
	set<int> DisRe;
	vector<int> ch;   
	int height, hdepth;
	int pa;
	int uniqueVertex;
	vector<int> piv;
	Node(){
		vert.clear();
		neighInf.clear();
		pos.clear();
		dis.clear();
		cnt.clear();
		ch.clear();
		pa = -1;
		uniqueVertex = -1;
		height = 0;
		hdepth = 0;
		changedPos.clear();
		FN.clear();
		DisRe.clear();
		piv.clear();
	}
};

class FRNGraph{
public:
	int nodenum;
	int edgenum;

	int threadnum=40;

	int eNum;
	vector<pair<int,int>> Edge;
	unordered_map<pair<int,int>, int, hash_pair> EdgeRe;

	vector<int> NodeOrder;
	vector<int> vNodeOrder;

	void insertEMTOrderGenerate(int u,int v,int w);
	void NeighborComOrderGenerate(vector<pair<int,pair<int,int>>> &Neighvec, pair<int,int> p, int x);
	void NoRegularFRNGraphOrder();
	void ReadFRNGraph(string filename);
	void ReadFRNGraphForCHPinc(string filename);
	void ReadIns(string filename, vector<pair<pair<int,int>,int>>& data);
	void CorCheckFHLI();
	void EffiCheckFHLI(string filename);
	int QueryFHLI(int ID1,int ID2);

	long long FHLIindexsize();	
	void FHLIconOrderMT(string orderfile);
	void FHLIdecBat(vector<pair<pair<int,int>,pair<int,int>>>& wBatch);
	void FHLIincBatMT(vector<pair<pair<int,int>,pair<int,int>>>& wBatch);
	vector<int> rank;
	vector<Node> Tree;
	int heightMax;
	void buildTree();
	int match(int x,vector<pair<int,pair<int,int>>> &vert);
	vector<vector<int>> VidtoTNid;
	vector<int> EulerSeq;
	vector<int> toRMQ;
	vector<vector<int>> RMQIndex;
    void degreeFlowJoint(float &beta);
	void buildRMQDFS(int p, int height);
	void buildRMQ();
	int LCAQuery(int _p, int _q);
	void buildIndex();
	void buildIndexDFS(int p, vector<int> &list);
	void EachNodeProBDis5(int child,vector<int>& line,set<int>& vertexIDChL, map<int,int>& checkedDis);
	void eachNodeProcessIncrease1(int children, vector<int>& line, int& changelabel);
	vector<int> PathRetriFHLI(int s, int t);
	vector<int> PathRetriFHLIpartial(int ID1, int ID2);
	void FHLIconPath(string orderfile);
	void buildIndexPath();
	void buildIndexDFSPath(int p, vector<int>& list);
};
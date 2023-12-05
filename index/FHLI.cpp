#include "head.h"

void FRNGraph::FHLIconOrderMT(string orderfile){
	UPdateconsorder(orderfile);
	buildTree();
	buildIndex();
}

void FRNGraph::buildTree(){
	vector<int> vecemp; 
	VidtoTNid.assign(nodenum,vecemp);

	rank.assign(nodenum,0);
	int len=vNodeOrder.size()-1; 
	heightMax=0;

	Node rootn;
	int x=vNodeOrder[len]; 

	while(x==-1){
		len--;
		x=vNodeOrder[len];
	}
	rootn.vert=NeighborCon[x];
	rootn.uniqueVertex=x;
	rootn.pa=-1;
	rootn.height=1;
	rank[x]=0;
	Tree.push_back(rootn);
	len--;

	int nn;
	for(;len>=0;len--){
		int x=vNodeOrder[len];
		Node nod;
		nod.vert=NeighborCon[x];
		nod.uniqueVertex=x;
		int pa=match(x,NeighborCon[x]);
		Tree[pa].ch.push_back(Tree.size());
		nod.pa=pa;
		nod.height=Tree[pa].height+1;

		nod.hdepth=Tree[pa].height+1;
		for(int i=0;i<NeighborCon[x].size();i++){
			nn=NeighborCon[x][i].first;
			VidtoTNid[nn].push_back(Tree.size());
			if(Tree[rank[nn]].hdepth<Tree[pa].height+1)
				Tree[rank[nn]].hdepth=Tree[pa].height+1;
		}
		if(nod.height>heightMax) heightMax=nod.height;
		rank[x]=Tree.size();
		Tree.push_back(nod);
	}
}

void FRNGraph::FHLIdecBat(vector<pair<pair<int,int>,pair<int,int>>>& wBatch){
	map<int,int> checkedDis;

	for(int i=0;i<Tree.size();i++){
		Tree[i].DisRe.clear();
	}

	NodeOrderss.assign(NodeOrder.begin(),NodeOrder.end());
	vector<set<int>> SCre; 
	set<int> ss; 
	SCre.assign(nodenum,ss);
	set<OrderCompp> OC;

	set<int> vertexIDChL; 

	int a,b,oldW,newW,lid,hid;
	for(int k=0;k<wBatch.size();k++){
		a=wBatch[k].first.first; b=wBatch[k].first.second; oldW=wBatch[k].second.first;newW=wBatch[k].second.second;
		if(NodeOrder[a]<NodeOrder[b]){
			lid=a;hid=b;
		}else{
			lid=b;hid=a;
		}

		for(int i=0;i<Neighbor[a].size();i++){
			if(Neighbor[a][i].first==b){
				Neighbor[a][i].second=newW;
				break;
			}
		}
		for(int i=0;i<Neighbor[b].size();i++){
			if(Neighbor[b][i].first==a){
				Neighbor[b][i].second=newW;
				break;
			}
		}

		for(int i=0;i<Tree[rank[lid]].vert.size();i++){
			if(Tree[rank[lid]].vert[i].first==hid){
				if(Tree[rank[lid]].vert[i].second.first>newW){
					Tree[rank[lid]].vert[i].second.first=newW;
					Tree[rank[lid]].vert[i].second.second=1;
					SCre[lid].insert(hid);
					OC.insert(OrderCompp(lid));
				}else if(Tree[rank[lid]].vert[i].second.first==newW){
					Tree[rank[lid]].vert[i].second.second+=1;
				}
				break;
			}
		}

	}

	vector<int> StartPointVertexSet; 
	vector<int> StartPointVertexSetNew;
	int StartPointVertexID;
	int ProID;
	while(!OC.empty()){
		ProID=(*OC.begin()).x;
		OC.erase(OC.begin());
		vector<pair<int,pair<int,int>>> Vert=Tree[rank[ProID]].vert;
		bool ProIDdisCha=false;
		for(auto it=SCre[ProID].begin();it!=SCre[ProID].end();it++){
			int Cid=*it; int Cw;
			int cidH=Tree[rank[Cid]].height-1;

			map<int,int> Hnei;
			vector<pair<int,int>> Lnei; 
			for(int j=0;j<Vert.size();j++){
				if(NodeOrder[Vert[j].first]>NodeOrder[Cid]){
					Hnei[Vert[j].first]=Vert[j].second.first;
				}else if(NodeOrder[Vert[j].first]<NodeOrder[Cid]){
					Lnei.push_back(build_pair(Vert[j].first,Vert[j].second.first));
				}else{
					Cw=Vert[j].second.first;
				}
			}

			if(Tree[rank[ProID]].dis[cidH]>Cw){
				Tree[rank[ProID]].dis[cidH]=Cw;
				Tree[rank[ProID]].FN[cidH]=true;
				ProIDdisCha=true;
				Tree[rank[ProID]].DisRe.insert(Cid);
			}else if(Tree[rank[ProID]].dis[cidH]==Cw){
				Tree[rank[ProID]].FN[cidH]=true;
			}

			int hid,hidHeight,lid,lidHeight,wsum;
			for(int j=0;j<Tree[rank[Cid]].vert.size();j++){
				hid=Tree[rank[Cid]].vert[j].first;hidHeight=Tree[rank[hid]].height-1;
				if(Hnei.find(hid)!=Hnei.end()){
					wsum=Cw+Hnei[hid];
					if(wsum<Tree[rank[Cid]].vert[j].second.first){
						Tree[rank[Cid]].vert[j].second.first=wsum;
						Tree[rank[Cid]].vert[j].second.second=1;
						SCre[Cid].insert(hid);
						OC.insert(OrderCompp(Cid));
					}else if(wsum==Tree[rank[Cid]].vert[j].second.first){
						Tree[rank[Cid]].vert[j].second.second+=1;
					}

				}
			}
			for(int j=0;j<Lnei.size();j++){
				lid=Lnei[j].first;lidHeight=Tree[rank[lid]].height-1;
				for(int k=0;k<Tree[rank[lid]].vert.size();k++){
					if(Tree[rank[lid]].vert[k].first==Cid){
						wsum=Cw+Lnei[j].second;
						if(Tree[rank[lid]].vert[k].second.first>wsum){
							Tree[rank[lid]].vert[k].second.first=wsum;
							Tree[rank[lid]].vert[k].second.second=1;
							SCre[lid].insert(Cid);
							OC.insert(OrderCompp(lid));
						}else if(Tree[rank[lid]].vert[k].second.first==wsum){
							Tree[rank[lid]].vert[k].second.second+=1;
						}

						break;
					}
				}
			}
		}

		if(ProIDdisCha){
			vertexIDChL.insert(ProID);
			StartPointVertexSetNew.clear(); StartPointVertexSetNew.reserve(StartPointVertexSet.size()+1);
			StartPointVertexSetNew.push_back(ProID);
			int rnew=rank[ProID],r;
			for(int i=0;i<StartPointVertexSet.size();i++){
				r=rank[StartPointVertexSet[i]];
				if(LCAQuery(rnew,r)!=rnew){
					StartPointVertexSetNew.push_back(StartPointVertexSet[i]);
				}
			}
			StartPointVertexSet=StartPointVertexSetNew;
		}
	}
	for(int i=0;i<StartPointVertexSet.size();i++){
		StartPointVertexID=StartPointVertexSet[i];
		vector<int> linee; 
		linee.reserve(heightMax);
		int pachidd=Tree[Tree[rank[StartPointVertexID]].pa].uniqueVertex;
		while(Tree[rank[pachidd]].height>1){
			linee.insert(linee.begin(),pachidd);
			pachidd=Tree[Tree[rank[pachidd]].pa].uniqueVertex;
		}
		linee.insert(linee.begin(),pachidd);
		EachNodeProBDis5(rank[StartPointVertexID], linee, vertexIDChL,checkedDis);
	}
}

void FRNGraph::buildIndex(){
	buildRMQ();
	vector<int> list; 
	list.push_back(Tree[0].uniqueVertex);
	Tree[0].pos.clear();
	Tree[0].pos.push_back(0);

	for(int i=0;i<Tree[0].ch.size();i++){
		buildIndexDFS(Tree[0].ch[i],list);
	}

}

int FRNGraph::match(int x,vector<pair<int,pair<int,int>>> &vert){
	int nearest=vert[0].first;
	for(int i=1;i<vert.size();i++){
		if(rank[vert[i].first]>rank[nearest])
			nearest=vert[i].first;
	}
	int p=rank[nearest];
	return p;
}

void FRNGraph::degreeFlowJoint(float &beta){
    int maxrecord = nodenum;
	int joint[maxrecord];
    for (int i = 0; i < Numvertex.size(); i++){
        joint[i] = beta * vdegree + (1 - beta) * vflow;
		if(Tree[p].vert[i].first==list[j]){
			Tree[p].pos[i]=j;
			Tree[p].dis[j]=Tree[p].vert[i].second.first;
			Tree[p].cnt[j]=1;
			break;
		}
	}
}

void FRNGraph::buildRMQDFS(int p, int height){
	toRMQ[p] = EulerSeq.size();
	EulerSeq.push_back(p);
	for (int i = 0; i < Tree[p].ch.size(); i++){
		buildRMQDFS(Tree[p].ch[i], height + 1);
		EulerSeq.push_back(p);
	}
}

int FRNGraph::LCAQuery(int _p, int _q){
	int p = toRMQ[_p], q = toRMQ[_q];
	if (p > q){
		int x = p;
		p = q;
		q = x;
	}
	int len = q - p + 1;
	int i = 1, k = 0;
	while (i * 2 < len){
		i *= 2;
		k++;
	}
	q = q - i + 1;
	if (Tree[RMQIndex[k][p]].height < Tree[RMQIndex[k][q]].height)
		return RMQIndex[k][p];
	else return RMQIndex[k][q];
}

void FRNGraph::buildIndexDFS(int p, vector<int>& list){
	int NeiNum=Tree[p].vert.size();
	Tree[p].pos.assign(NeiNum+1,0);
	Tree[p].dis.assign(list.size(),INF);
	Tree[p].cnt.assign(list.size(),0);
	Tree[p].FN.assign(list.size(),true);
	for(int i=0;i<NeiNum;i++){
		for(int j=0;j<list.size();j++){
			if(Tree[p].vert[i].first==list[j]){
				Tree[p].pos[i]=j;
				Tree[p].dis[j]=Tree[p].vert[i].second.first;
				Tree[p].cnt[j]=1;
				break;
			}
		}
	}
	Tree[p].pos[NeiNum]=list.size();


	for(int i=0;i<NeiNum;i++){
		int x=Tree[p].vert[i].first;
		int disvb=Tree[p].vert[i].second.first;
		int k=Tree[p].pos[i];

		for(int j=0;j<list.size();j++){
			int y=list[j];

			int z;
			if(k!=j){
				if(k<j)
					z=Tree[rank[y]].dis[k];
				else if(k>j)
					z=Tree[rank[x]].dis[j];

				if(Tree[p].dis[j]>z+disvb){
					Tree[p].dis[j]=z+disvb;
					Tree[p].FN[j]=false;
					Tree[p].cnt[j]=1;
				}else if(Tree[p].dis[j]==z+disvb){
					Tree[p].cnt[j]+=1;
				}
			}
		}
	}

	list.push_back(Tree[p].uniqueVertex);
	for(int i=0;i<Tree[p].ch.size();i++){
		buildIndexDFS(Tree[p].ch[i],list);
	}
	list.pop_back();
}

vector<int> NodeOrderss;
struct OrderCompp{
	int x;
	OrderCompp(int _x){
		x=_x;
	}
	bool operator< (const OrderCompp& d) const{
		if(x==d.x){
			return false;
		}else{
			if(x!=d.x)
				return NodeOrderss[x]<NodeOrderss[d.x];
		}
	}
};

void FRNGraph::eachNodeProcessIncrease1(int children, vector<int>& line, int& changelabel){
	int childID=Tree[children].uniqueVertex;
	int childH=Tree[children].height-1;
	for(int i=0;i<Tree[children].dis.size();i++){
		if(Tree[children].cnt[i]==0){
			changelabel+=1;
			int disBF=Tree[children].dis[i];
			int PID;
			for(int k=0;k<VidtoTNid[childID].size();k++){
				PID=VidtoTNid[childID][k];
				if(Tree[PID].FN[childH] && Tree[PID].dis[i]==disBF+Tree[PID].dis[childH]){
					Tree[PID].cnt[i]-=1;
				}
			}

			for(int k=0;k<VidtoTNid[line[i]].size();k++){
				PID=VidtoTNid[line[i]][k];
				if(PID>children){
					if(Tree[PID].FN[i] && Tree[PID].dis[childH]==disBF+Tree[PID].dis[i]){
						Tree[PID].cnt[childH]-=1;
					}
				}
			}

			int dis=INF; int count=0;
			int Dvb; int b,bH; int DDvb=INF;
			for(int j=0;j<Tree[children].vert.size();j++){
				Dvb=Tree[children].vert[j].second.first;
				b=Tree[children].vert[j].first;
				bH=Tree[rank[b]].height-1;
				if(bH<i){
					if(Dvb+Tree[rank[line[i]]].dis[bH]<dis){
						dis=Dvb+Tree[rank[line[i]]].dis[bH];
						count=1;
					}else if(Dvb+Tree[rank[line[i]]].dis[bH]==dis){
						count+=1;
					}
				}else if(bH==i){
					DDvb=Dvb;
					if(Dvb<dis){
						dis=Dvb;
						count=1;
					}else if(Dvb==dis){
						count+=1;
					}
				}else{
					if(Dvb+Tree[rank[b]].dis[i]<dis){
						dis=Dvb+Tree[rank[b]].dis[i];
						count=1;
					}else if(Dvb+Tree[rank[b]].dis[i]==dis){
						count+=1;
					}
				}
			}
			if(DDvb==dis) Tree[children].FN[i]=true;
			Tree[children].dis[i]=dis;
			Tree[children].cnt[i]=count;
		}
	}

	line.push_back(childID);
	for(int i=0;i<Tree[children].ch.size();i++){
		eachNodeProcessIncrease1(Tree[children].ch[i],line,changelabel);
	}
	line.pop_back();
}


void FRNGraph::buildRMQ(){
	toRMQ.assign(nodenum,0);
	buildRMQDFS(0, 1);
	RMQIndex.push_back(EulerSeq);

	int m = EulerSeq.size();
	for (int i = 2, p = 1; i < m; i = i * 2, p++){
		vector<int> tmp;
		tmp.assign(m,0);
		for (int j = 0; j < m - i; j++){
			int x = RMQIndex[p - 1][j], y = RMQIndex[p - 1][j + i / 2];
			if (Tree[x].height < Tree[y].height)
				tmp[j] = x;
			else tmp[j] = y;
		}
		RMQIndex.push_back(tmp);
	}
}

void FRNGraph::EachNodeProBDis5(int child,vector<int>& line,set<int>& vertexIDChL, map<int,int>& checkedDis){
	bool ProIDdisCha=false;

	if(Tree[child].DisRe.size()!=0){
		for(int k=0;k<Tree[child].vert.size();k++){
			int b=Tree[child].vert[k].first, bH=Tree[rank[b]].height-1,vbW=Tree[child].vert[k].second.first;
			if(Tree[child].FN[bH]){
				if(Tree[child].DisRe.find(b)!=Tree[child].DisRe.end()){
					for(int i=0;i<bH;i++){
						checkedDis.insert(build_pair(child,i));
						if(Tree[child].dis[i]>vbW+Tree[rank[b]].dis[i]){
							Tree[child].dis[i]=vbW+Tree[rank[b]].dis[i];
							Tree[child].FN[i]=false;
							ProIDdisCha=true;
						}
					}
					for(int i=bH+1;i<line.size();i++){
						checkedDis.insert(build_pair(child,i));
						if(Tree[child].dis[i]>vbW+Tree[rank[line[i]]].dis[bH]){
							Tree[child].dis[i]=vbW+Tree[rank[line[i]]].dis[bH];
							Tree[child].FN[i]=false;
							ProIDdisCha=true;
						}
					}

				}else{

					if(vertexIDChL.find(b)!=vertexIDChL.end()){
						for(int i=0;i<bH;i++){
							checkedDis.insert(build_pair(child,i));
							if(Tree[child].dis[i]>vbW+Tree[rank[b]].dis[i]){
								Tree[child].dis[i]=vbW+Tree[rank[b]].dis[i];
								Tree[child].FN[i]=false;
								ProIDdisCha=true;
							}
						}
					}
					for(int i=bH+1;i<line.size();i++){
						checkedDis.insert(build_pair(child,i));
						if(Tree[child].dis[i]>vbW+Tree[rank[line[i]]].dis[bH]){
							Tree[child].dis[i]=vbW+Tree[rank[line[i]]].dis[bH];
							Tree[child].FN[i]=false;
							ProIDdisCha=true;
						}
					}

				}
			}
		}
	}else{
		for(int k=0;k<Tree[child].vert.size();k++){
			int b=Tree[child].vert[k].first, bH=Tree[rank[b]].height-1,vbW=Tree[child].vert[k].second.first;
			if(Tree[child].FN[bH]){
				if(vertexIDChL.find(b)!=vertexIDChL.end()){
					for(int i=0;i<bH;i++){
						checkedDis.insert(build_pair(child,i));
						if(Tree[child].dis[i]>vbW+Tree[rank[b]].dis[i]){
							Tree[child].dis[i]=vbW+Tree[rank[b]].dis[i];
							Tree[child].FN[i]=false;
							ProIDdisCha=true;
						}
					}
				}
				for(int i=bH+1;i<line.size();i++){
					checkedDis.insert(build_pair(child,i));
					if(Tree[child].dis[i]>vbW+Tree[rank[line[i]]].dis[bH]){
						Tree[child].dis[i]=vbW+Tree[rank[line[i]]].dis[bH];
						Tree[child].FN[i]=false;
						ProIDdisCha=true;
					}
				}
			}
		}
	}

	if(ProIDdisCha){
		vertexIDChL.insert(Tree[child].uniqueVertex);
	}

	line.push_back(Tree[child].uniqueVertex);
	for(int i=0;i<Tree[child].ch.size();i++){
		EachNodeProBDis5(Tree[child].ch[i], line, vertexIDChL,checkedDis);
	}
	line.pop_back();

}

void FRNGraph::FHLIincBatMT(vector<pair<pair<int,int>,pair<int,int>>>& wBatch){
	int checknum=0;
	map<pair<int,int>,int> OCdis;
	NodeOrderss.assign(NodeOrder.begin(),NodeOrder.end());
	vector<set<int>> SCre; 
	set<int> ss; 
	SCre.assign(nodenum,ss);
	set<OrderCompp> OC; OC.clear();

	for(int k=0;k<wBatch.size();k++){
		int a=wBatch[k].first.first;
		int b=wBatch[k].first.second;
		int oldW=wBatch[k].second.first;
		int newW=wBatch[k].second.second;

		if(oldW!=newW){
		for(int i=0;i<Neighbor[a].size();i++){
			if(Neighbor[a][i].first==b){
				Neighbor[a][i].second=newW;
				break;
			}
		}
		for(int i=0;i<Neighbor[b].size();i++){
			if(Neighbor[b][i].first==a){
				Neighbor[b][i].second=newW;
				break;
			}
		}

		int lid,hid;
		if(NodeOrder[a]<NodeOrder[b]){
			lid=a;hid=b;
		}else{
			lid=b;hid=a;
		}

		for(int i=0;i<Tree[rank[lid]].vert.size();i++){
			if(Tree[rank[lid]].vert[i].first==hid){
				if(Tree[rank[lid]].vert[i].second.first==oldW){
					Tree[rank[lid]].vert[i].second.second-=1;
					if(Tree[rank[lid]].vert[i].second.second<1){
						OCdis[build_pair(lid,hid)]=oldW;
						SCre[lid].insert(hid);
						OC.insert(OrderCompp(lid));
					}
				}
				break;
			}
		}
	}
	}

	vector<int> StartPointVertexSet; StartPointVertexSet.clear();
	vector<int> StartPointVertexSetNew;
	bool influence;
	int ProID; vector<int> line;
	while(!OC.empty()){
		ProID=(*OC.begin()).x;
		OC.erase(OC.begin());
		vector<pair<int,pair<int,int>>> Vert=Tree[rank[ProID]].vert;
		influence=false;

		line.clear(); line.reserve(heightMax);
		int pachid=ProID;
		while(Tree[rank[pachid]].height>1){
			line.insert(line.begin(),pachid);
			pachid=Tree[Tree[rank[pachid]].pa].uniqueVertex;
		}
		line.insert(line.begin(),pachid);

		for(auto it=SCre[ProID].begin();it!=SCre[ProID].end();it++){
			int Cid=*it; int Cw=OCdis[build_pair(ProID,Cid)];
			int cidH=Tree[rank[Cid]].height-1;

			map<int,int> Hnei; 
			vector<pair<int,int>> Lnei; 
			for(int j=0;j<Vert.size();j++){
				if(NodeOrder[Vert[j].first]>NodeOrder[Cid]){
					Hnei[Vert[j].first]=Vert[j].second.first;
				}else if(NodeOrder[Vert[j].first]<NodeOrder[Cid]){
					Lnei.push_back(build_pair(Vert[j].first,Vert[j].second.first));
				}
			}
			int hid,lid;
			for(int j=0;j<Tree[rank[Cid]].vert.size();j++){
				hid=Tree[rank[Cid]].vert[j].first;
				if(Hnei.find(hid)!=Hnei.end()){
					if(Cw+Hnei[hid]==Tree[rank[Cid]].vert[j].second.first){
						Tree[rank[Cid]].vert[j].second.second-=1;
						if(Tree[rank[Cid]].vert[j].second.second<1){
							SCre[Cid].insert(hid);
							OC.insert(OrderCompp(Cid));
							OCdis[build_pair(Cid,hid)]=Cw+Hnei[hid];
						}
					}
				}
			}
			for(int j=0;j<Lnei.size();j++){
				lid=Lnei[j].first;
				for(int k=0;k<Tree[rank[lid]].vert.size();k++){
					if(Tree[rank[lid]].vert[k].first==Cid){
						if(Tree[rank[lid]].vert[k].second.first==Cw+Lnei[j].second){
							Tree[rank[lid]].vert[k].second.second-=1;
							if(Tree[rank[lid]].vert[k].second.second<1){
								SCre[lid].insert(Cid);
								OC.insert(OrderCompp(lid));
								OCdis[build_pair(lid,Cid)]=Cw+Lnei[j].second;
							}
						}
						break;
					}
				}
			}

			if(Tree[rank[ProID]].FN[cidH]){
				influence=true;
				for(int i=0;i<cidH;i++){
					if(Tree[rank[ProID]].dis[i]==Cw+Tree[rank[Cid]].dis[i]){
						Tree[rank[ProID]].cnt[i]-=1;
					}
				}

				Tree[rank[ProID]].FN[cidH]=false;
				Tree[rank[ProID]].cnt[cidH]-=1;

				for(int i=cidH+1;i<Tree[rank[ProID]].dis.size();i++){
					if(Tree[rank[ProID]].dis[i]==Cw+Tree[rank[line[i]]].dis[cidH]){
						Tree[rank[ProID]].cnt[i]-=1;
					}
				}
			}

			Cw=INF; int countwt=0;

			for(int i=0;i<Neighbor[ProID].size();i++){
				if(Neighbor[ProID][i].first==Cid){
					Cw=Neighbor[ProID][i].second;
					countwt=1;
					break;
				}
			}

			int ssw,wtt,wid;
			vector<int> Wnodes;
			Wnodes.clear();

			if(ProID<Cid)
				Wnodes=SCconNodesMT[ProID][Cid]; 
			else
				Wnodes=SCconNodesMT[Cid][ProID];
			if(Wnodes.size()>0){
				for(int i=0;i<Wnodes.size();i++){
					wid=Wnodes[i];
					for(int j=0;j<Tree[rank[wid]].vert.size();j++){
						if(Tree[rank[wid]].vert[j].first==ProID){
							ssw=Tree[rank[wid]].vert[j].second.first;
						}
						if(Tree[rank[wid]].vert[j].first==Cid){
							wtt=Tree[rank[wid]].vert[j].second.first;
						}
					}

					if(ssw+wtt<Cw){
						Cw=ssw+wtt;
						countwt=1;
					}else if(ssw+wtt==Cw){
						countwt+=1;
					}
				}
			}

			for(int i=0;i<Tree[rank[ProID]].vert.size();i++){
				if(Tree[rank[ProID]].vert[i].first==Cid){
					Tree[rank[ProID]].vert[i].second.first=Cw;
					Tree[rank[ProID]].vert[i].second.second=countwt;
					break;
				}
			}
		}

		if(influence){
			StartPointVertexSetNew.clear(); StartPointVertexSetNew.reserve(StartPointVertexSet.size()+1);
			StartPointVertexSetNew.push_back(ProID);
			int rnew=rank[ProID],r;
			for(int i=0;i<StartPointVertexSet.size();i++){
				r=rank[StartPointVertexSet[i]];
				if(LCAQuery(rnew,r)!=rnew){
					StartPointVertexSetNew.push_back(StartPointVertexSet[i]);
				}
			}
			StartPointVertexSet=StartPointVertexSetNew;
		}

	}

	int StartPointVertexID;
	for(int i=0;i<StartPointVertexSet.size();i++){
		StartPointVertexID=StartPointVertexSet[i];
		vector<int> linee; 
		linee.reserve(heightMax);
		int pachidd=Tree[Tree[rank[StartPointVertexID]].pa].uniqueVertex;
		while(Tree[rank[pachidd]].height>1){
			linee.insert(linee.begin(),pachidd);
			pachidd=Tree[Tree[rank[pachidd]].pa].uniqueVertex;
		}
		linee.insert(linee.begin(),pachidd);

		eachNodeProcessIncrease1(rank[StartPointVertexID], linee,checknum);
	}
}


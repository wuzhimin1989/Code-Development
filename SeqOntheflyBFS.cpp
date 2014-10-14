#include <iostream>
#include <iostream>
#include <string>
#include <time.h>
#include <queue>
#include <set>
#include <list>
#include <fstream>
#include <iomanip>
using namespace std;

class Nodemark{
public:
	unsigned int beginInt;
	unsigned int endInt;
	unsigned int synbeginInt;
	unsigned int synendInt;
};

unsigned int GetStateinVec(int index, unsigned int * ltssbits, unsigned int svec)
{
	int sbeginbit, sendbit;
	unsigned int ltsid;

	sbeginbit = 0;
	sendbit = 0;

	for(int i = 0; i < index; i++){
		sbeginbit += ltssbits[i]; 
	}
	sendbit = sbeginbit + ltssbits[index] - 1;
	svec  = svec << sbeginbit; 
	svec = svec >> (sbeginbit + 31 - sendbit);
	ltsid = svec;
	return ltsid;

}

bool GetAllsuccessors(unsigned int allLTS, unsigned int allStates, unsigned int allTrans, unsigned int ltsindex, unsigned int sindex, Nodemark * result)
{
	unsigned int statesbegin, transbegin, transborder;
	statesbegin = allLTS[ltsindex];
	transbegin = allStates[statesbegin + sindex];

	if(transbegin == 0 && (ltsindex!=0 || sindex!=0))
		return false;

	transborder = allStates[statesbegin + sindex + 1];

	result->beginInt = transbegin;
	result->endInt = transborder - 1;

	result->synendInt = allTrans[transborder - 1];

	transborder = allStates[statesbegin + sindex];
	if(transborder == 0)
		result->synbeginInt = 0;
	else
		result->synbeginInt = allTrans[transborder - 1];

	if(result->beginInt == result->endInt && result->synendInt == result->synbeginInt)
		return false;

	return true;
}

bool NewStateV(unsigned int * targetV, int tindex, int * index, int *count,  unsigned char* OutgoingTs, unsigned int OutGTbyte, unsigned int * ltssbits, unsigned int EEncode)
{
	unsigned int tmp = *targetV;
	unsigned int tostate = 0;
	int newsbeginbit = 0, endbit;
	unsigned int Secode = ltssbits[tindex];

	int replacebeginbyte, replaceendbyte;
	unsigned int i,j;
	int k;

	replacebeginbyte = *count * OutGTbyte;
	replaceendbyte =  (*count + 1)*OutGTbyte;

	//if(EEncode < 8){
	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] << EEncode;
	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] >> EEncode; //event
	//}else{
	//	replacebeginbyte++;
	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] << EEncode - 8;
	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] >> EEncode - 8;
	//}

	for(i = 0; i < tindex; i++){
		newsbeginbit += ltssbits[i];
	}

	endbit = newsbeginbit + Secode - 1;

	if((Secode+EEncode) <= 8){
		tostate = (int) OutgoingTs[replaceendbyte - 1];
	}else{
		tostate = 0;

		for( k = replaceendbyte - 1; k > replacebeginbyte-1; k--)
			tostate = tostate | (OutgoingTs[k] << 8 * (replaceendbyte - 1 - k));
	}

	if(tostate == 0){
		(*index)++;
		(*count)=0;
		return false;
	}		

	tostate = tostate << 32-Secode;
	tostate = tostate >> 32-Secode; 
	tostate = tostate << (31-endbit);

	i = tmp >> (31 - newsbeginbit + 1);
	i = i << (31 - newsbeginbit + 1);
	j = tmp << endbit+1;
	j = j >> endbit+1;

	* targetV = (unsigned int) (i | j | tostate);

	if((OutGTbyte)*(*count + 2) > 4){
		* index += 1;
		*count = 0;
	}else
		(*count)++;

	return true;
}

void DecodeTransitions(int type, unsigned int * synctrans, int beginindex, int count, unsigned int * Tostate, unsigned int * Tevent, unsigned int OutGTe, unsigned int Statebitwidth)
{
	unsigned int tmp = 0;
	unsigned int startbyte, endbyte;

	while(tmp==0 && count >= 0){
		startbyte = (count * OutGTe)%4;
		endbyte = ((count + 1)*OutGTe)%4;

		if(endbyte == 0)
			endbyte = 4;

		tmp = synctrans[beginindex];

		tmp = tmp << (startbyte) * 8;
		tmp = tmp >> (startbyte + 4 - endbyte)*8; 

		*Tostate = (unsigned int)(tmp << 32 - Statebitwidth) >> (32- Statebitwidth);
		*Tevent = (unsigned int)tmp >> Statebitwidth;

		if(tmp == 0 && type == 1)
			break;
		count--;
	}
}

void SynTwoStatesCPU(unsigned int * tmpStateV, unsigned int succStateV, int i, unsigned int * bitwidth){
	unsigned int localstate;
	int beginbit = 0, endbit;
	unsigned int i,j;

	for(i = 0; i < index;i++){
		beginbit += tex1Dfetch(LTSSTATEBITS, i);
	} 

	endbit = beginbit + tex1Dfetch(LTSSTATEBITS,index);

	s2 = s2 << 32-endbit;

	i = ((*s1) << endbit)>>endbit;
	//i = (*s1) >> endbit;
	j = ((*s1) >> 32-beginbit)<<32-beginbit;

	*s1 = i | j | s2;
}

void CallCudaBFS(unsigned int * H_AllLTS, unsigned int * H_AllStates, unsigned int * H_AllTransitions, unsigned int* H_AllSyncTrans, unsigned int H_InitialSV, unsigned int * H_LTSStateEncodeBits, unsigned int LTSNum,unsigned int AllLTSStateNum, unsigned int AllTransLength, unsigned int AllSyncTransLength, unsigned int HEventEncodeBits,  unsigned int * OutgoingTEbytes)
{
	int i,j,m,k;
	int SuccessorCount;

	queue<unsigned int> Taskqueue;
	set<unsigned int> Taskset;
	vector<unsigned int> Syncqueue;
	vector<unsigned int>::iterator Syncit;
	
	vector<unsigned int> Syncevents;
	vector<unsigned int>::iterator ITS;
	bool ifexist;
	
	queue<unsigned int> VisitedS;
	
	unsigned int newStateV;
	unsigned int * succStateV;
	unsigned int * tmpStateV;
	unsigned int newState;
	unsigned int belonglts;
	unsigned int transevent;
	
	unsigned int *tmp;
	
	unsigned int tmpcount;
	unsigned int tmpoutT;
	unsigned char tmpT[4];
	
	int x,y;
		
	bool ifoutgoing;
	int ifoutgoingcount;
	int realsucccount = 0;
	Nodemark allsucc;
		
	//SuccessorCount = 1;
	Taskqueue.push(H_InitialStateV);
	while(!Taskqueue.empty()){
		newStateV = Taskqueue.front();
		ifoutgoingcount = 0;
		realsucccount = 0;
		for(i = 0; i < LTSNum; i++){
			ifoutgoing = false;
			newState = GetStateinVec(i, H_LTSStateEncodeBits, newStateV);
			ifoutgoing = GetAllsuccessors(H_AllLTS, H_AllStates, H_AllTransitions, belonglts, newState, &allsucc);
			if(!ifoutgoing){
				ifoutgoingcount++;
				continue;
			}
	
			m = allsucc.beginInt;
			x = -1;
			y = 0;
			while(m < allsucc.endInt){
				succStateV = new unsigned int[1];
					
				if(x != m){
					tmpoutT = H_AllTransitions[m];
					tmpT[0] = (char)(tmpoutT >> 24);
					tmpT[1] = (char)(tmpoutT >> 16);
					tmpT[2] = (char)(tmpoutT >> 8);
					tmpT[3] = (char)tmpoutT;
					x = m;	
				}
				if(!NewStateV(succStateV, i, &m, &y, tmpT, H_LTSStateEncodeBits, OutgoingTEbytes[i], H_LTSStateEncodeBits, HEventEncodeBits))
					continue;
				else
					realsucccount++;

				if(Taskset.insert(*succStateV).second){
					Taskqueue.push(*succStateV);
					//SuccessorCount++;
				}
			}
	
			k = allsucc.synbeginInt;
			tmpcount = 0;
			x = -1;
			y = 0;
			while(k < allsucc.synendInt){
				succStateV = new unsigned int[1];
	
				DecodeTransitions(1, H_AllSynctrans, k, tmpcount, &newState, &transevent, OutgoingTEbytes[belonglts], H_LTSStateEncodeBits[i]);
	
				if(x != k){
					tmpoutT = H_AllSynctrans[k];
					tmpT[0] = (char)(tmpoutT >> 24);
					tmpT[1] = (char)(tmpoutT >> 16);
					tmpT[2] = (char)(tmpoutT >> 8);
					tmpT[3] = (char)tmpoutT;
					x = k;	
				}			
				if(!NewStateV(succStateV, i, &k, &y, tmpT, H_LTSStateEncodeBits, OutgoingTEbytes[i], H_LTSStateEncodeBits, HEventEncodeBits))
					continue;
	
				tmpcount++;
				j = 0;
				for(ITS = Syncevents.begin(); ITS < Syncevents.end(); ++ITS){
					if(*ITS == transevent){
						ifexist = true;
						break;
					}else
						j++;
				}
				if(ifexist){			
					tmpStateV = (unsigned int *)&(Syncqueue[j]);
					SynTwoStatesCPU(tmpStateV, *succStateV, i, H_LTSStateEncodeBits);
					realsucccount++;
					if(Taskset.insert(*tmpStateV).second)
						Taskqueue.push(*tmpStateV);
						
				}else{
					Syncevents.push_back(transevent);
					Syncqueue.push_back(*succStateV);
					SuccessorCount++;
				}	
			}

			/*for(Syncit = Syncqueue.begin(); Syncit != Syncqueue.end(); Syncit++) {
				Taskqueue.push(*Syncit);
			}*/
			Syncqueue.clear();

			Taskset.push(newStateV);
			Taskqueue.pop();
		}
		if(ifoutgoingcount == LTSNum || realsucccount == 0){
			cout<<"detected deadlock"<<endl;
			break;
		}
			
	}
}

int main()
{
	//read data from file
	int i;
	unsigned int * AllLTS;
	unsigned int * AllStates;
	unsigned int * AllTransitions;
	unsigned int * AllSyncTrans;

	unsigned int InitialV;
	unsigned int LTSNum;
	unsigned int StatesNUM;
	unsigned int AlltransNum;
	unsigned int AllsynctransNum;
	//unsigned int Synindexencodebyte;
	//unsigned int LTSEncodebyte;
	unsigned int EventEncodebits;

	unsigned int * LTSStateEncodebits;
	unsigned int * OutgoingTEbytes;

	ifstream file1; //for all LTS
	ifstream file2; //for All states
	ifstream file3; //for all trans
	ifstream file4; //for all sync trans;
	ifstream file5; //for other parameters

	file1.open("../test/encode/alllts.txt");
	file2.open("../test/encode/allstates.txt");
	file3.open("../test/encode/alltrans.txt");
	file4.open("../test/encode/allsynctrans.txt");
	file5.open("../test/encode/parameters.txt");

	//parameters
	file5>>InitialV;
	file5>>LTSNum;
	file5>>StatesNUM;
	file5>>AlltransNum;
	file5>>AllsynctransNum;
	//file5>>Synindexencodebyte;
	//file5>>LTSEncodebyte;
	file5>>EventEncodebits;

	LTSStateEncodebits = new unsigned int[LTSNum];
	OutgoingTEbytes = new unsigned int[LTSNum];

	for(i=0; i < LTSNum; i++){
		file5>>LTSStateEncodebits[i];
	}

	for(i=0; i < LTSNum; i++){
		file5>>OutgoingTEbytes[i];
	}

	AllLTS = new unsigned int[LTSNum];
	AllStates = new unsigned int[StatesNUM + 1];
	AllTransitions = new unsigned int[AlltransNum];
	AllSyncTrans = new unsigned int[AllsynctransNum];

	file5.close();

	for(i=0; i <LTSNum; i++){
		file1>>AllLTS[i];
	}
	file1.close();

	for(i=0; i < StatesNUM+1; i++){
		file2>>AllStates[i];
	}
	file2.close();

	for(i=0; i < AlltransNum; i++){
		file3>>AllTransitions[i];
	}
	file3.close();

	for(i=0; i < AllsynctransNum; i++){
		file4>>AllSyncTrans[i];
	}
	file4.close();

	clock_t t1=clock();
	CallCudaBFS(AllLTS,AllStates,AllTransitions,AllSyncTrans,InitialV,LTSStateEncodebits, LTSNum, StatesNUM,AlltransNum,AllsynctransNum,EventEncodebits, OutgoingTEbytes);
	clock_t t2=clock();
	cout<<"TotalTime:"<<t2-t1<<"ms"<<endl;
}


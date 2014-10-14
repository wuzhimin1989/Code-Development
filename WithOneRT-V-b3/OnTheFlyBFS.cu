
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <sm_35_atomic_functions.h>


#include <iostream>
#include <string>
#include <time.h>
#include <queue>
#include <set>
#include <list>
#include <fstream>
#include <iomanip>
using namespace std;

texture<unsigned int, 1, cudaReadModeElementType> LTSOFFSET;  //1 means 1-dimension
texture<unsigned int, 1, cudaReadModeElementType> STATEOFFSET;
texture<unsigned int, 1, cudaReadModeElementType> OUTGOINGDETAIL;
texture<unsigned int, 1, cudaReadModeElementType> SYNCOUTGOING;
texture<unsigned int, 1, cudaReadModeElementType> TRANSEBYTES;
texture<unsigned int, 1, cudaReadModeElementType> LTSSTATEBITS;

__constant__ int LA1;
__constant__ int LA2;
__constant__ int LA3;
__constant__ int LA4;
__constant__ int LB4;
__constant__ int GA1;
__constant__ int GA2;
__constant__ int GA3;
__constant__ int LB1;
__constant__ int LB2;
__constant__ int LB3;
__constant__ int GB1;
__constant__ int GB2;
__constant__ int GB3;
__constant__ int GA4;
__constant__ int GB4;
__constant__ int BUCA;
__constant__ int BUCB;
__constant__ int TableSize;
__constant__ unsigned int PrimeNum = 334214459;
__constant__ int IterationTime;
__constant__ int HashNum;
__constant__ int ProbeTimes;

static const unsigned int EMPTYVECT32 = 0x7FFFFFFF;
static const unsigned int P = 334214459;
static const unsigned int blocksize = 512;

//class LocalRecord{
//public:
//	char localmark;  //record the BFS layer in Shared Memory
//	char toevent;
//	unsigned int statevector;
//
//	__device__ void operator= (LocalRecord t){
//		localmark = t.localmark;
//		toevent = t.toevent;
//		statevector = t.statevector;
//	}
//};

class Bucket{
public:
	unsigned int beginindex;
	unsigned int endindex;

};

class Nodemark{
public:
	unsigned int beginInt;
	unsigned int endInt;
	unsigned int synbeginInt;
	unsigned int synendInt;

};

__device__ unsigned int *GlobalOpenHash;  
__device__ Bucket *GlobalBuckets;
__device__ unsigned int GlobalBucketNum;

__device__ unsigned int *GlobalbucketIndex;  //just used for open
__device__ unsigned int *GlobalbucketCount;

__device__ unsigned int *GlobalVisitedHash;  //here, for visited stateV, use hash to store back to global memory. While this hash doesn't kick the original one. For open stateV, use buckets hash.
//__device__ unsigned int GlobalVisitedHashoffset[3];

__device__ unsigned int communicationlayer[100];
__device__ bool communicationcollision[100];
__device__ Bucket *communicationGstore;  //store the buckets that child blocks store their data
__device__ bool Ifreturn2parent[100];

//__device__ volatile unsigned int * GlobalBucketsCount;

__device__ unsigned int OpenSize;

__device__ unsigned int openvisitedborder;

__device__ bool IFDeadlockDetected;
__device__ bool IfDeadlockfree;

volatile __device__ int SynMutex = 0;

__device__ void CudaInterBlocksSyn( int GoalValue)
{
	__syncthreads();
	int tid_in_block = threadIdx.x;

	// only thread 0 is used for synchronization
	//switch(tid_in_block) 
	//{
	//	case 0:
	if(tid_in_block == 0) 
	{
		atomicAdd((int*)&SynMutex, 1);
		while(SynMutex < GoalValue);
	}
	//}
	__syncthreads();
}

__device__ unsigned int Buckethash(unsigned int k)
{
	unsigned int bucketindex;
	bucketindex = k % GlobalBucketNum;
	return bucketindex;

}

__device__ unsigned int Globalhash(unsigned int k, int index)
{
	if(index == 0)
		return (GA1 * k + GB1) % PrimeNum % (3*TableSize);

	if(index == 1)
		return (GA2 * k + GB2) % PrimeNum % (3*TableSize);

	if(index == 2)
		return (GA3 * k + GB3) % PrimeNum % (3*TableSize);
	if(index == 3)
		return (GA4 * k + GB4) % PrimeNum % (3*TableSize);

}

__device__ unsigned int Localhash(unsigned int k, int index)
{
	if(index == 0){
		return (LA1 ^ k + LB1) % PrimeNum % TableSize;
	}

	if(index == 1){
		return (LA2 ^ k + LB2) % PrimeNum % TableSize;
	}

	if(index == 2){
		return (LA3 ^ k + LB3) % PrimeNum % TableSize;
	}

	if(index == 3){
		return (LA4^k + LB4) % PrimeNum % TableSize;
	}
	
}


__device__ unsigned int CudaGetStateinVec(int index, unsigned int svec)
{
	int sbeginbit, sendbit;
	unsigned int ltsid;

	sbeginbit = 0;
	sendbit = 0;

	for(int i = 0; i < index; i++){
		sbeginbit += tex1Dfetch(LTSSTATEBITS, i); 
	}
	sendbit = sbeginbit + tex1Dfetch(LTSSTATEBITS, index) - 1;
	svec  = svec << sbeginbit; 
	svec = svec >> (sbeginbit + 31 - sendbit);
	ltsid = svec;
	return ltsid;

}

__device__ bool CudaGetAllsuccessors(unsigned int ltsindex, unsigned int sindex, Nodemark * result)
{
	unsigned int statesbegin, transbegin, transborder;
	statesbegin = tex1Dfetch(LTSOFFSET, ltsindex);
	transbegin = tex1Dfetch(STATEOFFSET, statesbegin + sindex);
	
	if(transbegin == 0 && (ltsindex!=0 || sindex!=0))
		return false;

	transborder = tex1Dfetch(STATEOFFSET, statesbegin + sindex + 1);

	result->beginInt = transbegin;
	result->endInt = transborder - 1;
	
	result->synendInt = tex1Dfetch(OUTGOINGDETAIL, transborder - 1);

	transborder = tex1Dfetch(STATEOFFSET, statesbegin + sindex);
	if(transborder == 0)
		result->synbeginInt = 0;
	else
		result->synbeginInt = tex1Dfetch(OUTGOINGDETAIL, transborder - 1);

	if(result->beginInt == result->endInt && result->synendInt == result->synbeginInt)
		return false;

	return true;
}

__device__ bool CudaNewStateV(unsigned int * targetV, int tindex, int * index, int *count,  unsigned char* OutgoingTs, unsigned int OutGTbyte, unsigned int EEncode)
{
	unsigned int tmp = *targetV;
	unsigned int tostate = 0;
	int newsbeginbit = 0, endbit;
	unsigned int Secode = tex1Dfetch(LTSSTATEBITS, tindex);

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
		newsbeginbit += tex1Dfetch(LTSSTATEBITS, i);
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

__device__ void CudaDecodeTransitions(int type, int beginindex, int count, unsigned int * Tostate, unsigned int * Tevent, unsigned int OutGTe, unsigned int Statebitwidth)
{
	unsigned int tmp = 0;
	unsigned int startbyte, endbyte;

	while(tmp==0 && count >= 0){
		startbyte = (count * OutGTe)%4;
		endbyte = ((count + 1)*OutGTe)%4;

		if(endbyte == 0)
			endbyte = 4;

		tmp = tex1Dfetch(SYNCOUTGOING, beginindex);

		tmp = tmp << (startbyte) * 8;
		tmp = tmp >> (startbyte + 4 - endbyte)*8; 

		*Tostate = (unsigned int)(tmp << 32 - Statebitwidth) >> (32- Statebitwidth);
		*Tevent = (unsigned int)tmp >> Statebitwidth;
		
		if(tmp == 0 && type == 1)
			break;
		count--;
	}
}

//__device__ unsigned int CudaGenerateKey(unsigned int KV,  int snum)
//{
//	return KV;
//
//}

__device__ void SynTwoStates(unsigned int * s1, unsigned int s2, int index)
{
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

//void SynTwoStatesCPU(unsigned int * tmpStateV, unsigned int succStateV, int i, unsigned int newStateV, unsigned int * bitwidth){
//	int beginbit, endbit;
//	int beginbyte, endbyte;
//	int j,m;
//
//	unsigned char tmp1[4];
//	unsigned char tmp2[4];
//
//	tmp1[0] = (char)(*tmpStateV);
//	tmp1[1] = (char)(*tmpStateV >> 8);
//	tmp1[2] = (char)(*tmpStateV >> 16);
//	tmp1[3] = (char)(*tmpStateV >> 24);
//
//	tmp2[0] = (char)(succStateV);
//	tmp2[1] = (char)(succStateV >> 8);
//	tmp2[2] = (char)(succStateV >> 16);
//	tmp2[3] = (char)(succStateV >> 24);
//
//	for(j = 0; j < i; j++){
//		beginbit += bitwidth[j];
//	}
//	endbit = beginbit + bitwidth[i];
//
//	beginbyte = beginbit / 8;
//	endbyte = endbit / 8;
//	beginbit = beginbit % 8;
//	endbit = endbit % 8;
//
//	for(m = beginbyte; m < endbyte; m++){
//		tmp1[m] = tmp1[m] >> (8 - beginbit);
//		tmp2[m] = tmp2[m] << beginbit;
//		tmp2[m] = tmp2[m] >> beginbit;
//		tmp1[m] = tmp1[m] | tmp2[m];
//	}
//
//	*tmpStateV = (unsigned int)(tmp1[0] | tmp1[1] << 8 | tmp1[2] << 16 | tmp1[3] << 24);
//
//
//}


//__device__ bool CudaHashStore2()     //use cuckoo+probe
__device__ bool CudaHashStore(unsigned int beginHV, unsigned int PLTSNum, unsigned int * AllT, int Tnum, unsigned int * RkickoutRecord){
	unsigned int localhash;
	//LocalRecord kickoutRecord;
	char tmp;
	int i = 0, j = 0;

	unsigned int KeyV = beginHV;
	unsigned int kickKeyV;

	*RkickoutRecord = EMPTYVECT32;
	while(i < IterationTime){
		localhash = Localhash(KeyV, i % HashNum);
		if((atomicCAS(&(AllT[localhash]), EMPTYVECT32, KeyV))==EMPTYVECT32)
			return false;
		else{
			if(AllT[localhash] == KeyV)
				return false;

			kickKeyV = atomicExch(&(AllT[localhash]), KeyV);

			//kickKeyV =  atomicExch(&(AllT[(i%HashNum) * Tnum + (localhash - j + Tnum)%Tnum]), KeyV);
			KeyV = kickKeyV;
			i++;
		}
	}

	for(j = 0; j < Tnum; j++){
		if(atomicCAS(&(AllT[(localhash + j)%Tnum]), EMPTYVECT32, kickKeyV)==EMPTYVECT32)
			return false;
		else if(AllT[(localhash+j)%Tnum] == KeyV)
			return false;
			
		if(atomicCAS(&(AllT[(localhash - j + Tnum)%Tnum]), EMPTYVECT32, kickKeyV)==EMPTYVECT32){
			return false;
		}
		else if(AllT[(localhash-j + Tnum)%Tnum] == KeyV)
                        return false;
	}

	*RkickoutRecord = kickKeyV;
	return true;
}

__device__ bool CudaVisitedGlobalHashcal(unsigned int * HT, Bucket belongBucket, unsigned int hkey, unsigned int * hashresult){
	unsigned int hashposition;
	unsigned int KeyV;
	int i = 0;

	KeyV = hkey;

	while(i < HashNum){
		hashposition = Globalhash(KeyV, i);
		if(HT[belongBucket.beginindex + hashposition] == hkey)
			return true;
		i++;
	}

	KeyV = hkey;
	i = 0;
	while(i < HashNum){
		hashposition = Globalhash(KeyV, i);
		if(HT[belongBucket.beginindex + hashposition] == EMPTYVECT32){
			*hashresult = hashposition;
			return false;
		}
		i++;
	}
	
	*hashresult = Globalhash(KeyV, 0);
	return false;
}

__device__ bool CudaVisitedGlobalHashstore(unsigned int * HT, unsigned int hasbucket, unsigned int hashv, unsigned int insertedrecord, unsigned int ltsnum){
	Bucket buckethash;
	unsigned int kickV;
	int i = 0, j = 0;
	bool ifstored;
	unsigned int kickou;

	while(true){
		buckethash = GlobalBuckets[hasbucket];

		if(atomicCAS(&(HT[buckethash.beginindex + hashv]),EMPTYVECT32,insertedrecord)==EMPTYVECT32){
			return true;
		}else{
			i = 1;
			kickV = insertedrecord;
			while(i < IterationTime){
				kickou = atomicExch(&(HT[buckethash.beginindex + hashv]), kickV);
				hashv = Globalhash(kickou, i);
				if(atomicCAS(&(HT[buckethash.beginindex + hashv]),EMPTYVECT32,kickV)==EMPTYVECT32)
					return true;
				i++;
				kickV = kickou;
			}
			hasbucket++;
		}
		if(hasbucket > openvisitedborder-1)
			break;
	}

	i = 0;
	while(i < HashNum){
		hashv = Globalhash(kickV, i);
		for(j = 0; j < ProbeTimes; j++){
			if(atomicCAS(&(HT[buckethash.beginindex + (hashv + j) % TableSize]),EMPTYVECT32,kickV)==EMPTYVECT32)
				return true;
			if(atomicCAS(&(HT[buckethash.beginindex + (hashv - j+ TableSize) % TableSize]),EMPTYVECT32,kickV)==EMPTYVECT32)
				return true;
		}
	}

	return false;

}
			
__global__ void CUDADeadlockBFSVerifyChild(unsigned int ParentID, unsigned int PBucket, Bucket * Cbucket, unsigned int * CG_AllLTS, unsigned int * CG_AllStates, unsigned int * CG_AllTransitions, unsigned int * CG_AllSynctransitions,  unsigned int * CG_LTSStateEncodeBits, unsigned int CEventEncodeBits,unsigned int * OutgoingTEbytes,  unsigned int CG_Bucketnum, unsigned int PLTSNum, unsigned int StartNum)
{
	int i,j,m,k,x;

	int Inblocktid = threadIdx.x;
	int Ingridtid = threadIdx.x + blockIdx.x * blockDim.x;
	int InWarptid = Inblocktid % 31;
	int InvthreadgroupID;
	int vthreadgroupID;
	int Warpid = Inblocktid/32;
	int WarpNum = blockDim.x/32;

	unsigned int layer;
	
	unsigned int localstateV;
	unsigned int localstate;
	unsigned int localstate2;
	unsigned int belonglts;
	unsigned int transevent;
	unsigned int maxtransevent;

	unsigned int globalbuckethash;
	unsigned int visitedstore;

	unsigned int offsetborder; //used to mark the border of successors.
	bool ifanyoutgoing, ifgetnewstatev, ifglobaldup; //ifglobaldup means if this state is duplicated

	int vthreadgroupnuminblock;
	int vthreadgroupnuminwarp;
	//char tmp;

	//unsigned int localKey, localhash;
	unsigned int kickoutRecord;
	unsigned int insertRecord;
	unsigned int visitedRecord;
	unsigned int hkey;
	unsigned int getindex; // the index to get tasks

	unsigned int storeposition;

	unsigned int tmpoutT;
	unsigned char tmpT[4];

	int outgoingcount;
	Nodemark SuccessorMark;

	vthreadgroupnuminwarp = 32/PLTSNum;
	vthreadgroupnuminblock = vthreadgroupnuminwarp * (blockDim.x/32);
	if(InWarptid < vthreadgroupnuminwarp * PLTSNum){
		vthreadgroupID = Warpid*vthreadgroupnuminwarp + InWarptid/PLTSNum;
		InvthreadgroupID = InWarptid % PLTSNum;
	}else{
		vthreadgroupID = -1;
		InvthreadgroupID = -1;
	}

	__shared__ int nonewcount;
	__shared__ bool Ifcollisionhappens;
	__shared__ int maxlayer;

	extern __shared__ bool C[]; 
	bool * syncduplicate = C;
	bool * needsyndupdetect = &syncduplicate[vthreadgroupnuminblock*PLTSNum];
	bool * ifnooutgoing = &needsyndupdetect[vthreadgroupnuminblock];
	unsigned int * SynEventInteractive = (unsigned int *)&ifnooutgoing[vthreadgroupnuminblock*PLTSNum];
	unsigned int * SynStateInteractive = (unsigned int *)&(SynEventInteractive[vthreadgroupnuminblock*PLTSNum]);
	
	unsigned int * RecordTable = &(SynStateInteractive[vthreadgroupnuminblock*PLTSNum]);
	unsigned int * GroupStore = &RecordTable[blockDim.x];

	Bucket * WarpCBindex = (Bucket *)&GroupStore[vthreadgroupnuminblock];
	if(Inblocktid == 0){
		for(i=0; i<WarpNum; i++){
			WarpCBindex[i].beginindex = 0;
			WarpCBindex[i].endindex = 0;
		}

		for(i = 0; i < vthreadgroupnuminblock * PLTSNum; i++){
			ifnooutgoing[i] = false; 
			SynEventInteractive[i] = EMPTYVECT32;
		}
		
		for(i = 0; i < vthreadgroupnuminblock; i++)
			GroupStore[i] = EMPTYVECT32;
		nonewcount = 0;
		maxlayer = 0;
		Ifcollisionhappens = false;
	}

	if(InvthreadgroupID != -1){
		syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
	}

	if(InvthreadgroupID == 0){
		getindex = vthreadgroupnuminblock * blockIdx.x + vthreadgroupID;
		j=0;
		if(getindex < GlobalbucketCount[PBucket]){
			globalbuckethash = PBucket;
		}else{
			for(i = Cbucket->beginindex; i < Cbucket->endindex; i++){
				j += GlobalbucketCount[i];
				if(getindex < j){
					globalbuckethash = i;
					j -= GlobalbucketCount[i];
					getindex = getindex - j;
					break;
				}
			}
		}

	}
	__syncthreads();

	if(InvthreadgroupID == 0){
		GroupStore[vthreadgroupID] = GlobalOpenHash[globalbuckethash];
		GlobalOpenHash[globalbuckethash] = EMPTYVECT32;
	}
	
	do{
		if(IFDeadlockDetected)
			break;
		if(GroupStore[vthreadgroupID] != EMPTYVECT32){		
			localstate = CudaGetStateinVec(InvthreadgroupID, GroupStore[vthreadgroupID]);

			printf("vtg%d, itgid%d, gets%d\n", vthreadgroupID, InvthreadgroupID, localstate);
			belonglts = InvthreadgroupID;
			ifanyoutgoing = CudaGetAllsuccessors(belonglts, localstate-1, &SuccessorMark);
			ifglobaldup = false;
			//The successor generation consists of two steps: 1. For trans in alltransitions, process them directly. 2.For trans in allsynctrans, parallel sync is needed.
			if(ifanyoutgoing){
				outgoingcount=0;
				i = SuccessorMark.beginInt;
				//calculate global hash position for visited stateV
				if(InvthreadgroupID == 0){
					globalbuckethash = Buckethash(GroupStore[vthreadgroupID]);
					hkey = GroupStore[vthreadgroupID];
					ifglobaldup = CudaVisitedGlobalHashcal(GlobalVisitedHash, GlobalBuckets[globalbuckethash],hkey, &visitedstore);
				}

				localstateV = GroupStore[vthreadgroupID];
				visitedRecord = GroupStore[vthreadgroupID];

				j = 0;
				m = -1;
				while(i < SuccessorMark.endInt && !ifglobaldup){
					if(m != i){
						tmpoutT = tex1Dfetch(OUTGOINGDETAIL, i);
						tmpT[0] = (char)(tmpoutT >> 24);
						tmpT[1] = (char)(tmpoutT >> 16);
						tmpT[2] = (char)(tmpoutT >> 8);
						tmpT[3] = (char)tmpoutT;
						m = i;	
					}
					if(!CudaNewStateV(&localstateV, InvthreadgroupID, &i, &j, tmpT, tex1Dfetch(TRANSEBYTES, InvthreadgroupID), CEventEncodeBits ))
						continue;
				
					if(!Ifcollisionhappens){
						insertRecord = localstateV;					
					//hash store and duplicate elimination module.....
						if(CudaHashStore(insertRecord, PLTSNum, RecordTable, blockDim.x, &kickoutRecord))
						{
							Ifcollisionhappens = true;
						}
						outgoingcount++;
					}
					localstateV = GroupStore[vthreadgroupID];
					if(Ifcollisionhappens){
						break;
					}
			
			      }

				//synchronization part
				j = SuccessorMark.synbeginInt;
			
				if(!Ifcollisionhappens){
					bool  ifmatch;
					//int tmpcount=0;
					int tmpj = 0;
					int nosync;
					int lessthanall;

					m = 0;
					x = -1;
					CudaDecodeTransitions(0,SuccessorMark.synendInt-1, (SuccessorMark.synendInt - j + 1)*(4/tex1Dfetch(TRANSEBYTES,belonglts))-1,&localstate2, &maxtransevent, tex1Dfetch(TRANSEBYTES,belonglts), tex1Dfetch(LTSSTATEBITS,belonglts));
					while(j < SuccessorMark.synendInt){
						ifmatch = false;
						if(m == 0 && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum]){
							if(j == SuccessorMark.synendInt)
								break;
							CudaDecodeTransitions(1, j, tmpj, &SynStateInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], &SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], tex1Dfetch(TRANSEBYTES,belonglts), tex1Dfetch(LTSSTATEBITS, belonglts));
							
							if(SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] == 0)
							{
								SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] = EMPTYVECT32;
								break;
							}

							if(x != j){
								tmpoutT = tex1Dfetch(SYNCOUTGOING, j);
								tmpT[0] = (char)(tmpoutT >> 24);
								tmpT[1] = (char)(tmpoutT >> 16);
								tmpT[2] = (char)(tmpoutT >> 8);
								tmpT[3] = (char)tmpoutT;
								x = j;	
							}			
							CudaNewStateV(&localstateV, InvthreadgroupID, &j, &tmpj, tmpT, tex1Dfetch(TRANSEBYTES,InvthreadgroupID), CEventEncodeBits);
							
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;
						}
						nosync = 0;
						lessthanall = 0;
						m=0;
						for(i=0; i<PLTSNum; i++){
							if(i == InvthreadgroupID)
								continue;

							if(SynEventInteractive[i + vthreadgroupID * PLTSNum] == EMPTYVECT32)
							{
								nosync++;
								continue;
							}	
							if(SynEventInteractive[i + vthreadgroupID * PLTSNum] <= maxtransevent){  //if bigger than the maxtransevent of local, no need to compare as it's impossible to sync
								if(SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] > SynEventInteractive[i + vthreadgroupID * PLTSNum]){
									m++;

								}else if (SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] == SynEventInteractive[i + vthreadgroupID * PLTSNum]){
									if(needsyndupdetect[vthreadgroupID] == false)
										needsyndupdetect[vthreadgroupID] = true;
									//GENERATE SYNC STATE V.......
									SynTwoStates(&localstateV, SynStateInteractive[i + vthreadgroupID * PLTSNum], i);
									syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
									ifmatch = true;
								}else
									lessthanall++;
							}
						}
						if(nosync == PLTSNum - 1){
							break;
						}

						if(lessthanall == PLTSNum -1){
							m = 0;
							syncduplicate[InvthreadgroupID + vthreadgroupID*PLTSNum] = true;
							continue;
						}
	
						if(syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum])
							m = 0;

						if(needsyndupdetect[vthreadgroupID] && InvthreadgroupID == 0){   //duplicate elimination after synchronization, so just one synchronized result will be copied to hashtable.
							for(i = 0; i < PLTSNum; i++){
								if(syncduplicate[i + vthreadgroupID * PLTSNum]){
									for(k = 0; k < i; k++)
									{
										if(SynEventInteractive[k + vthreadgroupID * PLTSNum] == SynEventInteractive[i + vthreadgroupID * PLTSNum]){
											break;
										}
									}
									if(k == i){
										syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;
									}
								}
							}
						}

						if(ifmatch && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] == false){
							//hash copy to table
							
							insertRecord = localstateV;

							if(!Ifcollisionhappens)
							{
								if(CudaHashStore(insertRecord,  PLTSNum, RecordTable, blockDim.x, &kickoutRecord))
									Ifcollisionhappens = true;
								outgoingcount++;
							}
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
							if(Ifcollisionhappens){
								for(k = 511; k > 0; k--){
									if(kickoutRecord != EMPTYVECT32){
										if(atomicCAS(&(RecordTable[k]), EMPTYVECT32, kickoutRecord)==EMPTYVECT32){
											kickoutRecord = EMPTYVECT32;
											break;
										}
									}else{
										if(atomicCAS(&(RecordTable[k]), EMPTYVECT32, localstateV) == EMPTYVECT32){
											break;
										}
									}
								}
							}
						}

						if(ifmatch && m == 0){
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
						}
						if(j >= SuccessorMark.synendInt){
							SynEventInteractive[InvthreadgroupID + vthreadgroupID*PLTSNum] = EMPTYVECT32;
						}
						localstateV = GroupStore[vthreadgroupID];
					}
				}
				if(outgoingcount == 0 && !ifglobaldup)
					ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			
			}else{
				ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			}

			if(InvthreadgroupID == 0 && !ifglobaldup){
				for(i = 0; i < PLTSNum; i++){
					if(!ifnooutgoing[i + vthreadgroupID * PLTSNum] && !Ifcollisionhappens)
						break;
				}

				if(i == PLTSNum){
					printf("vtg%d detect deadlock\n", vthreadgroupID);
					IFDeadlockDetected = true;
				}
			}
		}
		CudaInterBlocksSyn(gridDim.x);
		if(IFDeadlockDetected){
			break;
		}

		if(GroupStore[vthreadgroupID] != EMPTYVECT32){
			if(InWarptid == 0&&!Ifcollisionhappens&&!ifglobaldup){
				//copy visited state to global memory
				CudaVisitedGlobalHashstore(GlobalVisitedHash, globalbuckethash, visitedstore, GroupStore[vthreadgroupID], PLTSNum);
				if(InvthreadgroupID == 0){
					GroupStore[vthreadgroupID] = EMPTYVECT32;
				}
			}
			
			if(Ifcollisionhappens || communicationcollision[ParentID]){
				

				if(IFDeadlockDetected)
					break;
				//load new kernel, copy data back
				unsigned int myareacount = 0;
			

				globalbuckethash = Buckethash((unsigned int)(blockIdx.x)) + openvisitedborder;
				if(blockIdx.x == 0){
					communicationGstore[ParentID].beginindex = (unsigned int)blockIdx.x;
				}
				if(blockIdx.x == blockDim.x - 1){
					communicationGstore[ParentID].endindex = (unsigned int)(blockIdx.x);
				}

				if(InWarptid == 0){
					for(m = Warpid*32; m<(Warpid + 1)*32; m++){
						//for(k = 0; k < HashNum; k++){
							if(RecordTable[m] != EMPTYVECT32)
								myareacount++;
						//}
					}
				
					k = 0;
					for(m = 0; m < vthreadgroupnuminwarp; m++){
						if(GroupStore[vthreadgroupnuminwarp * Warpid + m] != EMPTYVECT32){
							myareacount++;
							k++;
						}
					}

					WarpCBindex[Warpid].beginindex = atomicAdd(&GlobalbucketIndex[globalbuckethash], myareacount);
					WarpCBindex[Warpid].endindex = WarpCBindex[Warpid].beginindex + myareacount;
					atomicAdd(&GlobalbucketCount[globalbuckethash], myareacount);
				}

				if(InWarptid == 0){
					for(m = 0; m < k; m++){
						GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + m] = GroupStore[m];
					}
				}
				storeposition = WarpCBindex[Warpid].beginindex + InWarptid + k;
				//for(i=0; i<HashNum; i++){
					if(RecordTable[Warpid * 32 + InWarptid] != EMPTYVECT32){
						GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable[Warpid * 32 + InWarptid];
						RecordTable[Warpid * 32 + InWarptid] = EMPTYVECT32;
						storeposition+=32;
					}
				//}

				if(storeposition < WarpCBindex[Warpid].endindex)
				{
					//for(i=0; i<HashNum; i++){
						for(k = Warpid*32; k<(Warpid+1)*32; k++){
							if(RecordTable[k] != EMPTYVECT32){
								kickoutRecord = RecordTable[k];
								if(atomicCAS(&(RecordTable[k]), kickoutRecord, EMPTYVECT32) == kickoutRecord){
									GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
									storeposition+=32;
								}
							}
						}
					//}
				}

				//for the elements larger than 512, to be expanded........
				break;
			}
		
		}
		if(IFDeadlockDetected)
			break;

		if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID] == EMPTYVECT32){
			//got new stateV
			localstateV = EMPTYVECT32;
			ifgetnewstatev = false;
			
			//for(j = 0; j < HashNum; j++){
				for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
					if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable[i]), EMPTYVECT32)) != EMPTYVECT32)
					{
						ifgetnewstatev = true;
						break;
					}
				}

				//if(ifgetnewstatev == true)
				//	break;

				for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
					if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable[i]), EMPTYVECT32)) != EMPTYVECT32)
					{
						ifgetnewstatev = true;
						break;
					}
				}

				//if(ifgetnewstatev == true)
				//	break;
			//}
		}
	
		__syncthreads();

		if(InvthreadgroupID == 0 && layer == maxlayer + 1 && ifgetnewstatev == false){
			if(Inblocktid == 0){
				for(nonewcount = 0; nonewcount < vthreadgroupnuminblock; nonewcount++){
					if(GroupStore[nonewcount] != EMPTYVECT32){
						break;
					}
				}
				if(nonewcount == vthreadgroupnuminblock){
					break;
				}
			}
		}

		__syncthreads();

	}while(!IFDeadlockDetected);

	CudaInterBlocksSyn(gridDim.x);
}

__global__ void CUDADeadlockBFSVerify(unsigned int * PG_AllLTS, unsigned int * PG_AllStates, unsigned int * PG_AllTransitions, unsigned int * PG_AllSynctransitions, unsigned int * PG_Startlist, unsigned int * PG_LTSStateEncodeBits, unsigned int PEventEncodeBits, unsigned int * OutgoingTEbytes, unsigned int PLTSNum, unsigned int * G_RESULT, unsigned int PGBucketNum, unsigned int PAllLTSStatesNum, unsigned int StartNum)
{
	int i,j,m,k,x,y;

	int Inblocktid = threadIdx.x;
	int Ingridtid = threadIdx.x + blockIdx.x * blockDim.x;
	int InWarptid = Inblocktid % 32;
	int InvthreadgroupID;
	int vthreadgroupID;
	int Warpid = Inblocktid/32;
	int WarpNum = blockDim.x/32;

	unsigned int getindex; //the index to get the initial task from global memory.

	unsigned int localstateV;
	unsigned int localstate;
	unsigned int localstate2;
	unsigned int belonglts;
	unsigned int transevent;
	unsigned int maxtransevent;

	int nosync,lessthanall;

	unsigned int globalbuckethash;
	unsigned int visitedstore;

	unsigned int tmpoutT;

	int outgoingcount;

	unsigned int offsetborder; //used to mark the border of successors.
	bool ifanyoutgoing, ifgetnewstatev, ifglobaldup; //ifglobaldup means if this state is duplicated

	int vthreadgroupnuminblock;
	int vthreadgroupnuminwarp;
	unsigned char tmpT[4];

	unsigned int localKey, localhash;
	unsigned int kickoutRecord;
	unsigned int insertRecord;
	unsigned int visitedRecord;
	unsigned int hkey;

	unsigned int storeposition;
	Nodemark SuccessorMark;
	
	vthreadgroupnuminwarp = 32/PLTSNum;
	vthreadgroupnuminblock = vthreadgroupnuminwarp * (blockDim.x/32);
	if(InWarptid < vthreadgroupnuminwarp * PLTSNum){
		vthreadgroupID = Warpid*vthreadgroupnuminwarp + InWarptid/PLTSNum;
		InvthreadgroupID = InWarptid % PLTSNum;
	}else{
		vthreadgroupID = -1;
		InvthreadgroupID = -1;
	}
	
	__shared__ bool Ifcollisionhappens;
	__shared__ int collisiontimes; //how the collision times reflect the occupation rate is needed to be explored with experiments.
	__shared__ bool ifblocknostate;
	__shared__ int nonewcount;

	__shared__ bool haveChild;
	__shared__ int launchtime;

	i = vthreadgroupnuminblock * PLTSNum;
	extern __shared__ bool C[]; 
	bool * syncduplicate = C;
	bool * needsyndupdetect = &syncduplicate[i];
	bool * ifnooutgoing = &needsyndupdetect[vthreadgroupnuminblock];

	unsigned int * SynEventInteractive = (unsigned int *)&ifnooutgoing[i];
	unsigned int * SynStateInteractive = &SynEventInteractive[i];
	unsigned int * RecordTable = &(SynStateInteractive[i]);
	
	unsigned int * GroupStore = &RecordTable[blockDim.x];

	Bucket * WarpCBindex = (Bucket *)&GroupStore[vthreadgroupnuminblock];
	
	if(Inblocktid == 0){
		for(i = 0; i < vthreadgroupnuminblock * PLTSNum; i++){
			ifnooutgoing[i] = false; 
			SynEventInteractive[i] = EMPTYVECT32;
		}
		for(i = 0; i < blockDim.x; i++){
			RecordTable[i] = EMPTYVECT32;
		}

		for(i = 0; i < vthreadgroupnuminblock; i++)
			GroupStore[i] = EMPTYVECT32;
		nonewcount = 0;
		haveChild = false;
		launchtime = 0;
		ifblocknostate = false;
	
		for(i=0; i<WarpNum; i++){
			WarpCBindex[i].beginindex = 0;
			WarpCBindex[i].endindex = 0;
		}
		Ifcollisionhappens = false;
	}

	__syncthreads();

	if(Ingridtid == 0){
		GlobalbucketCount = new unsigned int[PGBucketNum];
		GlobalbucketIndex = new unsigned int[PGBucketNum];
		GlobalBucketNum = PGBucketNum;
		GlobalOpenHash = new unsigned int[blockDim.x * 3 * PLTSNum * 4 ];
		GlobalBuckets = new Bucket[GlobalBucketNum];
		GlobalVisitedHash = new unsigned int[blockDim.x * 3 * PLTSNum * 4]; //bucket/2
		communicationGstore = new Bucket[100];
		
		for(i = 0; i < blockDim.x * 3 * PLTSNum * 4; i++)
			GlobalOpenHash[i] = EMPTYVECT32;

		for(i = 0; i < blockDim.x * 3 * PLTSNum * 4; i++)
			GlobalVisitedHash[i] = EMPTYVECT32;

		for(i = 0; i < PLTSNum * 4; i++){
			GlobalBuckets[i].beginindex = i * blockDim.x;
			GlobalBuckets[i].endindex = (i+1)* 3 *blockDim.x - 1;
		}
	
		for(i = PLTSNum * 4; i < PLTSNum * 8; i++){
			GlobalBuckets[i].beginindex = (i-PLTSNum*4)*blockDim.x;
			GlobalBuckets[i].endindex = (i+1-PLTSNum*4)* 3 *blockDim.x - 1;
		}
	}

	if(InvthreadgroupID != -1){
		syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
	}

	if(InvthreadgroupID == 0 && vthreadgroupID < StartNum){
		getindex = vthreadgroupnuminblock * blockIdx.x + vthreadgroupID;
		GroupStore[vthreadgroupID] = PG_Startlist[getindex];

		needsyndupdetect[vthreadgroupID] = false;
	}

	CudaInterBlocksSyn(gridDim.x);
	//while(GroupStore[vthreadgroupID].statevector == EMPTYVECT32);

	do{
		if(GroupStore[vthreadgroupID] != EMPTYVECT32){
			localstate = CudaGetStateinVec(InvthreadgroupID, GroupStore[vthreadgroupID]);
			printf("vtg%d, itgid%d, gets%d\n", vthreadgroupID, InvthreadgroupID, localstate);

			belonglts = InvthreadgroupID;
			ifanyoutgoing = CudaGetAllsuccessors(belonglts, localstate-1, &SuccessorMark);
			ifglobaldup = false;
			//The successor generation consists of two steps: 1. For trans in alltransitions, process them directly. 2.For trans in allsynctrans, parallel sync is needed.
			if(ifanyoutgoing){
				outgoingcount = 0;
				i = SuccessorMark.beginInt;
				//calculate global hash position for visited stateV
				if(InvthreadgroupID == 0){
					globalbuckethash = Buckethash(GroupStore[vthreadgroupID]);
					hkey = GroupStore[vthreadgroupID];
					ifglobaldup = CudaVisitedGlobalHashcal(GlobalVisitedHash, GlobalBuckets[globalbuckethash],hkey, &visitedstore);
				}

				localstateV = GroupStore[vthreadgroupID];

				j = 0;
				m = -1;
				while(i < SuccessorMark.endInt && !ifglobaldup){
					if(m != i){
						tmpoutT = tex1Dfetch(OUTGOINGDETAIL, i);
						tmpT[0] = (char)(tmpoutT >> 24);
						tmpT[1] = (char)(tmpoutT >> 16);
						tmpT[2] = (char)(tmpoutT >> 8);
						tmpT[3] = (char)tmpoutT;
						m = i;	
					}
					if(!CudaNewStateV(&localstateV, InvthreadgroupID, &i, &j, tmpT,tex1Dfetch(TRANSEBYTES,InvthreadgroupID), PEventEncodeBits ))
						continue;
				
					if(!Ifcollisionhappens){
						insertRecord = localstateV;	
						//hash store and duplicate elimination module.....
						if(CudaHashStore(insertRecord, PLTSNum, RecordTable, blockDim.x, &kickoutRecord))
							Ifcollisionhappens = true;
						outgoingcount++;
					
					}
					localstateV = GroupStore[vthreadgroupID];
					if(Ifcollisionhappens){
						break;
					}
			
				}
				//synchronization part
				j = SuccessorMark.synbeginInt;
			
				if(!Ifcollisionhappens && SuccessorMark.synbeginInt != SuccessorMark.synendInt && !ifglobaldup){
					bool  ifmatch;
					//int tmpcount=0;
					int tmpj = 0;
					m = 0;
					x = -1;
					CudaDecodeTransitions(0,SuccessorMark.synendInt-1, (SuccessorMark.synendInt - j)*(4/tex1Dfetch(TRANSEBYTES, belonglts))-1,&localstate2, &maxtransevent, tex1Dfetch(TRANSEBYTES, belonglts), tex1Dfetch(LTSSTATEBITS, belonglts));
					while(j < SuccessorMark.synendInt){
						ifmatch = false;
						if(m == 0 && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum]){
							if(j == SuccessorMark.synendInt)
								break;
							CudaDecodeTransitions(1, j, tmpj, &SynStateInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], &SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], tex1Dfetch(TRANSEBYTES,belonglts), tex1Dfetch(LTSSTATEBITS, belonglts));
							if(SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] == 0)
							{
								SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] = EMPTYVECT32;
								break;
							}
							
							if(x != j){
								tmpoutT = tex1Dfetch(SYNCOUTGOING, j);
								tmpT[0] = (char)(tmpoutT >> 24);
								tmpT[1] = (char)(tmpoutT >> 16);
								tmpT[2] = (char)(tmpoutT >> 8);
								tmpT[3] = (char)tmpoutT;
								x = j;	
							}			
							CudaNewStateV(&localstateV, InvthreadgroupID, &j, &tmpj, tmpT, tex1Dfetch(TRANSEBYTES, InvthreadgroupID), PEventEncodeBits);
							//tmpcount++;
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;

						}
						nosync = 0;
						lessthanall = 0;
						m=0;

						for(i=0; i<PLTSNum; i++){
							if(i == InvthreadgroupID)
								continue;
							if(SynEventInteractive[i + vthreadgroupID * PLTSNum] == EMPTYVECT32)
							{
								nosync++;
								continue;
							}	

							if(SynEventInteractive[i + vthreadgroupID * PLTSNum] <= maxtransevent){  //if bigger than the maxtransevent of local, no need to compare as it's impossible to sync
								if(SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] > SynEventInteractive[i + vthreadgroupID * PLTSNum]){
									m++;

								}else if (SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] == SynEventInteractive[i + vthreadgroupID * PLTSNum]){
									if(needsyndupdetect[vthreadgroupID] == false)
										needsyndupdetect[vthreadgroupID] = true;
									//GENERATE SYNC STATE V.......
									SynTwoStates(&localstateV, SynStateInteractive[i + vthreadgroupID * PLTSNum], i);
									syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
									ifmatch = true;
								}else
									lessthanall++;
							}
						}
					
						if(nosync == PLTSNum - 1){
							break;
						}

						if(lessthanall == PLTSNum -1){
							m = 0;
							syncduplicate[InvthreadgroupID + vthreadgroupID*PLTSNum] = true;
							continue;
						}
	

						if(syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum])
							m = 0;

						if(needsyndupdetect[vthreadgroupID] && InvthreadgroupID == 0){   //duplicate elimination after synchronization, so just one synchronized result will be copied to hashtable.
							for(i = 0; i < PLTSNum; i++){
								if(syncduplicate[i + vthreadgroupID * PLTSNum]){
									for(k = 0; k < i; k++)
									{
										if(SynEventInteractive[k + vthreadgroupID * PLTSNum] == SynEventInteractive[i + vthreadgroupID * PLTSNum]){
											break;
										}
									}
									if(k == i){
										syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;
									}
								}
							}
						}

						if(ifmatch && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] == false){
							//hash copy to table
							insertRecord = localstateV;

							if(!Ifcollisionhappens)
							{
								if(CudaHashStore(insertRecord,  PLTSNum, RecordTable, blockDim.x, &kickoutRecord))
									Ifcollisionhappens = true;
								outgoingcount++;
							}
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
							if(Ifcollisionhappens){
								for(k = 511; k >= 0; k--){
									if(kickoutRecord != EMPTYVECT32){
										if(atomicCAS(&(RecordTable[k]), EMPTYVECT32, kickoutRecord)==EMPTYVECT32){
											kickoutRecord = EMPTYVECT32;
											break;
										}
									}else{
										if(atomicCAS(&(RecordTable[k]), EMPTYVECT32, localstateV) == EMPTYVECT32){
											break;
										}
									}
								}
							}
						}

						if(!ifmatch && m == 0){
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
						}
						localstateV = GroupStore[vthreadgroupID];

					}
				}
				if(outgoingcount == 0 && !ifglobaldup)
					ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			
			}else{
				ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			}

			if(InvthreadgroupID == 0&&!ifglobaldup &&!Ifcollisionhappens){
				for(i = 0; i < PLTSNum; i++){
					if(!ifnooutgoing[i + vthreadgroupID * PLTSNum])
						break;
				}

				if(i == PLTSNum){
					printf("tgid%d find deadlock\n", vthreadgroupID);
					IFDeadlockDetected = true;
				}
			}
		}
		CudaInterBlocksSyn(gridDim.x);

		if(IFDeadlockDetected)
			break;

		if(GroupStore[vthreadgroupID] != EMPTYVECT32){
			if(InvthreadgroupID == 0&&!Ifcollisionhappens&&!ifglobaldup){
				//copy visited state to gl)obal memory
				CudaVisitedGlobalHashstore(GlobalVisitedHash, globalbuckethash, visitedstore, GroupStore[vthreadgroupID], PLTSNum);
				if(InvthreadgroupID == 0){
					GroupStore[vthreadgroupID] = EMPTYVECT32;
				}
			}
		}
		__syncthreads();
		if(Ifcollisionhappens){
				if(haveChild)
					cudaDeviceSynchronize();

				//if(IFDeadlockDetected)
				//	break;
				//load new kernel, copy data back
				unsigned int myareacount = 0;
			

				globalbuckethash = Buckethash((unsigned int)(blockIdx.x)) + openvisitedborder;

				if(InWarptid == 0){
					for(m = Warpid*32; m<(Warpid + 1)*32; m++){
						//for(k = 0; k < HashNum; k++){
							if(RecordTable[m] != EMPTYVECT32)
								myareacount++;
						//}
					}
				
					WarpCBindex[Warpid].beginindex = atomicAdd(&GlobalbucketIndex[globalbuckethash], myareacount);
					WarpCBindex[Warpid].endindex = WarpCBindex[Warpid].beginindex + myareacount;
					atomicAdd(&GlobalbucketCount[globalbuckethash], myareacount);
				}

				storeposition = WarpCBindex[Warpid].beginindex + InWarptid;
				//for(m = 0; m < HashNum; m++){
					if(RecordTable[ Warpid * 32 + InWarptid] != EMPTYVECT32){
						GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable[Warpid * 32 + InWarptid];
						RecordTable[Warpid * 32 + InWarptid] = EMPTYVECT32;
						storeposition+=32;
					}
				//}

				if(storeposition < WarpCBindex[Warpid].endindex)
				{
					//for(m = 0; m < HashNum; m++){
						for(k = Warpid*32; k<(Warpid+1)*32; k++){
							if(RecordTable[k] != EMPTYVECT32){
								kickoutRecord = RecordTable[k];
								if(atomicCAS(&(RecordTable[k]), RecordTable[k], EMPTYVECT32) == kickoutRecord){
									GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
									storeposition+=32;
								}
							}
						}
					//}
				}

				//for the elements larger than 512, to be expanded........

				//launch new kernel
				if(Inblocktid == launchtime){
					if(GlobalbucketCount[globalbuckethash]*PLTSNum % 512 == 0){
						m = (GlobalbucketCount[globalbuckethash]*PLTSNum) / 512;
					}else{
						m = (GlobalbucketCount[globalbuckethash]*PLTSNum) / 512 + 1;
					}
					StartNum = GlobalbucketCount[globalbuckethash]*PLTSNum;
					if(launchtime > 0){
						i=0;
						for(k = communicationGstore[blockIdx.x].beginindex; k < communicationGstore[blockIdx.x].endindex; k++){
							i+=GlobalbucketCount[k];
						}
						StartNum += i;
						if(i*PLTSNum % 512 == 0){
							m += i*PLTSNum / 512;
						}else{
							m += (i*PLTSNum / 512 +1);
						}
					}
					dim3 cgridstructure(m,1,1);
					dim3 cblockstructure(512,1,1);
					CUDADeadlockBFSVerifyChild<<<cgridstructure, cblockstructure>>>(blockIdx.x, globalbuckethash, communicationGstore, PG_AllLTS, PG_AllStates, PG_AllTransitions, PG_AllSynctransitions, PG_LTSStateEncodeBits, PEventEncodeBits, OutgoingTEbytes, PGBucketNum, PLTSNum, StartNum );
					launchtime++;
					haveChild = true;
				}
			
		}
		

		__syncthreads();

		if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID] == EMPTYVECT32){
			//got new stateV
			localstateV = EMPTYVECT32;
			ifgetnewstatev = false;
			
			//for(j = 0; j < HashNum; j++){
				for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
					if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable[i]), EMPTYVECT32)) != EMPTYVECT32)
					{
						ifgetnewstatev = true;
						break;
					}
				}

				//if(ifgetnewstatev == true)
				//	break;
				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
						if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable[i]), EMPTYVECT32)) != EMPTYVECT32)
						{
							ifgetnewstatev = true;
							break;
						}
					}
				}
				//if(ifgetnewstatev == true)
				//	break;
			//}
			
		}



		__syncthreads();
		
		if(Inblocktid == launchtime - 1 && ifgetnewstatev == false){
			for(nonewcount = 0; nonewcount < vthreadgroupnuminblock; nonewcount++){
				if(GroupStore[nonewcount] != EMPTYVECT32){
					break;
				}
			}
			if(nonewcount == vthreadgroupnuminblock){
				cudaDeviceSynchronize();
				haveChild = false;
			}
			
		}


		__syncthreads();

		if(nonewcount == vthreadgroupnuminblock){
			//get new state again, if no, block stop.
			if(InvthreadgroupID == 0){
				//got new stateV
				if(vthreadgroupID < GlobalbucketCount[communicationGstore[blockIdx.x].beginindex])
				{
					globalbuckethash =  communicationGstore[blockIdx.x].beginindex;
					storeposition = vthreadgroupID;
				}

				//layer = communicationlayer[blockIdx.x];
				
				GroupStore[vthreadgroupID] = GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition]; 
				if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID] == EMPTYVECT32)
				{
					ifblocknostate = true;
				}
					

			}

			__syncthreads();


			if(ifblocknostate)
				break;

			if(communicationcollision[blockIdx.x] && Inblocktid == launchtime){
				//need more blocks
				k = 0;
				for(m = communicationGstore[blockIdx.x].beginindex; m < communicationGstore[blockIdx.x].endindex; m++){
					k += GlobalbucketCount[m];
				}
				k -= vthreadgroupnuminblock;
				StartNum = k;
				if(k*PLTSNum % 512 == 0)
					m = (k*PLTSNum)/512;
				else
					m = (k*PLTSNum)/512 + 1;

				dim3 gridstruc(m,1,1);
				dim3 blockstruc(512,1,1);
				CUDADeadlockBFSVerifyChild<<<gridstruc, blockstruc>>>(blockIdx.x, globalbuckethash, communicationGstore, PG_AllLTS, PG_AllStates, PG_AllTransitions, PG_AllSynctransitions, PG_LTSStateEncodeBits, PEventEncodeBits, OutgoingTEbytes, PGBucketNum, PLTSNum, StartNum );
				launchtime++;
				haveChild=true;
			}
		}
			
		
		
	}while(!IFDeadlockDetected);

	CudaInterBlocksSyn(gridDim.x);
	if(!IFDeadlockDetected && Ingridtid == 0){
		*G_RESULT = 1;
	}else{
		*G_RESULT = 0;
	}
}


//void NewStateV(unsigned int * targetV, int tindex, int * index, int *count,  unsigned char* OutgoingTs, unsigned int * bitwidth, unsigned int OutGTbyte, unsigned int EEncode)
//{
//	unsigned int tmp = *targetV;
//	unsigned int tostate = 0;
//	int newsbeginbit = 0, endbit;
//	unsigned int Secode = bitwidth[tindex];
//
//	int i,j,replacebeginbyte, replaceendbyte;
//
//	replacebeginbyte = *count * OutGTbyte;
//	replaceendbyte =  (*count + 1)*OutGTbyte;
//
//	//if(EEncode < 8){
//	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] << EEncode;
//	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] >> EEncode; //event
//	//}else{
//	//	replacebeginbyte++;
//	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] << EEncode - 8;
//	//	OutgoingTs[replacebeginbyte] = OutgoingTs[replacebeginbyte] >> EEncode - 8;
//	//}
//
//	for(i = 0; i < tindex; i++){
//		newsbeginbit += bitwidth[i];
//	}
//
//	endbit = newsbeginbit + bitwidth[tindex];
//
//	if(Secode == 8){
//		tostate = (int) OutgoingTs[replaceendbyte - 1];
//		tostate = tostate << (31 - endbit);
//
//	}else{
//		tostate = 0;
//
//		for( i = replaceendbyte - 1; i > replacebeginbyte; i--)
//			tostate = tostate | (OutgoingTs[i] << 8 * (replaceendbyte - 1 - i));
//
//		tostate = tostate << (31-Secode);
//		tostate = tostate >> (31-Secode);
//		tostate = tostate << (31-endbit);
//
//	}
//	
//	i = tmp >> (endbit + Secode);
//	i = i << (endbit + Secode);
//	j = tmp << (newsbeginbit + Secode);
//	j = j >> (newsbeginbit + Secode);
//
//	* targetV = (int) (i | j | tostate);
//
//	if((EEncode+Secode)*(*count + 1) > 32){
//		* index += 1;
//		*count = 0;
//	}else
//		(*count)++;
//}
//
//void DecodeTransitions(unsigned int * outgoingT, int beginindex, int count, unsigned int * Tostate, unsigned int * Tevent, unsigned int OutGTe, unsigned int Statebitwidth)
//{
//	int i, j;
//	unsigned int tmp;
//	unsigned int startbyte, endbyte;
//	startbyte = (count * OutGTe)%4;
//	endbyte = ((count + 1)*OutGTe)%4;
//
//	if(endbyte == 0)
//		endbyte = 4;
//
//	tmp = outgoingT[beginindex];
//
//	tmp = tmp << (startbyte - 1);
//	tmp = tmp >> (startbyte + 3 - endbyte); 
//
//	*Tostate = (tmp << 31 - Statebitwidth) >> (31- Statebitwidth);
//	*Tevent = tmp >> Statebitwidth;
//}
//
//
//
//bool GetAllsuccessors(unsigned int * AllLTS, unsigned int * Allstates, unsigned int * Alltransitions, unsigned int ltsindex, unsigned int sindex, Nodemark * result)
//{
//	unsigned int statesbegin, transbegin, transborder, syncbegin;
//	statesbegin = AllLTS[ltsindex];
//	transbegin = Allstates[statesbegin + sindex];
//	transborder = Allstates[statesbegin + sindex + 1];
//
//	if(transbegin == 0 && (ltsindex != 0 || sindex !=0))
//		return false;
//
//	result->beginInt = transbegin;
//	result->endInt = transborder - 4;
//
//	result->synbeginInt = Alltransitions[transborder - 1] | Alltransitions[transborder - 2] | Alltransitions[transborder - 3] | Alltransitions[transborder - 4];
//
//	transborder = Allstates[statesbegin + sindex + 2];
//
//	syncbegin = Alltransitions[transborder - 1] | Alltransitions[transborder - 2] | Alltransitions[transborder - 3] | Alltransitions[transborder - 4];
//
//	result->synendInt = syncbegin - 1;
//	return true;
//}
//
//unsigned int GetStateinVec(int index, unsigned int svec, unsigned int * stateencodebits)
//{
//	int sbeginbit, sendbit;
//	unsigned int ltsid;
//
//	sbeginbit = 0;
//	sendbit = 0;
//
//	for(int i = 0; i < index; i++){
//		sbeginbit += stateencodebits[i]; 
//	}
//	sendbit = sbeginbit + stateencodebits[index] - 1;
//	svec  = svec << sbeginbit; 
//	svec = svec >> (sbeginbit + 31 - sendbit);
//	ltsid = svec;
//	return ltsid;
//
//}
//
//int HostGenerateStateSpace(int LTSNum, unsigned int * H_AllLTS, unsigned int * H_AllStates, unsigned int * H_AllTransitions, unsigned int * H_AllSynctrans, unsigned int ** RecordList, unsigned int RequestNum, unsigned int H_InitialStateV, unsigned int * H_LTSStateEncodeBits, unsigned int * OutgoingTEbytes, unsigned int HEventEncodeBits)
//{
//	int i,j,m,k;
//	int SuccessorCount;
//	queue<unsigned int> Taskqueue;
//	set<unsigned int> Taskset;
//	vector<unsigned int> Syncqueue;
//	vector<unsigned int>::iterator Syncit;
//
//	vector<unsigned int> Syncevents;
//	vector<unsigned int>::iterator ITS;
//	bool ifexist;
//
//	queue<unsigned int> VisitedS;
//
//	unsigned int newStateV;
//	unsigned int * succStateV;
//	unsigned int * tmpStateV;
//	unsigned int newState;
//	unsigned int belonglts;
//	unsigned int transevent;
//
//	unsigned int *tmp;
//
//	unsigned int tmpcount;
//	unsigned int tmpoutT;
//	unsigned char tmpT[4];
//
//	int x,y;
//	
//	bool ifoutgoing;
//	int ifoutgoingcount;
//	Nodemark allsucc;
//	
//	SuccessorCount = 1;
//	Taskqueue.push(H_InitialStateV);
//	while(SuccessorCount < RequestNum){
//		newStateV = Taskqueue.front();
//		ifoutgoingcount = 0;
//		for(i = 0; i < LTSNum; i++){
//			ifoutgoing = false;
//			GetStateinVec(i, newStateV, &newState);
//			ifoutgoing = GetAllsuccessors(H_AllLTS, H_AllStates, H_AllTransitions, belonglts, newState, &allsucc);
//			if(!ifoutgoing){
//				ifoutgoingcount++;
//				continue;
//			}
//
//			m = allsucc.beginInt;
//			x = -1;
//			y = 0;
//			while(m < allsucc.endInt){
//				succStateV = new unsigned int[1];
//				
//				if(x != m){
//					tmpoutT = H_AllTransitions[m];
//					tmpT[0] = (char)(tmpoutT >> 24);
//					tmpT[1] = (char)(tmpoutT >> 16);
//					tmpT[2] = (char)(tmpoutT >> 8);
//					tmpT[3] = (char)tmpoutT;
//					x = m;	
//				}
//				NewStateV(succStateV, i, &m, &y, tmpT, H_LTSStateEncodeBits, OutgoingTEbytes[i], HEventEncodeBits );
//				
//				if(Taskset.insert(*succStateV).second){
//					Taskqueue.push(*succStateV);
//					SuccessorCount++;
//				}
//			}
//
//			k = allsucc.synbeginInt;
//			tmpcount = 0;
//			x = -1;
//			y = 0;
//			while(k < allsucc.synendInt){
//				succStateV = new unsigned int[1];
//
//				DecodeTransitions(H_AllSynctrans, k, tmpcount, &newState, &transevent, OutgoingTEbytes[belonglts], H_LTSStateEncodeBits[i]);
//
//				if(x != k){
//					tmpoutT = H_AllSynctrans[k];
//					tmpT[0] = (char)(tmpoutT >> 24);
//					tmpT[1] = (char)(tmpoutT >> 16);
//					tmpT[2] = (char)(tmpoutT >> 8);
//					tmpT[3] = (char)tmpoutT;
//					x = k;	
//				}			
//				NewStateV(succStateV, i, &k, &y, tmpT, H_LTSStateEncodeBits, OutgoingTEbytes[i], HEventEncodeBits);
//
//				tmpcount++;
//				j = 0;
//				for(ITS = Syncevents.begin(); ITS < Syncevents.end(); ++ITS){
//					if(*ITS == transevent){
//						ifexist = true;
//						break;
//					}else
//						j++;
//				}
//				if(ifexist){
//					
//					tmpStateV = (unsigned int *)&(Syncqueue[j]);
//					SynTwoStatesCPU(tmpStateV, *succStateV, i, newStateV, H_LTSStateEncodeBits);
//					
//				}else{
//					Syncevents.push_back(transevent);
//					Syncqueue.push_back(*succStateV);
//					SuccessorCount++;
//				}	
//			}
//			for(Syncit = Syncqueue.begin(); Syncit != Syncqueue.end(); Syncit++) {
//				Taskqueue.push(*Syncit);
//			}
//			Syncqueue.clear();
//		}
//		if(ifoutgoingcount == LTSNum){
//			return -1;
//		}
//		
//	}
//
//	*RecordList = new unsigned int[SuccessorCount];
//	for(i = 0; i < SuccessorCount; i++){
//		(*RecordList)[i] = Taskqueue.front();
//		Taskqueue.pop();
//	}
//
//	return SuccessorCount;
//}

void CallCudaBFS(unsigned int * AllLTS, unsigned int * AllStates, unsigned int * AllTransitions, unsigned int* AllSyncTrans, unsigned int H_InitialSV, unsigned int * H_LTSStateEncodeBits, unsigned int LTSNum,unsigned int AllLTSStateNum, unsigned int AllTransLength, unsigned int AllSyncTransLength, unsigned int EventEncodeBits,  unsigned int * OutgoingTEbytes)
{
	int i,j;
	unsigned int * G_AllLTS;
	unsigned int * G_AllStates;
	unsigned int * G_AllTransitions;
	unsigned int * G_AllSyncTrans;  //all trans with sync events.
	//unsigned int * G_InitialStateV;
	unsigned int * G_OutgoingTEbytes;
	unsigned int * G_LTSStateEncodeBits;
	unsigned int * G_DetectResult;
	//Choose to generate some statevectors firstly in CPU---OPTIONAL
	unsigned int * G_Startlist; 
	
	//Choose to generate some statevectors firstly in CPU---OPTIONAL
	unsigned int * H_Startlist;
	unsigned int * H_Result;
	unsigned int H_startsize;
	unsigned int * LTSStateNum = new unsigned int[LTSNum];

	unsigned int Startblocknum;
	unsigned int Startthreadnum1block;
	unsigned int Startthreadgroupnum;

	unsigned int H_GlobalbucketNum;

	int * parameters = new int[2];
	parameters[0]=4;
	parameters[1]=12;
	//unsigned int * G_GlobalbucketNum;

	int rv[10];
	srand(time(NULL));
	for(i = 0; i < 10; i++){
		rv[i] = rand();
	}

	cudaSetDevice(0);
	
	H_Result = new unsigned int[1];
	Startthreadnum1block = 512;
	Startblocknum = 1;
	//Initialize Startlist
	Startthreadgroupnum = (((Startthreadnum1block/32)/LTSNum)*(Startthreadnum1block/32))*Startblocknum;  //initial value, not the final one?
	
	
	//i = HostGenerateStateSpace(LTSNum, AllLTS,AllStates,AllTransitions, AllSyncTrans, &H_Startlist, 1, H_InitialSV,H_LTSStateEncodeBits, OutgoingTEbytes,  EventEncodeBits);
	if(i > 0){
		j = i * LTSNum;
		if(i > Startthreadgroupnum){
			Startthreadgroupnum = i;
			Startblocknum = Startthreadgroupnum/(Startthreadnum1block/LTSNum);
		}
	}else if(i == -1){
		cout<<"deadlock being detected";
		exit(0);
	}

	Startthreadgroupnum = 1;
	H_Startlist = new unsigned int[1];
	H_Startlist[0] = H_InitialSV;
	H_GlobalbucketNum = LTSNum * 2;
    	cudaMalloc((void **)&G_AllLTS, sizeof(unsigned int) * LTSNum);
	cudaMalloc((void **)&G_AllStates, sizeof(unsigned int) * (AllLTSStateNum+1));
	cudaMalloc((void **)&G_AllTransitions, sizeof(unsigned int) * AllTransLength);
	cudaMalloc((void **)&G_AllSyncTrans,sizeof(unsigned int) * AllSyncTransLength);
	cudaMalloc((void **)&G_OutgoingTEbytes, sizeof(unsigned int) * LTSNum);
	cudaMalloc((void **)&G_LTSStateEncodeBits, sizeof(unsigned int) * LTSNum);
	cudaMalloc((void **)&G_Startlist, sizeof(unsigned int) * Startthreadgroupnum);
	//cudaMalloc((unsigned int *)&G_InitialStateV, sizeof(int));

	cudaMalloc((void **)&G_DetectResult, sizeof(unsigned int));

	cudaMemcpy(G_AllLTS, AllLTS, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllStates, AllStates, sizeof(unsigned int) * (AllLTSStateNum+1), cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllTransitions, AllTransitions, sizeof(unsigned int) * AllTransLength, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllSyncTrans, AllSyncTrans, sizeof(unsigned int) * AllSyncTransLength, cudaMemcpyHostToDevice);
	//cudaMemcpy(G_InitialStateV, &H_InitialSV, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(G_LTSStateEncodeBits, H_LTSStateEncodeBits, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_Startlist, H_Startlist, sizeof(unsigned int) * Startthreadgroupnum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_OutgoingTEbytes, OutgoingTEbytes, sizeof(unsigned int)*LTSNum, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(LA1, &rv[0], sizeof(int));
	cudaMemcpyToSymbol(LB1, &rv[1], sizeof(int));
	cudaMemcpyToSymbol(LA2, &rv[2], sizeof(int));
	cudaMemcpyToSymbol(LB2, &rv[3], sizeof(int));
	cudaMemcpyToSymbol(LA3, &rv[4], sizeof(int));
	cudaMemcpyToSymbol(LB3, &rv[5], sizeof(int));
	cudaMemcpyToSymbol(LA4, &rv[6], sizeof(int));
	cudaMemcpyToSymbol(LB4, &rv[7], sizeof(int));
	cudaMemcpyToSymbol(BUCA, &rv[8], sizeof(int));
	cudaMemcpyToSymbol(BUCB, &rv[9], sizeof(int));

	for(i = 0; i < 8; i++){
		rv[i] = rand();
	}
	i = 512;
	cudaMemcpyToSymbol(GA1, &rv[0], sizeof(int));
	cudaMemcpyToSymbol(GB2, &rv[1], sizeof(int));
	cudaMemcpyToSymbol(GA2, &rv[2], sizeof(int));
	cudaMemcpyToSymbol(GB2, &rv[3], sizeof(int));
	cudaMemcpyToSymbol(GA3, &rv[4], sizeof(int));
	cudaMemcpyToSymbol(GB3, &rv[5], sizeof(int));
	cudaMemcpyToSymbol(GA4, &rv[6], sizeof(int));
	cudaMemcpyToSymbol(GB4, &rv[7], sizeof(int));

	cudaMemcpyToSymbol(TableSize, &i, sizeof(int));

	cudaMemcpyToSymbol(HashNum, &parameters[0],sizeof(int));
	cudaMemcpyToSymbol(IterationTime, &parameters[1], sizeof(int));

	//bind data to texture
	cudaBindTexture(NULL, LTSOFFSET, G_AllLTS, sizeof(unsigned int)*LTSNum);
	cudaBindTexture(NULL, STATEOFFSET, G_AllStates, sizeof(unsigned int)*(AllLTSStateNum+1));
	cudaBindTexture(NULL, OUTGOINGDETAIL, G_AllTransitions, sizeof(unsigned int)*AllTransLength);  //how texture memory can accelerate the access rate need to be explored
	cudaBindTexture(NULL, SYNCOUTGOING, G_AllSyncTrans, sizeof(unsigned int)*AllSyncTransLength);
	cudaBindTexture(NULL, LTSSTATEBITS, G_LTSStateEncodeBits, sizeof(unsigned int)* LTSNum);
	cudaBindTexture(NULL, TRANSEBYTES, G_OutgoingTEbytes, sizeof(unsigned int)*LTSNum);

	dim3 g(1,1,1);
	dim3 b(512,1,1);
	H_startsize = 1;

	//test time
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);//record time

	CUDADeadlockBFSVerify<<<g, b, 5120*sizeof(unsigned int)>>>( G_AllLTS, G_AllStates, G_AllTransitions, G_AllSyncTrans, G_Startlist, G_LTSStateEncodeBits, EventEncodeBits, G_OutgoingTEbytes, LTSNum, G_DetectResult, H_GlobalbucketNum, AllLTSStateNum, H_startsize);

	cudaEventRecord(stop,0);//record time
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"cuda cost time(ms):"<<elapsedTime<<endl;
	
	cudaMemcpy(H_Result, G_DetectResult, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cout<<"Result"<<*H_Result<<endl;

	cudaUnbindTexture(LTSOFFSET);
	cudaUnbindTexture(STATEOFFSET);
	cudaUnbindTexture(OUTGOINGDETAIL);
	cudaUnbindTexture(SYNCOUTGOING);
	cudaUnbindTexture(LTSSTATEBITS);
	cudaUnbindTexture(TRANSEBYTES);

	cudaFree(G_AllLTS);
	cudaFree(G_AllStates);
	cudaFree(G_AllTransitions);
	//udaFree(GlobalBuckets);
	//cudaFree(GlobalOpenHash);
	//cudaFree(GlobalVisitedHash);
	free(AllLTS);
	free(AllStates);
	free(AllTransitions);
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

	CallCudaBFS(AllLTS,AllStates,AllTransitions,AllSyncTrans,InitialV,LTSStateEncodebits, LTSNum, StatesNUM,AlltransNum,AllsynctransNum,EventEncodebits, OutgoingTEbytes);

}


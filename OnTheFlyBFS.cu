
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_35_atomic_functions.h"

#include <iostream>
#include <string>
#include <time.h>
#include <queue>
#include <set>
#include <list>
using namespace std;

texture<unsigned int, 1, cudaReadModeElementType> LTSOFFSET;  //1 means 1-dimension
texture<unsigned int, 1, cudaReadModeElementType> STATEOFFSET;
texture<unsigned int, 1, cudaReadModeElementType> OUTGOINGDETAIL;
texture<unsigned int, 1, cudaReadModeElementType> STATEENCODE;

__constant__ int LA1;
__constant__ int LA2;
__constant__ int LA3;
__constant__ int GA1;
__constant__ int GA2;
__constant__ int GA3;
__constant__ int LB1;
__constant__ int LB2;
__constant__ int LB3;
__constant__ int GB1;
__constant__ int GB2;
__constant__ int GB3;
__constant__ int BUCA;
__constant__ int BUCB;
__constant__ unsigned int PrimeNum = 334214459;

static const unsigned int EMPTYVECT32 = 0x7FFFFFFF;
static const unsigned int P = 334214459;
static const unsigned int blocksize = 512;

class LocalRecord{
public:
	char localmark;  //record the BFS layer in Shared Memory
	char toevent;
	int statevector;

	LocalRecord(){
		statevector = 0x7FFFFFFF;
		localmark = 0x00;

	}
	~LocalRecord();
};

class Bucket{
public:
	unsigned int beginindex;
	unsigned int endindex;

	Bucket(){
		beginindex = 0;
		endindex = 0;
	}
	~Bucket();
};

class Nodemark{
public:
	unsigned int beginbyte;
	unsigned int endbyte;
	unsigned int synbeginbyte;
	unsigned int synendbyte;

	Nodemark(){
		beginbyte = 0;
		endbyte = 0;
	}
	~Nodemark();
};

__device__ LocalRecord *GlobalOpenHash;  
__device__ Bucket *GlobalBuckets;
__device__ unsigned int GlobalBucketNum;
__device__ LocalRecord *GlobalVisitedHash;  //here, for visited stateV, use hash to store back to global memory. While this hash doesn't kick the original one. For open stateV, use buckets hash.
//__device__ unsigned int GlobalVisitedHashoffset[3];

__device__ volatile unsigned int * GlobalBucketsCount;

__device__ unsigned int OpenSize;

__device__ bool IFDeadlockDetected;

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

__device__ unsigned int Globalhash1(unsigned int k)
{

}

__device__ unsigned int Globalhash2(unsigned int k)
{

}

__device__ unsigned int Globalhash3(unsigned int k)
{

}

__device__ unsigned int Localhash1(unsigned int k)
{
	;
}

__device__ unsigned int Localhash2(unsigned int k)
{
	;
}

__device__ unsigned int Localhash3(unsigned int k)
{
	;
}


__global__ unsigned int CudaGetStateinVec(int index, unsigned int svec, unsigned int * stateencodebits)
{
	int sbeginbit, sendbit;
	unsigned int ltsid;

	sbeginbit = 0;
	sendbit = 0;

	for(int i = 0; i < index; i++){
		sbeginbit += stateencodebits[i]; 
	}
	sendbit = sbeginbit + stateencodebits[index] - 1;
	svec  = svec << sbeginbit; 
	svec = svec >> (sbeginbit + 31 - sendbit);
	ltsid = svec;
	return ltsid;

}

__global__ bool CudaGetAllsuccessors(unsigned int * AllLTS, unsigned int * Allstates, unsigned char * Alltransitions, unsigned int ltsindex, unsigned int sindex, Nodemark * result)
{
	unsigned int statesbegin, transbegin, transborder, syncbegin;
	statesbegin = AllLTS[ltsindex];
	transbegin = Allstates[statesbegin + sindex];
	transborder = Allstates[statesbegin + sindex + 1];

	result->beginbyte = transbegin;
	result->endbyte = transborder - 4;

	result->synbeginbyte = Alltransitions[transborder - 1] | Alltransitions[transborder - 2] | Alltransitions[transborder - 3] | Alltransitions[transborder - 4];

	transborder = Allstates[statesbegin + sindex + 2];

	syncbegin = Alltransitions[transborder - 1] | Alltransitions[transborder - 2] | Alltransitions[transborder - 3] | Alltransitions[transborder - 4];

	result->synendbyte = syncbegin - 1;

}

__global__ void CudaNewStateV(unsigned int * targetV, int tindex, int * index, unsigned char * Atrans, unsigned int * bitwidth, unsigned int bytewidth)
{
	unsigned int tmp = *targetV;
	unsigned int tostate = 0;
	int newsbeginbit = 0, endbit;

	unsigned char tmpbyte1[4], tmpbyte2[4];
	int insidebytes, insidebytee;

	for(int i = 0; i < tindex; i++){
		newsbeginbit += bitwidth[i];
	}

	endbit = newsbeginbit + bitwidth[tindex];

	insidebytes = newsbeginbit / 8;
	insidebytee = endbit / 8;

	tmpbyte1[0] = (char) tmp;
	tmpbyte1[1] = (char) tmp >> 8;
	tmpbyte1[2] = (char) tmp >> 16;
	tmpbyte1[3] = (char) tmp >> 24;

	if(bytewidth == 1){
		tostate = (int) Atrans[*index];
		tostate = tostate << (31 - endbit);

	}else{
		tmpbyte2[0] = (char) 0;
		tmpbyte2[1] = (char) 0;
		tmpbyte2[2] = (char) 0;
		tmpbyte2[3] = (char) 0;

		for(int i = *index; i < *index + bytewidth; i++){
			tmpbyte1[i-*index] = Atrans[i];
		}

		tostate = (int) (tmpbyte2[0] | tmpbyte2[1] << 8 | tmpbyte2[2] << 16 | tmpbyte2[3] << 24);

		tostate = tostate << (31-endbit);

	}

	for(int j = insidebytes; j < insidebytee; j++){
		tmpbyte1[j] = (char) (tostate >> 8*(4-j));

		tmpbyte2[j] = tmpbyte2[j] >> (8 - newsbeginbit % 8);
		tmpbyte2[j] = tmpbyte2[j] << (8 - newsbeginbit % 8);

		tmpbyte2[j] = tmpbyte2[j] | tmpbyte1[j];
	}
	
	* targetV = (int) (tmpbyte1[0] | tmpbyte1[1] << 8 | tmpbyte1[2] << 16 | tmpbyte1[3] << 24);
	*index += bytewidth;
}

__global__ void CudaDecodeTransitions(unsigned char * outgoingT, int beginindex, unsigned int * Tostate, unsigned int * Tevent, unsigned int Eventwidth, unsigned int Statewidth)
{
	unsigned int stateendbyte, eventendbyte;
	stateendbyte = beginindex + Eventwidth + Statewidth;
	eventendbyte = beginindex + Eventwidth;
	int i;
	unsigned char tmp[4];
	for(i = 0; i < 4; i++){
		tmp[i] = 0x00;
	}
	for(i = beginindex; i < eventendbyte; i++){
		tmp[eventendbyte - i] = outgoingT[i];
	}

	for(i = 4 - (eventendbyte - beginindex) ; i < 4; i++){
		*Tevent = *Tevent | tmp[i] << (i - 4 + (eventendbyte - beginindex)) * 8; 
	}
	for(i = eventendbyte; i < stateendbyte; i++){
		tmp[stateendbyte - i] = outgoingT[i];
	}
	for(i = 4 - (stateendbyte - eventendbyte) ; i < 4; i++){
		*Tostate = *Tostate | tmp[i] << (i - 4 + (stateendbyte - eventendbyte)) * 8; 
	}
}

__global__ unsigned int CudaGenerateKey(unsigned int KV, unsigned int *ecbit, int snum)
{

}

__device__ int SynTwoStates(unsigned int * s1, int s2idx, unsigned int evtid, unsigned int svec, unsigned char * alltrans, unsigned char * allsynctrans)
{
	unsigned int localstate;
	localstate = CudaGetStateinVec(s2idx, svec, PG_LTSStateEncodeBits);
	belonglts = InvthreadgroupID;
	ifanyoutgoing = CudaGetAllsuccessors(PG_AllLTS, PG_AllStates, PG_AllTransitions, belonglts, localstate, &SuccessorMark);
}

__global__ bool CudaHashStore(LocalRecord beginHV, unsigned int layer, unsigned int * EBits, unsigned int PLTSNum, LocalRecord * T1, LocalRecord * T2, LocalRecord * T3, LocalRecord * RkickoutRecord){
	unsigned int localKey, localhash;
	LocalRecord kickoutRecord;
	char tmp;

	localKey = CudaGenerateKey(beginHV.statevector, EBits, PLTSNum);
	localhash = Localhash1(localKey);
	if(!atomicCAS(&(T1[localhash].statevector), beginHV.statevector, EMPTYVECT32)){
		if(T1[localhash].statevector == beginHV.statevector){
			return false;
		}else{
			kickoutRecord.statevector = atomicExch(&(T1[localhash].statevector), beginHV.statevector);
			kickoutRecord.localmark = T1[localhash].localmark;
			T1[localhash].localmark = beginHV.localmark;

			localKey = CudaGenerateKey(kickoutRecord.statevector, EBits, PLTSNum);
			localhash = Localhash2(localKey);
			if(atomicCAS(&(T2[localhash].statevector), kickoutRecord.statevector, EMPTYVECT32)){
				T2[localhash].localmark = kickoutRecord.localmark;
			}else{
				if(T2[localhash].statevector == kickoutRecord.statevector){
					return false;
				}else{
					kickoutRecord.statevector = atomicExch(&(T2[localhash]), kickoutRecord.statevector);
					tmp = T2[localhash].localmark;
					T2[localhash].localmark = kickoutRecord.localmark;
					kickoutRecord.localmark = tmp;

					localKey = CudaGenerateKey(kickoutRecord.statevector, EBits, PLTSNum);
					localhash = Localhash3(localKey);
					if(atomicCAS(&(T3[localhash].statevector), kickoutRecord.statevector, EMPTYVECT32)){
						T3[localhash].localmark = (char)(layer + 1);
					}else{
						if(T3[localhash].statevector == kickoutRecord.statevector){
							return;
						}else{
							//kick out the one in localhash3
							kickoutRecord.statevector = atomicExch(&(T3[localhash]), kickoutRecord.statevector);
							tmp = T3[localhash].localmark;
							T3[localhash].localmark = kickoutRecord.localmark;
							kickoutRecord.localmark = tmp;

							localKey = CudaGenerateKey(kickoutRecord.statevector, EBits, PLTSNum);
							localhash = Localhash1(localKey);
							if(atomicCAS(&(T1[localhash].statevector), kickoutRecord.statevector, EMPTYVECT32)){
								T1[localhash].localmark = (char)(layer +1);
							}else{
								return true;
							}
						}
					}
				}
			}
		}
	}else{
		T1[localhash].localmark = (char)(layer+1);
	}
}

__global__ bool CudaVisitedGlobalHashcal(LocalRecord * HT, Bucket belongBucket, unsigned int hkey, LocalRecord insertrecord, unsigned int * hashresult){
	unsigned int hashposition1, hashposition2, hashposition3;

	hashposition1 = Globalhash1(hkey);
	hashposition2 = Globalhash2(hkey);
	hashposition3 = Globalhash3(hkey);

	if(HT[belongBucket.beginindex + hashposition1].statevector  == insertrecord.statevector){
		return true;
	}
	if(HT[belongBucket.beginindex + hashposition2].statevector == insertrecord.statevector){
		return true;
	}
	if(HT[belongBucket.beginindex + hashposition3].statevector == insertrecord.statevector){
		return true;
	}

	if(HT[belongBucket.beginindex + hashposition1].statevector == EMPTYVECT32){
		*hashresult = hashposition1;
	}
	
	if(HT[belongBucket.beginindex + hashposition2].statevector == EMPTYVECT32){
		*hashresult = hashposition2;
		return false;
	}

	if(HT[belongBucket.beginindex + hashposition3].statevector == EMPTYVECT32){
		*hashresult = hashposition3;
		return false;
	}

	*hashresult = hashposition1;
	return false;
}

__global__ bool CudaVisitedGlobalHashstore(LocalRecord * HT, unsigned int hasbucket, unsigned int hashv, LocalRecord insertedrecord, unsigned int * EBits, unsigned int ltsnum){
	Bucket buckethash;
	LocalRecord kickoutRecord;

	unsigned int kickoutkey, kickouthash1, kickouthash2, kickouthash3;

	char tmp;

	while(true){
		buckethash = GlobalBuckets[hasbucket];
		if(atomicCAS(&HT[buckethash.beginindex + hashv].statevector, insertedrecord.statevector, EMPTYVECT32)){
			HT[buckethash.beginindex + hashv].localmark = insertedrecord.localmark;
			break;
		}else{
			kickoutRecord.statevector = atomicExch(&(HT[buckethash.beginindex + hashv].statevector), kickoutRecord.statevector);
			tmp = HT[buckethash.beginindex + hashv].localmark;
			HT[buckethash.beginindex + hashv].localmark = insertedrecord.localmark;
			kickoutRecord.localmark = tmp;

			kickoutkey = CudaGenerateKey(kickoutRecord.statevector, EBits, ltsnum);
			kickouthash1 = Globalhash1(kickoutkey);
			kickouthash2 = Globalhash2(kickoutkey);
			kickouthash3 = Globalhash3(kickoutkey);

			if(hashv == kickouthash1){
				hashv = kickouthash2;
			}else if(hashv == kickouthash2){
				hashv = kickouthash3;
			}else{
				hashv = kickouthash1;
			}

			if(atomicCAS(&HT[buckethash.beginindex + hashv].statevector, kickoutRecord.statevector, EMPTYVECT32)){
				HT[buckethash.beginindex + hashv].localmark = kickoutRecord.localmark;
				break;
			}else{
				if(hashv == kickouthash1){
					if(hasbucket < GlobalBucketNum)
						hasbucket++;
					else
						hasbucket--;

					insertedrecord = kickoutRecord;
					continue;
				}

				/*kickoutRecord.statevector = atomicExch(&(HT[buckethash.beginindex + kickoutstore].statevector), kickoutRecord.statevector);
				tmp = HT[buckethash.beginindex + kickoutstore].localmark;
				HT[buckethash.beginindex + kickoutstore].localmark = kickoutRecord.localmark;
				kickoutRecord.localmark = tmp;

				kickoutkey = CudaGenerateKey(kickoutRecord.statevector, EBits, ltsnum);
				kickouthash1 = Globalhash1(kickoutkey);
				kickouthash2 = Globalhash2(kickoutkey);
				kickouthash3 = Globalhash3(kickoutkey);

				if(kickoutstore == kickouthash1){
					kickoutstore = kickouthash2;
				}else if(kickoutstore == kickouthash2){
					kickoutstore = kickouthash3;
				}else{
					kickoutstore = kickouthash1;
				}

				if(atomicCAS(&HT[buckethash.beginindex + kickoutstore].statevector, kickoutRecord.statevector, EMPTYVECT32)){
					HT[buckethash.beginindex + kickoutstore].localmark = kickoutRecord.localmark;
					break;
				}*/
			}
		}
	}

	return true;
	
}

__global__ void CudaGenerateCounterexample(){

}

__global__ void CUDADeadlockBFSVerify(unsigned int * PG_AllLTS, unsigned int * PG_AllStates, unsigned char * PG_AllTransitions, unsigned char * PG_AllSynctransitions, unsigned int * PGlobalBuckets, unsigned int * PGlobalHash, unsigned int * PG_Startlist, unsigned int * PG_LTSStateEncodeBits, unsigned int * PG_LTSStateEncodeBytes, unsigned int PEventEncodeBytes, unsigned int PG_Bucketnum, unsigned int PLTSNum)
{
	int i,j,m,k;

	int Inblocktid = threadIdx.x;
	int Ingridtid = threadIdx.x + blockIdx.x * blockDim.x;
	int InWarptid = Inblocktid % 32;
	int InvthreadgroupID;
	int vthreadgroupID;
	int Warpid = Inblocktid/32;
	int WarpNum = blockDim.x/32;

	unsigned int layer;
	unsigned int getindex; //the index to get the initial task from global memory.

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

	int tmpsucc, tmpnode;

	int vthreadgroupnuminblock;
	int vthreadgroupnuminwarp;
	char tmp;

	//unsigned int localKey, localhash;
	LocalRecord kickoutRecord;
	LocalRecord insertRecord;
	LocalRecord visitedRecord;
	unsigned int hkey;

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
	__shared__ int maxlayer;
	//__shared__ bool IfexistFree; //

	extern __shared__ LocalRecord S[]; 
	LocalRecord * RecordTable1 = S;
	LocalRecord * RecordTable2 = &RecordTable1[blockDim.x];
	LocalRecord * RecordTable3 = &RecordTable2[blockDim.x];
	LocalRecord * GroupStore = &RecordTable3[blockDim.x];
	bool * syncduplicate = (bool *)&GroupStore[vthreadgroupnuminblock];
	bool * needsyndupdetect = &syncduplicate[vthreadgroupnuminblock*PLTSNum];

	unsigned int * SynEventInteractive = (unsigned int *)&GroupStore[vthreadgroupnuminblock];
	bool * IFNOOUTGOING = (bool *)&SynEventInteractive[vthreadgroupnuminblock*PLTSNum];

	if(Inblocktid == 0){
		for(i = 0; i < vthreadgroupnuminblock * PLTSNum; i++){
			IFNOOUTGOING[i] = false; 
			SynEventInteractive[i] = EMPTYVECT32;
		}
		maxlayer=0;
	}

	if(InvthreadgroupID != -1){
		syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;
	}
	layer = 0;
	if(InvthreadgroupID == 0){
		getindex = vthreadgroupnuminblock * blockIdx.x + vthreadgroupID;
		GroupStore[vthreadgroupID].statevector = PG_Startlist[getindex];
		GroupStore[vthreadgroupID].localmark = GroupStore[vthreadgroupID].localmark | layer;
		needsyndupdetect[vthreadgroupID] = false;
	}

	while(GroupStore[vthreadgroupID].statevector == EMPTYVECT32);

	do{
		localstate = CudaGetStateinVec(InvthreadgroupID, GroupStore[vthreadgroupID].statevector, PG_LTSStateEncodeBits);
		belonglts = InvthreadgroupID;
		ifanyoutgoing = CudaGetAllsuccessors(PG_AllLTS, PG_AllStates, PG_AllTransitions, belonglts, localstate, &SuccessorMark);
		ifglobaldup = false;
		//The successor generation consists of two steps: 1. For trans in alltransitions, process them directly. 2.For trans in allsynctrans, parallel sync is needed.
		if(ifanyoutgoing){
			i = SuccessorMark.beginbyte;
			//calculate global hash position for visited stateV
			if(vthreadgroupID == 0){
				globalbuckethash = CudaGenerateKey(GroupStore[vthreadgroupID].statevector, PG_LTSStateEncodeBits, PLTSNum);
				hkey = CudaGenerateKey(GroupStore[vthreadgroupID].statevector, PG_LTSStateEncodeBits, PLTSNum);
				ifglobaldup = CudaVisitedGlobalHashcal(GlobalVisitedHash, GlobalBuckets[globalbuckethash],hkey, GroupStore[vthreadgroupID], &visitedstore);
			}

			localstateV = GroupStore[vthreadgroupID].statevector;
			visitedRecord.statevector = localstateV;
			visitedRecord.localmark = (char)layer;

			while(i < SuccessorMark.endbyte && !ifglobaldup){
				CudaNewStateV(&localstateV, InvthreadgroupID, &i, PG_AllTransitions, PG_LTSStateEncodeBits, PG_LTSStateEncodeBytes[belonglts]);
				
				if(!Ifcollisionhappens){
					insertRecord.localmark = (char)(layer+1);
					insertRecord.statevector = localstateV;
					
					//hash store and duplicate elimination module.....
					CudaHashStore(insertRecord, layer, PG_LTSStateEncodeBits, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
					
				}
					
				if(Ifcollisionhappens){
					break;
				}
			
			}
			//synchronization part
			j = SuccessorMark.synbeginbyte;
			
			if(!Ifcollisionhappens){
				bool  ifmatch;
				m = 0;
				CudaDecodeTransitions(PG_AllSynctransitions, SuccessorMark.synendbyte-PEventEncodeBytes-PG_LTSStateEncodeBytes[belonglts], &localstate2, &maxtransevent, PEventEncodeBytes, PG_LTSStateEncodeBytes[belonglts]);
				while(j <= SuccessorMark.synendbyte){
					ifmatch = false;
					if(m == 0 && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum]){
						if(j == SuccessorMark.synendbyte)
							break;
						CudaDecodeTransitions(PG_AllSynctransitions, j, &localstate, &SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], PEventEncodeBytes, PG_LTSStateEncodeBytes[belonglts]);
						CudaNewStateV(&localstateV, InvthreadgroupID, &j, PG_AllSynctransitions, PG_LTSStateEncodeBits, PG_LTSStateEncodeBytes[belonglts]);
					}

					for(i=0; i<PLTSNum; i++){
						if(i == InvthreadgroupID)
							continue;

						if(SynEventInteractive[i + vthreadgroupID * PLTSNum] > maxtransevent){  //if bigger than the maxtransevent of local, no need to compare as it's impossible to sync
							if(SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] > SynEventInteractive[i + vthreadgroupID * PLTSNum]){
								m++;

							}else if (SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum] == SynEventInteractive[i + vthreadgroupID * PLTSNum]){
								if(needsyndupdetect[vthreadgroupID] == false)
									needsyndupdetect[vthreadgroupID] = true;
								//GENERATE SYNC STATE V.......
								SynTwoStates(&localstateV, i, SynEventInteractive[i + vthreadgroupID * PLTSNum], GroupStore[vthreadgroupID], PG_AllTransitions， PG_AllSynctransitions)；
								syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
								ifmatch = true;
							}
						}
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
						insertRecord.localmark = (char)(layer+1);
						insertRecord.statevector = localstateV;

						if(!Ifcollisionhappens)
							CudaHashStore(insertRecord, layer, PG_LTSStateEncodeBits, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
						
						syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
						if(Ifcollisionhappens){
							for(k = 511; k > 0; k--){
								if(kickoutRecord.statevector != EMPTYVECT32){
									if(atomicCAS(&(RecordTable3[k].statevector), kickoutRecord.statevector, EMPTYVECT32)){
										RecordTable3[k].localmark = (char)(layer + 1);
									}
									kickoutRecord.statevector = EMPTYVECT32;
								}else{
									if(atomicCAS(&(RecordTable3[k].statevector), localstateV, EMPTYVECT32)){
										RecordTable3[k].localmark = (char)(layer + 1);
									}
								}
							}
						}
					}

					if(!ifmatch && m == 0){
						syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
					}
				}
			}
			
		}else{
			IFNOOUTGOING[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
		}

		if(InvthreadgroupID == 0){
			for(i = 0; i < PLTSNum; i++){
				if(!IFNOOUTGOING[i + vthreadgroupID * PLTSNum] && !Ifcollisionhappens&&!ifglobaldup)
					break;
			}

			if(i == PLTSNum){
				IFDeadlockDetected = true;
				break;
			}
		}

		CudaInterBlocksSyn(gridDim.x);

		if(!IFDeadlockDetected && InWarptid == 0&&!Ifcollisionhappens&&!ifglobaldup){
			//copy visited state to global memory
			CudaVisitedGlobalHashstore(GlobalVisitedHash, globalbuckethash, visitedstore, GroupStore[vthreadgroupID], PG_LTSStateEncodeBits, PLTSNum);
			if(InvthreadgroupID == 0){
				GroupStore[vthreadgroupID].statevector = EMPTYVECT32;
			}
		}else if(IFDeadlockDetected){
			break;
		}else if(Ifcollisionhappens){
			//load new kernel

		}
		

		if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID].statevector == EMPTYVECT32){
			//got new stateV
			localstateV = EMPTYVECT32;
			ifgetnewstatev = false;
			while(layer < maxlayer + 1 || ifgetnewstatev == true){
				for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
					if((int)RecordTable1[i].localmark == layer){
						if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32) != EMPTYVECT32))
						{
							ifgetnewstatev = true;
							break;
						}

					}
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
						if((int)RecordTable2[i].localmark == layer){
							if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								ifgetnewstatev = true;
								break;
							}

						}
					}
				}else{
					break;
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
						if((int)RecordTable3[i].localmark == layer){
							if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								ifgetnewstatev = true;
								break;
							}

						}
					}
				}else{
					break;
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
						if((int)RecordTable1[i].localmark == layer){
							if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								ifgetnewstatev = true;
								break;
							}
						}
					}
				}else{
					break;
				}
				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
						if((int)RecordTable2[i].localmark == layer){
							if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								ifgetnewstatev = true;
								break;
							}
						}
					}
				}else{
					break;
				}
				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
						if((int)RecordTable3[i].localmark == layer){
							if((GroupStore[vthreadgroupID] = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								ifgetnewstatev = true;
								break;
							}
						}
					}
				}else{
					break;
				}

				if(ifgetnewstatev == false){
					layer++;
				}
			}
			
		}
		
	}while(!IFDeadlockDetected);

	if(!IFDeadlockDetected && InWarptid == 0){

	}

	if(!IFDeadlockDetected && Ingridtid == 0){

	}

}

int HostGenerateStateSpace(int LTSNum, unsigned int * H_AllLTS, unsigned int * H_AllStates, unsigned char * H_AllTransitions, unsigned char * H_AllSynctrans, unsigned int * RecordList, unsigned int RequestNum, unsigned int H_InitialStateV, unsigned int * H_LTSStateEncodeBits, unsigned int * H_LTSStateEncodeBytes, unsigned int HEventEncodeBytes)
{
	int i,j,m,k;
	int SuccessorCount = 0;
	queue<unsigned int> Taskqueue;
	set<unsigned int> Taskset;
	list<unsigned int> Syncqueue;
	list<unsigned int>::iterator Syncit;

	set<unsigned int> Syncevents;
	set<unsigned int>::iterator Syncposition;

	queue<unsigned int> VisitedS;

	unsigned int newStateV;
	unsigned int * succStateV;
	unsigned int * tmpStateV;
	unsigned int newState;
	unsigned int belonglts;
	unsigned int transevent;
	
	bool ifoutgoing;
	int ifoutgoingcount;
	Nodemark allsucc;
	
	Taskqueue.push(H_InitialStateV);
	while(SuccessorCount < RequestNum){
		newStateV = Taskqueue.front();
		ifoutgoingcount = 0;
		for(i = 0; i < LTSNum; i++){
			ifoutgoing = false;
			CudaGetStateinVec(i, newStateV, &newState, HEventEncodeBytes, &belonglts);
			ifoutgoing = CudaGetAllsuccessors(H_AllLTS, H_AllStates, H_AllTransitions, belonglts, newState, &allsucc);
			if(!ifoutgoing){
				ifoutgoingcount++;
				continue;
			}

			m = allsucc.beginbyte;
			while(m < allsucc.endbyte){
				succStateV = new unsigned int[1];
				CudaNewStateV(succStateV, i, &m, H_AllTransitions, H_LTSStateEncodeBits[belonglts], H_LTSStateEncodeBytes[belonglts], HEventEncodeBytes);
				if(Taskset.insert(succStateV).second){
					Taskqueue.push(*succStateV);
					SuccessorCount++;
				}
			}

			k = allsucc.synbeginbyte;
			while(k < allsucc.synendbyte){
				succStateV = new unsigned int[1];
				CudaDecodeTransitions(H_AllSynctrans, k, &newState, &transevent, HEventEncodeBytes, H_LTSStateEncodeBytes[belonglts]);
				CudaNewStateV(succStateV, i, &k, H_AllSynctrans, H_LTSStateEncodeBits[belonglts], H_LTSStateEncodeBytes[belonglts], HEventEncodeBytes);
				Syncposition = Syncevents.find(transevent);
				if(Syncposition != Syncevents.end()){
					tmpStateV = (unsigned int *)&(*(Syncqueue.rbegin() + *Syncposition));
					SynTwoStates(tmpStateV, *succStateV, i, newStateV);
					
				}else{
					Syncevents.insert(transevent);
					Syncqueue.push_back(*succStateV);
					SuccessorCount++;
				}	
			}
			for(Syncit = Syncqueue.begin(); Syncit != Syncqueue.end(); Syncit++) {
				Taskqueue.push(*Syncit);
			}
			Syncqueue.clear();
		}
		if(ifoutgoingcount == LTSNum){
			return -1;
		}
		
	}

	RecordList = new unsigned int[SuccessorCount];
	for(i = 0; i < SuccessorCount; i++){
		RecordList[i] = Taskqueue.front();
		Taskqueue.pop();
	}

	return SuccessorCount;
}

int main()
{
	int i,j;
	unsigned int * G_AllLTS;
	unsigned int * G_AllStates;
	unsigned char * G_AllTransitions;
	unsigned char * G_AllSyncTrans;  //all trans with sync events.
	//unsigned int * G_InitialStateV;
	
	unsigned int * G_LTSStateEncodeBytes;
	unsigned int * G_LTSStateEncodeBits;

	//Choose to generate some statevectors firstly in CPU---OPTIONAL
	unsigned int * G_Startlist; 

	unsigned int * AllLTS;  //read from file
	unsigned int * AllStates;
	unsigned char * AllTransitions;
	unsigned char* AllSyncTrans;

	unsigned int SyncindexEncodeBytes; //mark the encode bytes of index from AllTransitions->AllSyncTrans.
	
	LocalRecord * H_Globalhash;
	Bucket * H_GlobalBuckets;
	//Choose to generate some statevectors firstly in CPU---OPTIONAL
	unsigned int * H_Startlist;

	unsigned int * H_GlobalVisitedhash;

	unsigned int H_InitialSV;

	unsigned int AllLTSStateNum;
	unsigned int LTSNum;   //equals to the number of states in a statevector
	unsigned int AllTransLength = 0;
	unsigned int AllSyncTransLength = 0;

	unsigned int LTSEncodeBytes;
	unsigned int EventEncodeBytes;

	unsigned int * LTSStateNum = new unsigned int[LTSNum];
	unsigned int * LTSStateEncodeBits = new unsigned int[LTSNum];
	unsigned int * LTSStateEncodeBytes = new unsigned int[LTSNum];

	unsigned int Startblocknum;
	unsigned int Startthreadnum1block;
	unsigned int Startthreadgroupnum;

	int rv[8];
	srand(time(NULL));
	for(i = 0; i < 8; i++){
		rv[i] = rand();
	}

	cudaMemcpyToSymbol(LA1, &rv[0], sizeof(int));
	cudaMemcpyToSymbol(LA2, &rv[1], sizeof(int));
	cudaMemcpyToSymbol(LA3, &rv[2], sizeof(int));
	cudaMemcpyToSymbol(LB1, &rv[3], sizeof(int));
	cudaMemcpyToSymbol(LB2, &rv[4], sizeof(int));
	cudaMemcpyToSymbol(LB3, &rv[5], sizeof(int));
	cudaMemcpyToSymbol(BUCA, &rv[6], sizeof(int));
	cudaMemcpyToSymbol(BUCB, &rv[7], sizeof(int));

	for(i = 0; i < 6; i++){
		rv[i] = rand();
	}
	cudaMemcpyToSymbol(GA1, &rv[0], sizeof(int));
	cudaMemcpyToSymbol(GA2, &rv[1], sizeof(int));
	cudaMemcpyToSymbol(GA3, &rv[2], sizeof(int));
	cudaMemcpyToSymbol(GB1, &rv[3], sizeof(int));
	cudaMemcpyToSymbol(GB2, &rv[4], sizeof(int));
	cudaMemcpyToSymbol(GB3, &rv[5], sizeof(int));

	for(i = 0; i < LTSNum; i++){
		
		AllLTSStateNum += LTSStateNum[i];
	}

	if(!InitCUDA()){
	    printf("Sorry,CUDA has not been initialized.\n");
	    return NULL;
    }

	H_Globalhash = new LocalRecord[AllLTSStateNum * 5];
	H_GlobalBuckets = new Bucket[LTSNum * 2];

	H_GlobalVisitedhash = new unsigned int[AllLTSStateNum * 10];
	
	for(i = 0; i < AllLTSStateNum * 5; i++){
		H_Globalhash[i] = EMPTYVECT32;
	}

	//Initial the global bucket
	for(i = 0; i < LTSNum*2; i++){
		
	}

	//Initialize Startlist
	Startthreadgroupnum = (((Startthreadnum1block/32)/LTSNum)*(Startthreadnum1block/32))*Startblocknum;  //initial value, not the final one?
	//H_Startlist = new unsigned int[Startthreadgroupnum];
	i = HostGenerateStateSpace(LTSNum, AllLTS,AllStates,AllTransitions, AllSyncTrans, H_Startlist,Startthreadgroupnum, H_InitialSV, LTSStateEncodeBits, LTSStateEncodeBytes, EventEncodeBytes);
	if(i > 0){
		j = i * LTSNum;
		Startthreadgroupnum += i;
		Startblocknum = Startthreadgroupnum/(Startthreadnum1block/LTSNum);
	}else if(i == -1){
		cout<<"deadlock being detected";
		return 0;
	}

    cudaMalloc((unsigned int *)&G_AllLTS, sizeof(unsigned int) * LTSNum);
	cudaMalloc((unsigned int *)&G_AllStates, sizeof(unsigned int) * AllLTSStateNum);
	cudaMalloc((unsigned char *)&G_AllTransitions, sizeof(unsigned char) * AllTransLength);
	cudaMalloc((unsigned char *)&G_AllSyncTrans,sizeof(unsigned char) * AllSyncTransLength);
	cudaMalloc((LocalRecord *)&GlobalOpenHash, sizeof(unsigned int) * AllLTSStateNum * 5);
	cudaMalloc((Bucket *)&GlobalBuckets, sizeof(Bucket) * LTSNum * 2);
	cudaMalloc((unsigned int *)&G_LTSStateEncodeBytes, sizeof(unsigned int) * LTSNum);
	cudaMalloc((unsigned int *)&G_LTSStateEncodeBits, sizeof(unsigned int) * LTSNum);
	cudaMalloc((unsigned int *)&G_Startlist, sizeof(unsigned int) * Startthreadgroupnum);
	cudaMalloc((LocalRecord *)&GlobalVisitedHash, sizeof(LocalRecord) * AllLTSStateNum * 10);
	//cudaMalloc((unsigned int *)&G_InitialStateV, sizeof(int));

	cudaMemcpy(G_AllLTS, AllLTS, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllStates, AllStates, sizeof(unsigned int) * AllLTSStateNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllTransitions, AllTransitions, sizeof(unsigned char) * AllTransLength, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllSyncTrans, AllSyncTrans, sizeof(unsigned char) * AllSyncTransLength, cudaMemcpyHostToDevice);
	//cudaMemcpy(G_InitialStateV, &H_InitialSV, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(GlobalOpenHash, H_Globalhash, sizeof(unsigned int) * AllLTSStateNum * 5, cudaMemcpyHostToDevice);
	cudaMemcpy(GlobalBuckets, H_GlobalBuckets, sizeof(Bucket) * LTSNum * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(G_LTSStateEncodeBytes, LTSStateEncodeBytes, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_LTSStateEncodeBits, LTSStateEncodeBits, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_Startlist, H_Startlist, sizeof(unsigned int) * Startthreadgroupnum, cudaMemcpyHostToDevice);

	cudaBindTexture(0, LTSOFFSET, G_AllLTS);
	cudaBindTexture(0, STATEOFFSET, G_AllStates);
	cudaBindTexture(0, OUTGOINGDETAIL, G_AllTransitions);  //how texture memory can accelerate the access rate need to be explored
	cudaBindTexture(0, STATEENCODE, G_LTSStateEncodeBytes);

	CUDADeadlockBFSVerify<<<>>>(G_AllLTS, G_AllStates, G_AllTransitions, G_AllSyncTrans, GlobalBuckets, GlobalHash, G_Startlist, G_LTSStateEncodeBits, G_LTSStateEncodeBytes, EventEncodeBytes, LTSNum*2, LTSNum);
	
	cudaUnbindTexture(LTSOFFSET);
	cudaUnbindTexture(STATEOFFSET);
	cudaUnbindTexture(OUTGOINGDETAIL);
	cudaUnbindTexture(STATEENCODE);

	cudaFree(G_AllLTS);
	cudaFree(G_AllStates);
	cudaFree(G_AllTransitions);
	cudaFree(GlobalBuckets);
	cudaFree(GlobalHash);
	free(AllLTS);
	free(AllStates);
	free(AllTransitions);
	free(H_GlobalBuckets);
	free(H_Globalhash);
}


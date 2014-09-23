
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
#include <fstream>
#include <iomanip>
using namespace std;

texture<unsigned int, 1, cudaReadModeElementType> LTSOFFSET;  //1 means 1-dimension
texture<unsigned int, 1, cudaReadModeElementType> STATEOFFSET;
texture<unsigned char, 1, cudaReadModeElementType> OUTGOINGDETAIL;
texture<unsigned char, 1, cudaReadModeElementType> SYNCOUTGOING;
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
__constant__ int TableSize;
__constant__ unsigned int PrimeNum = 334214459;

static const unsigned int EMPTYVECT32 = 0x7FFFFFFF;
static const unsigned int P = 334214459;
static const unsigned int blocksize = 512;

class LocalRecord{
public:
	char localmark;  //record the BFS layer in Shared Memory
	char toevent;
	int statevector;

	__device__ void operator= (LocalRecord t){
		localmark = t.localmark;
		toevent = t.toevent;
		statevector = t.statevector;
	}
};

class Bucket{
public:
	unsigned int beginindex;
	unsigned int endindex;

};

class Nodemark{
public:
	unsigned int beginbyte;
	unsigned int endbyte;
	unsigned int synbeginbyte;
	unsigned int synendbyte;

};

__device__ LocalRecord *GlobalOpenHash;  
__device__ Bucket *GlobalBuckets;
__device__ unsigned int GlobalBucketNum;

__device__ unsigned int *GlobalbucketIndex;  //just used for open
__device__ unsigned int *GlobalbucketCount;

__device__ LocalRecord *GlobalVisitedHash;  //here, for visited stateV, use hash to store back to global memory. While this hash doesn't kick the original one. For open stateV, use buckets hash.
//__device__ unsigned int GlobalVisitedHashoffset[3];

__device__ unsigned int communicationlayer[100];
__device__ bool communicationcollision[100];
__device__ Bucket *communicationGstore;  //store the buckets that child blocks store their data
__device__ bool Ifreturn2parent[100];

__device__ volatile unsigned int * GlobalBucketsCount;

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

__device__ unsigned int Globalhash1(unsigned int k)
{
	return (GA1 * k + GB1) % PrimeNum % (3*TableSize);
}

__device__ unsigned int Globalhash2(unsigned int k)
{
	return (GA2 * k + GB2) % PrimeNum % (3*TableSize);
}

__device__ unsigned int Globalhash3(unsigned int k)
{
	return (GA3 * k + GB3) % PrimeNum % (3*TableSize);
}

__device__ unsigned int Localhash1(unsigned int k)
{
	return (LA1 * k + GA1) % PrimeNum % TableSize;
}

__device__ unsigned int Localhash2(unsigned int k)
{
	return (LA2 * k + GA2) % PrimeNum % TableSize;
}

__device__ unsigned int Localhash3(unsigned int k)
{
	return (LA3 * k + GA3) % PrimeNum % TableSize;
}


__device__ unsigned int CudaGetStateinVec(int index, unsigned int svec, unsigned int * stateencodebits)
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

__device__ bool CudaGetAllsuccessors(unsigned int * AllLTS, unsigned int * Allstates, unsigned char * Alltransitions, unsigned int ltsindex, unsigned int sindex, Nodemark * result)
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

__device__ void CudaNewStateV(unsigned int * targetV, int tindex, int * index, unsigned char * Atrans, unsigned int * bitwidth, unsigned int bytewidth)
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
	* index += bytewidth;
}

__device__ void CudaDecodeTransitions(unsigned char * outgoingT, int beginindex, unsigned int * Tostate, unsigned int * Tevent, unsigned int Eventwidth, unsigned int Statewidth)
{
	unsigned int stateendbyte, eventendbyte;
	stateendbyte = beginindex + Eventwidth + Statewidth;
	eventendbyte = beginindex + Eventwidth;
	int i;
	unsigned char tmp[4];
	for(i = 0; i < 4; i++){
		tmp[i] = (char)0x00;
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

__device__ unsigned int CudaGenerateKey(unsigned int KV, unsigned int *ecbit, int snum)
{
	return KV;

}

__device__ void SynTwoStates(unsigned int * s1, int s2idx, unsigned int evtid, unsigned int * sbitwidth, unsigned int svec, unsigned int * alllts, unsigned int * allstates, unsigned char * alltrans, unsigned char * allsynctrans, unsigned int ewidth, unsigned int swidth)
{
	unsigned int localstate;
	int beginbit, endbit;
	localstate = CudaGetStateinVec(s2idx, svec, sbitwidth);
	Nodemark SuccessorMark;
	unsigned char tmp[4];
	int i,j, m;

	tmp[0] = (char) evtid;
	tmp[1] = (char) evtid >> 8;
	tmp[2] = (char) evtid >> 16;
	tmp[3] = (char) evtid >> 24;

	CudaGetAllsuccessors(alllts, allstates, alltrans, s2idx, localstate, &SuccessorMark);

	for(i = SuccessorMark.synbeginbyte; i < SuccessorMark.synendbyte;){
		for(j = 0; j < ewidth; j++){
			if(tmp[ewidth - j - 1] != allsynctrans[i+j])
				break;
		}
		if(j != ewidth){
			i += (ewidth + swidth);
		}else{
			i += ewidth;
			CudaNewStateV(s1, s2idx, &i, allsynctrans, sbitwidth, swidth);
			break;
		}

	}
}

void SynTwoStatesCPU(unsigned int * tmpStateV, unsigned int succStateV, int i, unsigned int newStateV, unsigned int * bitwidth){
	int beginbit, endbit;
	int beginbyte, endbyte;
	int j,m;

	unsigned char tmp1[4];
	unsigned char tmp2[4];

	tmp1[0] = (char)(*tmpStateV);
	tmp1[1] = (char)(*tmpStateV >> 8);
	tmp1[2] = (char)(*tmpStateV >> 16);
	tmp1[3] = (char)(*tmpStateV >> 24);

	tmp2[0] = (char)(succStateV);
	tmp2[1] = (char)(succStateV >> 8);
	tmp2[2] = (char)(succStateV >> 16);
	tmp2[3] = (char)(succStateV >> 24);

	for(j = 0; j < i; j++){
		beginbit += bitwidth[j];
	}
	endbit = beginbit + bitwidth[i];

	beginbyte = beginbit / 8;
	endbyte = endbit / 8;
	beginbit = beginbit % 8;
	endbit = endbit % 8;

	for(m = beginbyte; m < endbyte; m++){
		tmp1[m] = tmp1[m] >> (8 - beginbit);
		tmp2[m] = tmp2[m] << beginbit;
		tmp2[m] = tmp2[m] >> beginbit;
		tmp1[m] = tmp1[m] | tmp2[m];
	}

	*tmpStateV = (unsigned int)(tmp1[0] | tmp1[1] << 8 | tmp1[2] << 16 | tmp1[3] << 24);


}

__device__ bool CudaHashStore(LocalRecord beginHV, unsigned int layer, unsigned int * EBits, unsigned int PLTSNum, LocalRecord * T1, LocalRecord * T2, LocalRecord * T3, LocalRecord * RkickoutRecord){
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
					kickoutRecord.statevector = atomicExch(&(T2[localhash].statevector), kickoutRecord.statevector);
					tmp = T2[localhash].localmark;
					T2[localhash].localmark = kickoutRecord.localmark;
					kickoutRecord.localmark = tmp;

					localKey = CudaGenerateKey(kickoutRecord.statevector, EBits, PLTSNum);
					localhash = Localhash3(localKey);
					if(atomicCAS(&(T3[localhash].statevector), kickoutRecord.statevector, EMPTYVECT32)){
						T3[localhash].localmark = (char)(layer + 1);
					}else{
						if(T3[localhash].statevector == kickoutRecord.statevector){
							return true;
						}else{
							//kick out the one in localhash3
							kickoutRecord.statevector = atomicExch(&(T3[localhash].statevector), kickoutRecord.statevector);
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

__device__ bool CudaVisitedGlobalHashcal(LocalRecord * HT, Bucket belongBucket, unsigned int hkey, LocalRecord insertrecord, unsigned int * hashresult){
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

__device__ bool CudaVisitedGlobalHashstore(LocalRecord * HT, unsigned int hasbucket, unsigned int hashv, LocalRecord insertedrecord, unsigned int * EBits, unsigned int ltsnum){
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

__global__ void CudaGenerateCounterexample()
{

}

__global__ void CUDADeadlockBFSVerifyChild(unsigned int ParentID, unsigned int PBucket, Bucket * Cbucket, unsigned int * CG_AllLTS, unsigned int * CG_AllStates, unsigned char * CG_AllTransitions, unsigned char * CG_AllSynctransitions,  unsigned int * CG_LTSStateEncodeBits, unsigned int * CG_LTSStateEncodeBytes, unsigned int CEventEncodeBytes, unsigned int CG_Bucketnum, unsigned int PLTSNum)
{
	int i,j,m,k;

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

	int tmpsucc, tmpnode;

	int vthreadgroupnuminblock;
	int vthreadgroupnuminwarp;
	char tmp;

	//unsigned int localKey, localhash;
	LocalRecord kickoutRecord;
	LocalRecord insertRecord;
	LocalRecord visitedRecord;
	unsigned int hkey;
	unsigned int getindex; // the index to get tasks

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

	__shared__ int nonewcount;
	__shared__ bool Ifcollisionhappens;
	__shared__ int maxlayer;

	extern __shared__ bool C[]; 
	bool * syncduplicate = C;
	bool * needsyndupdetect = &syncduplicate[vthreadgroupnuminblock*PLTSNum];
	bool * ifnooutgoing = &needsyndupdetect[vthreadgroupnuminblock];
	unsigned int * SynEventInteractive = (unsigned int *)&ifnooutgoing[vthreadgroupnuminblock*PLTSNum];

	LocalRecord * RecordTable1 = (LocalRecord *)&(SynEventInteractive[vthreadgroupnuminblock*PLTSNum]);
	LocalRecord * RecordTable2 = &RecordTable1[blockDim.x];
	LocalRecord * RecordTable3 = &RecordTable2[blockDim.x];
	LocalRecord * GroupStore = &RecordTable3[blockDim.x];

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
		
		nonewcount = 0;
		maxlayer = 0;
		Ifcollisionhappens = false;
	}

	if(InvthreadgroupID != -1){
		syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;
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
		GlobalOpenHash[globalbuckethash].statevector = EMPTYVECT32;
	}
	
	do{
		if(GroupStore[vthreadgroupID].statevector != EMPTYVECT32){
			layer = (unsigned int)GroupStore[vthreadgroupID].localmark;
		
			localstate = CudaGetStateinVec(InvthreadgroupID, GroupStore[vthreadgroupID].statevector, CG_LTSStateEncodeBits);
			belonglts = InvthreadgroupID;
			ifanyoutgoing = CudaGetAllsuccessors(CG_AllLTS, CG_AllStates, CG_AllTransitions, belonglts, localstate, &SuccessorMark);
			ifglobaldup = false;
			//The successor generation consists of two steps: 1. For trans in alltransitions, process them directly. 2.For trans in allsynctrans, parallel sync is needed.
			if(ifanyoutgoing){
				i = SuccessorMark.beginbyte;
				//calculate global hash position for visited stateV
				if(vthreadgroupID == 0){
					globalbuckethash = CudaGenerateKey(GroupStore[vthreadgroupID].statevector, CG_LTSStateEncodeBits, PLTSNum);
					hkey = CudaGenerateKey(GroupStore[vthreadgroupID].statevector, CG_LTSStateEncodeBits, PLTSNum);
					ifglobaldup = CudaVisitedGlobalHashcal(GlobalVisitedHash, GlobalBuckets[globalbuckethash],hkey, GroupStore[vthreadgroupID], &visitedstore);
				}

				localstateV = GroupStore[vthreadgroupID].statevector;
				visitedRecord = GroupStore[vthreadgroupID];

				while(i < SuccessorMark.endbyte && !ifglobaldup){
					CudaNewStateV(&localstateV, InvthreadgroupID, &i, CG_AllTransitions, CG_LTSStateEncodeBits, CG_LTSStateEncodeBytes[belonglts]);
				
					if(!Ifcollisionhappens){
						insertRecord.localmark = (char)(layer+1);
						insertRecord.statevector = localstateV;
					
						//hash store and duplicate elimination module.....
						CudaHashStore(insertRecord, layer, CG_LTSStateEncodeBits, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
					
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
					CudaDecodeTransitions(CG_AllSynctransitions, SuccessorMark.synendbyte-CEventEncodeBytes-CG_LTSStateEncodeBytes[belonglts], &localstate2, &maxtransevent, CEventEncodeBytes, CG_LTSStateEncodeBytes[belonglts]);
					while(j <= SuccessorMark.synendbyte){
						ifmatch = false;
						if(m == 0 && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum]){
							if(j == SuccessorMark.synendbyte)
								break;
							CudaDecodeTransitions(CG_AllSynctransitions, j, &localstate, &SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], CEventEncodeBytes, CG_LTSStateEncodeBytes[belonglts]);
							CudaNewStateV(&localstateV, InvthreadgroupID, &j, CG_AllSynctransitions, CG_LTSStateEncodeBits, CG_LTSStateEncodeBytes[belonglts]);
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
									SynTwoStates(&localstateV, i, SynEventInteractive[i + vthreadgroupID * PLTSNum], CG_LTSStateEncodeBits, GroupStore[vthreadgroupID].statevector, CG_AllLTS, CG_AllStates, CG_AllTransitions, CG_AllSynctransitions, CEventEncodeBytes, CG_LTSStateEncodeBytes[i]);
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
								CudaHashStore(insertRecord, layer, CG_LTSStateEncodeBits, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
						
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
				ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			}

			if(InvthreadgroupID == 0){
				for(i = 0; i < PLTSNum; i++){
					if(!ifnooutgoing[i + vthreadgroupID * PLTSNum] && !Ifcollisionhappens&&!ifglobaldup)
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
				CudaVisitedGlobalHashstore(GlobalVisitedHash, globalbuckethash, visitedstore, GroupStore[vthreadgroupID], CG_LTSStateEncodeBits, PLTSNum);
				if(InvthreadgroupID == 0){
					GroupStore[vthreadgroupID].statevector = EMPTYVECT32;
				}
			}else if(IFDeadlockDetected){
				break;
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
						if(RecordTable1[m].statevector != EMPTYVECT32)
							myareacount++;
						if(RecordTable2[m].statevector != EMPTYVECT32)
							myareacount++;
						if(RecordTable3[m].statevector != EMPTYVECT32)
							myareacount++;
					}
				
					k = 0;
					for(m = 0; m < InWarptid/PLTSNum; m++){
						if(GroupStore[vthreadgroupnuminwarp * Warpid + m].statevector != EMPTYVECT32){
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
				if(RecordTable1[Warpid * 32 + InWarptid].statevector != EMPTYVECT32){
					GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable1[Warpid * 32 + InWarptid];
					RecordTable1[Warpid * 32 + InWarptid].statevector = EMPTYVECT32;
					storeposition+=32;
				}

				if(RecordTable2[Warpid * 32 + InWarptid].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
					GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable2[Warpid * 32 + InWarptid];
					RecordTable2[Warpid * 32 + InWarptid].statevector = EMPTYVECT32;
					storeposition+=32;
				}

				if(RecordTable3[Warpid * 32 + InWarptid].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
					GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable3[Warpid * 32 + InWarptid];
					RecordTable2[Warpid * 32 + InWarptid].statevector = EMPTYVECT32;
					storeposition+=32;
				}

				if(storeposition < WarpCBindex[Warpid].endindex)
				{
					for(k = Warpid*32; k<(Warpid+1)*32; k++){
						if(RecordTable1[k].statevector != EMPTYVECT32){
							kickoutRecord.statevector = RecordTable1[k].statevector;
							if(atomicCAS(&(RecordTable1[k].statevector), EMPTYVECT32, RecordTable1[k].statevector)){
								kickoutRecord.localmark = RecordTable1[k].localmark;
								kickoutRecord.toevent = RecordTable1[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}

						if(RecordTable2[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
							kickoutRecord.statevector = RecordTable2[k].statevector;
							if(atomicCAS(&(RecordTable2[k].statevector), EMPTYVECT32, RecordTable2[k].statevector)){
								kickoutRecord.localmark = RecordTable2[k].localmark;
								kickoutRecord.toevent = RecordTable2[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}

						if(RecordTable3[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
							kickoutRecord.statevector = RecordTable3[k].statevector;
							if(atomicCAS(&(RecordTable3[k].statevector), EMPTYVECT32, RecordTable3[k].statevector)){
								kickoutRecord.localmark = RecordTable3[k].localmark;
								kickoutRecord.toevent = RecordTable3[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}
					}
				}

				//for the elements larger than 512, to be expanded........
				break;
			}
		
		}
		if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID].statevector == EMPTYVECT32){
			//got new stateV
			localstateV = EMPTYVECT32;
			ifgetnewstatev = false;
			while(layer < maxlayer + 1 || ifgetnewstatev == true){
				for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
					if((int)RecordTable1[i].localmark <= layer){
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
						{
							GroupStore[vthreadgroupID].localmark = RecordTable1[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable1[i].toevent;
							ifgetnewstatev = true;
							break;
						}

					}
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
						if((int)RecordTable2[i].localmark <= layer){
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
							{
								GroupStore[vthreadgroupID].localmark = RecordTable2[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable2[i].toevent;
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
						if((int)RecordTable3[i].localmark <= layer){
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
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
						if((int)RecordTable1[i].localmark <= layer){
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
							{
								GroupStore[vthreadgroupID].localmark = RecordTable1[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable1[i].toevent;
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
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
							{
								GroupStore[vthreadgroupID].localmark = RecordTable2[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable2[i].toevent;
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
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
							{
								GroupStore[vthreadgroupID].localmark = RecordTable3[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable3[i].toevent;
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
		__syncthreads();
		if(InvthreadgroupID == 0 && layer == maxlayer + 1 && ifgetnewstatev == false){
			if(Inblocktid == 0){
				for(nonewcount = 0; nonewcount < vthreadgroupnuminblock; nonewcount++){
					if(GroupStore[nonewcount].statevector != EMPTYVECT32){
						break;
					}
				}
				if(nonewcount == vthreadgroupnuminblock){
					break;
				}
			}
		}
		__syncthreads();

		if(IFDeadlockDetected)
			break;
		
	}while(!IFDeadlockDetected);

	CudaInterBlocksSyn(blockDim.x);
}

__global__ void CUDADeadlockBFSVerify(unsigned int * PG_AllLTS, unsigned int * PG_AllStates, unsigned char * PG_AllTransitions, unsigned char * PG_AllSynctransitions, unsigned int * PG_Startlist, unsigned int * PG_LTSStateEncodeBits, unsigned int * PG_LTSStateEncodeBytes, unsigned int PEventEncodeBytes,  unsigned int PLTSNum, bool * G_RESULT, unsigned int PGBucketNum, unsigned int PAllLTSStatesNum)
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
	__shared__ int maxlayer;

	__shared__ bool ifblocknostate;
	//__shared__ bool IfexistFree; //
	__shared__ int nonewcount;

	__shared__ bool haveChild;
	__shared__ int launchtime;

	//__shared__ bool ifnooutgoing[16];
	//__shared__ LocalRecord RecordTable1[512];
	//__shared__ LocalRecord RecordTable2[512];
	//__shared__ LocalRecord RecordTable3[512];
	//__shared__ LocalRecord GroupStore[256];
	//__shared__ Bucket WarpCBindex[16];

	i = vthreadgroupnuminblock * PLTSNum;
	extern __shared__ bool C[]; 
	bool * syncduplicate = C;
	bool * needsyndupdetect = &syncduplicate[i];
	//bool * needsyndupdetect = C + i;
	bool * ifnooutgoing = &needsyndupdetect[vthreadgroupnuminblock];
	unsigned int * SynEventInteractive = (unsigned int *)&ifnooutgoing[i];
	//unsigned int * SynEventInteractive = (unsigned int *)&needsyndupdetect[vthreadgroupnuminblock];
	LocalRecord * RecordTable1 = (LocalRecord *)&(SynEventInteractive[i]);
	LocalRecord * RecordTable2 = &RecordTable1[blockDim.x];
	LocalRecord * RecordTable3 = &RecordTable2[blockDim.x];
	LocalRecord * GroupStore = &RecordTable3[blockDim.x];
	Bucket * WarpCBindex = (Bucket *)&GroupStore[vthreadgroupnuminblock];
	
	if(Inblocktid == 0){
		for(i = 0; i < vthreadgroupnuminblock * PLTSNum; i++){
			ifnooutgoing[i] = false; 
			SynEventInteractive[i] = EMPTYVECT32;
		}
		maxlayer=0;
		nonewcount = 0;
		haveChild = false;
		launchtime = 0;
		ifblocknostate = false;
	
		for(i=0; i<WarpNum; i++){
			WarpCBindex[i].beginindex = 0;
			WarpCBindex[i].endindex = 0;
		}
	}

	__syncthreads();

	if(Ingridtid == 0){
		GlobalbucketCount = new unsigned int[PGBucketNum];
		GlobalbucketIndex = new unsigned int[PGBucketNum];
		GlobalOpenHash = new LocalRecord[blockDim.x * PLTSNum * 2];
		GlobalBuckets = new Bucket[PLTSNum * 2];
		GlobalVisitedHash = new LocalRecord[PAllLTSStatesNum * 10];
		communicationGstore = new Bucket[100];
		
		for(i = 0; i < blockDim.x * PLTSNum * 2; i++)
			GlobalOpenHash[i].statevector = EMPTYVECT32;

		for(i = 0; i < PAllLTSStatesNum * 10; i++)
			GlobalVisitedHash[i].statevector = EMPTYVECT32;

		for(i = 0; i < PLTSNum * 2; i++){
			GlobalBuckets[i].beginindex = i * blockDim.x;
			GlobalBuckets[i].endindex = (i+1)*blockDim.x - 1;
		}
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

	CudaInterBlocksSyn(gridDim.x);
	//while(GroupStore[vthreadgroupID].statevector == EMPTYVECT32);

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
								SynTwoStates(&localstateV, i, SynEventInteractive[i + vthreadgroupID * PLTSNum], PG_LTSStateEncodeBits, GroupStore[vthreadgroupID].statevector, PG_AllLTS, PG_AllStates, PG_AllTransitions, PG_AllSynctransitions, PEventEncodeBytes, PG_LTSStateEncodeBytes[i]);
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
			ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
		}

		if(InvthreadgroupID == 0){
			for(i = 0; i < PLTSNum; i++){
				if(!ifnooutgoing[i + vthreadgroupID * PLTSNum] && !Ifcollisionhappens&&!ifglobaldup)
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
			if(haveChild)
				cudaDeviceSynchronize();

			if(IFDeadlockDetected)
				break;
			//load new kernel, copy data back
			unsigned int myareacount = 0;
			

			globalbuckethash = Buckethash((unsigned int)(blockIdx.x)) + openvisitedborder;

			if(InWarptid == 0){
				for(m = Warpid*32; m<(Warpid + 1)*32; m++){
					if(RecordTable1[m].statevector != EMPTYVECT32)
						myareacount++;
					if(RecordTable2[m].statevector != EMPTYVECT32)
						myareacount++;
					if(RecordTable3[m].statevector != EMPTYVECT32)
						myareacount++;
				}
				
				WarpCBindex[Warpid].beginindex = atomicAdd(&GlobalbucketIndex[globalbuckethash], myareacount);
				WarpCBindex[Warpid].endindex = WarpCBindex[Warpid].beginindex + myareacount;
				atomicAdd(&GlobalbucketCount[globalbuckethash], myareacount);
			}

			storeposition = WarpCBindex[Warpid].beginindex + InWarptid;
			if(RecordTable1[Warpid * 32 + InWarptid].statevector != EMPTYVECT32){
				GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable1[Warpid * 32 + InWarptid];
				RecordTable1[Warpid * 32 + InWarptid].statevector = EMPTYVECT32;
				storeposition+=32;
			}

			if(RecordTable2[Warpid * 32 + InWarptid].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
				GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable2[Warpid * 32 + InWarptid];
				RecordTable2[Warpid * 32 + InWarptid].statevector = EMPTYVECT32;
				storeposition+=32;
			}

			if(RecordTable3[Warpid * 32 + InWarptid].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
				GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = RecordTable3[Warpid * 32 + InWarptid];
				RecordTable2[Warpid * 32 + InWarptid].statevector = EMPTYVECT32;
				storeposition+=32;
			}

			if(storeposition < WarpCBindex[Warpid].endindex)
			{
				for(k = Warpid*32; k<(Warpid+1)*32; k++){
					if(RecordTable1[k].statevector != EMPTYVECT32){
						kickoutRecord.statevector = RecordTable1[k].statevector;
						if(atomicCAS(&(RecordTable1[k].statevector), EMPTYVECT32, RecordTable1[k].statevector)){
							kickoutRecord.localmark = RecordTable1[k].localmark;
							kickoutRecord.toevent = RecordTable1[k].toevent;
							GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
							storeposition+=32;
						}
					}

					if(RecordTable2[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
						kickoutRecord.statevector = RecordTable2[k].statevector;
						if(atomicCAS(&(RecordTable2[k].statevector), EMPTYVECT32, RecordTable2[k].statevector)){
							kickoutRecord.localmark = RecordTable2[k].localmark;
							kickoutRecord.toevent = RecordTable2[k].toevent;
							GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
							storeposition+=32;
						}
					}

					if(RecordTable3[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
						kickoutRecord.statevector = RecordTable3[k].statevector;
						if(atomicCAS(&(RecordTable3[k].statevector), EMPTYVECT32, RecordTable3[k].statevector)){
							kickoutRecord.localmark = RecordTable3[k].localmark;
							kickoutRecord.toevent = RecordTable3[k].toevent;
							GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
							storeposition+=32;
						}
					}
				}
			}

			//for the elements larger than 512, to be expanded........

			//launch new kernel
			if(Inblocktid == launchtime){
				if(GlobalbucketCount[globalbuckethash]*PLTSNum % 512 == 0){
					m = (GlobalbucketCount[globalbuckethash]*PLTSNum) / 512;
				}else{
					m = (GlobalbucketCount[globalbuckethash]*PLTSNum) / 512 + 1;
				}
				if(launchtime > 0){
					i=0;
					for(k = communicationGstore[blockIdx.x].beginindex; k < communicationGstore[blockIdx.x].endindex; k++){
						i+=GlobalbucketCount[k];
					}
					if(i*PLTSNum % 512 == 0){
						m += i*PLTSNum / 512;
					}else{
						m += (i*PLTSNum / 512 +1);
					}
				}
				dim3 cgridstructure(m,1,1);
				dim3 cblockstructure(512,1,1);
				//CUDADeadlockBFSVerifyChild<<<cgridstructure, cblockstructure>>>();
				launchtime++;
				haveChild = true;
			}
			

		}
		

		if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID].statevector == EMPTYVECT32){
			//got new stateV
			localstateV = EMPTYVECT32;
			ifgetnewstatev = false;
			while(layer < maxlayer + 1 || ifgetnewstatev == true){
				for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
					if((int)RecordTable1[i].localmark == layer){
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32) != EMPTYVECT32))
						{
							GroupStore[vthreadgroupID].localmark = RecordTable1[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable1[i].toevent;
							ifgetnewstatev = true;
							break;
						}

					}
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
						if((int)RecordTable2[i].localmark == layer){
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								GroupStore[vthreadgroupID].localmark = RecordTable2[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable2[i].toevent;
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
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								GroupStore[vthreadgroupID].localmark = RecordTable3[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable3[i].toevent;
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
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32) != EMPTYVECT32))
							{
								GroupStore[vthreadgroupID].localmark = RecordTable1[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable1[i].toevent;
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
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
							{
								GroupStore[vthreadgroupID].localmark = RecordTable2[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable2[i].toevent;
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
							if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
							{
								GroupStore[vthreadgroupID].localmark = RecordTable3[i].localmark;
								GroupStore[vthreadgroupID].toevent = RecordTable3[i].toevent;
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
		__syncthreads();
		if(InvthreadgroupID == 0 && layer == maxlayer + 1 && ifgetnewstatev == false){
			if(Inblocktid == 0){
				for(nonewcount = 0; nonewcount < vthreadgroupnuminblock; nonewcount++){
					if(GroupStore[nonewcount].statevector != EMPTYVECT32){
						break;
					}
				}
				if(nonewcount == vthreadgroupnuminblock){
					cudaDeviceSynchronize();
					haveChild = false;
				}
			}
		}
		__syncthreads();

		if(IFDeadlockDetected)
			break;

		if(nonewcount == vthreadgroupnuminblock){
			//get new state again, if no, block stop.
			if(InvthreadgroupID == 0){
				//got new stateV
				if(vthreadgroupID < GlobalbucketCount[communicationGstore[blockIdx.x].beginindex])
				{
					globalbuckethash =  communicationGstore[blockIdx.x].beginindex;
					storeposition = vthreadgroupID;
				}

				layer = communicationlayer[blockIdx.x];
				
				/*while(true){   //do I need to bfs strictly according to the layer?
					if(GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition].localmark != layer){
						GroupStore[vthreadgroupID] = GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition];
						GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition].statevector = EMPTYVECT32;
						break;
					}else{
						storeposition += vthreadgroupnuminblock;
					}

					if(storeposition > GlobalbucketCount[globalbuckethash]){
						globalbuckethash++;
						storeposition=vthreadgroupID;
					}

					if(globalbuckethash > communicationGstore[blockIdx.x].endindex){
						layer++;
					}
				
				}*/
				GroupStore[vthreadgroupID] = GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition]; 
				if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID].statevector == EMPTYVECT32)
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
				if(k*PLTSNum % 512 == 0)
					m = (k*PLTSNum)/512;
				else
					m = (k*PLTSNum)/512 + 1;

				dim3 gridstruc(m,1,1);
				dim3 blockstruc(512,1,1);
				//CUDADeadlockBFSVerifyChild<<<gridstruc, blockstruc>>>();
				launchtime++;
				haveChild=true;
			}
		}
			
		
		
	}while(!IFDeadlockDetected);

	CudaInterBlocksSyn(blockDim.x);
	if(!IFDeadlockDetected){
		*G_RESULT = false;
	}else{
		*G_RESULT = true;
	}
}

void NewStateV(unsigned int * targetV, int tindex, int * index, unsigned char * Atrans, unsigned int * bitwidth, unsigned int bytewidth)
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
	* index += bytewidth;
}

void DecodeTransitions(unsigned char * outgoingT, int beginindex, unsigned int * Tostate, unsigned int * Tevent, unsigned int Eventwidth, unsigned int Statewidth)
{
	unsigned int stateendbyte, eventendbyte;
	stateendbyte = beginindex + Eventwidth + Statewidth;
	eventendbyte = beginindex + Eventwidth;
	int i;
	unsigned char tmp[4];
	for(i = 0; i < 4; i++){
		tmp[i] = (char)0x00;
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


bool GetAllsuccessors(unsigned int * AllLTS, unsigned int * Allstates, unsigned char * Alltransitions, unsigned int ltsindex, unsigned int sindex, Nodemark * result)
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

unsigned int GetStateinVec(int index, unsigned int svec, unsigned int * stateencodebits)
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

int HostGenerateStateSpace(int LTSNum, unsigned int * H_AllLTS, unsigned int * H_AllStates, unsigned char * H_AllTransitions, unsigned char * H_AllSynctrans, unsigned int ** RecordList, unsigned int RequestNum, unsigned int H_InitialStateV, unsigned int * H_LTSStateEncodeBits, unsigned int * H_LTSStateEncodeBytes, unsigned int HEventEncodeBytes)
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
	unsigned int Syncposition;

	queue<unsigned int> VisitedS;

	unsigned int newStateV;
	unsigned int * succStateV;
	unsigned int * tmpStateV;
	unsigned int newState;
	unsigned int belonglts;
	unsigned int transevent;

	unsigned int *tmp;
	
	bool ifoutgoing;
	int ifoutgoingcount;
	Nodemark allsucc;
	
	SuccessorCount = 1;
	Taskqueue.push(H_InitialStateV);
	while(SuccessorCount < RequestNum){
		newStateV = Taskqueue.front();
		ifoutgoingcount = 0;
		for(i = 0; i < LTSNum; i++){
			ifoutgoing = false;
			GetStateinVec(i, newStateV, &newState);
			ifoutgoing = GetAllsuccessors(H_AllLTS, H_AllStates, H_AllTransitions, belonglts, newState, &allsucc);
			if(!ifoutgoing){
				ifoutgoingcount++;
				continue;
			}

			m = allsucc.beginbyte;
			while(m < allsucc.endbyte){
				succStateV = new unsigned int[1];
				NewStateV(succStateV, i, &m, H_AllTransitions, H_LTSStateEncodeBits, H_LTSStateEncodeBytes[belonglts]);
				if(Taskset.insert(*succStateV).second){
					Taskqueue.push(*succStateV);
					SuccessorCount++;
				}
			}

			k = allsucc.synbeginbyte;
			while(k < allsucc.synendbyte){
				succStateV = new unsigned int[1];
				DecodeTransitions(H_AllSynctrans, k, &newState, &transevent, HEventEncodeBytes, H_LTSStateEncodeBytes[belonglts]);
				NewStateV(succStateV, i, &k, H_AllSynctrans, H_LTSStateEncodeBits, H_LTSStateEncodeBytes[belonglts]);
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
					SynTwoStatesCPU(tmpStateV, *succStateV, i, newStateV, H_LTSStateEncodeBits);
					
				}else{
					Syncevents.push_back(transevent);
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

	*RecordList = new unsigned int[SuccessorCount];
	for(i = 0; i < SuccessorCount; i++){
		(*RecordList)[i] = Taskqueue.front();
		Taskqueue.pop();
	}

	return SuccessorCount;
}

void CallCudaBFS(unsigned int * AllLTS, unsigned int * AllStates, unsigned char * AllTransitions, unsigned char* AllSyncTrans, unsigned int H_InitialSV, unsigned int * G_LTSStateEncodeBytes, unsigned int * G_LTSStateEncodeBits, unsigned int LTSNum,unsigned int AllLTSStateNum, unsigned int AllTransLength, unsigned int AllSyncTransLength, unsigned int EventEncodeBytes)
{
	int i,j;
	unsigned int * G_AllLTS;
	unsigned int * G_AllStates;
	unsigned char * G_AllTransitions;
	unsigned char * G_AllSyncTrans;  //all trans with sync events.
	//unsigned int * G_InitialStateV;

	bool * G_DetectResult;
	
	//Choose to generate some statevectors firstly in CPU---OPTIONAL
	unsigned int * G_Startlist; 
	
	LocalRecord * H_Globalhash;
	Bucket * H_GlobalBuckets;
	//Choose to generate some statevectors firstly in CPU---OPTIONAL
	unsigned int * H_Startlist;

	unsigned int * H_GlobalVisitedhash;

	unsigned int * LTSStateNum = new unsigned int[LTSNum];
	unsigned int * LTSStateEncodeBits = new unsigned int[LTSNum];
	unsigned int * LTSStateEncodeBytes = new unsigned int[LTSNum];

	unsigned int Startblocknum;
	unsigned int Startthreadnum1block;
	unsigned int Startthreadgroupnum;

	unsigned int H_GlobalbucketNum;
	//unsigned int * G_GlobalbucketNum;

	unsigned int * G_AllltsstateNum;

	int rv[8];
	srand(time(NULL));
	for(i = 0; i < 8; i++){
		rv[i] = rand();
	}

	cudaSetDevice(0);
	/*if(!InitCUDA()){
	    printf("Sorry,CUDA has not been initialized.\n");
	    exit(NULL);
    }*/

	H_Globalhash = new LocalRecord[AllLTSStateNum * 5];
	H_GlobalBuckets = new Bucket[LTSNum * 2];

	H_GlobalVisitedhash = new unsigned int[AllLTSStateNum * 10];
	
	Startthreadnum1block = 512;
	Startblocknum = 1;
	//Initialize Startlist
	Startthreadgroupnum = (((Startthreadnum1block/32)/LTSNum)*(Startthreadnum1block/32))*Startblocknum;  //initial value, not the final one?
	//H_Startlist = new unsigned int[Startthreadgroupnum];
	i = HostGenerateStateSpace(LTSNum, AllLTS,AllStates,AllTransitions, AllSyncTrans, &H_Startlist, 1, H_InitialSV, LTSStateEncodeBits, LTSStateEncodeBytes, EventEncodeBytes);
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
	H_GlobalbucketNum = LTSNum * 2;
    	cudaMalloc((void **)&G_AllLTS, sizeof(unsigned int) * LTSNum);
	cudaMalloc((void **)&G_AllStates, sizeof(unsigned int) * AllLTSStateNum);
	cudaMalloc((void **)&G_AllTransitions, sizeof(unsigned char) * AllTransLength);
	cudaMalloc((void **)&G_AllSyncTrans,sizeof(unsigned char) * AllSyncTransLength);
	cudaMalloc((void **)&G_LTSStateEncodeBytes, sizeof(unsigned int) * LTSNum);
	cudaMalloc((void **)&G_LTSStateEncodeBits, sizeof(unsigned int) * LTSNum);
	cudaMalloc((void **)&G_Startlist, sizeof(unsigned int) * Startthreadgroupnum);
	//cudaMalloc((unsigned int *)&G_InitialStateV, sizeof(int));

	cudaMalloc((void **)&G_DetectResult, sizeof(bool));

	cudaMemcpy(G_AllLTS, AllLTS, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllStates, AllStates, sizeof(unsigned int) * AllLTSStateNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllTransitions, AllTransitions, sizeof(unsigned char) * AllTransLength, cudaMemcpyHostToDevice);
	cudaMemcpy(G_AllSyncTrans, AllSyncTrans, sizeof(unsigned char) * AllSyncTransLength, cudaMemcpyHostToDevice);
	//cudaMemcpy(G_InitialStateV, &H_InitialSV, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(G_LTSStateEncodeBytes, LTSStateEncodeBytes, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_LTSStateEncodeBits, LTSStateEncodeBits, sizeof(unsigned int) * LTSNum, cudaMemcpyHostToDevice);
	cudaMemcpy(G_Startlist, H_Startlist, sizeof(unsigned int) * Startthreadgroupnum, cudaMemcpyHostToDevice);


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


	cudaBindTexture(0, LTSOFFSET, G_AllLTS);
	cudaBindTexture(0, STATEOFFSET, G_AllStates);
	//cudaBindTexture(0, OUTGOINGDETAIL, G_AllTransitions);  //how texture memory can accelerate the access rate need to be explored
	//cudaBindTexture(0, SYNCOUTGOING, G_AllSyncTrans);
	cudaBindTexture(0, STATEENCODE, G_LTSStateEncodeBytes);

	dim3 g(1,1,1);
	dim3 b(512,1,1);
	CUDADeadlockBFSVerify<<<g, b, 48>>>(G_AllLTS, G_AllStates, G_AllTransitions, G_AllSyncTrans, G_Startlist, G_LTSStateEncodeBits, G_LTSStateEncodeBytes, EventEncodeBytes, LTSNum, G_DetectResult, H_GlobalbucketNum, AllLTSStateNum);
	
	cudaUnbindTexture(LTSOFFSET);
	cudaUnbindTexture(STATEOFFSET);
	//cudaUnbindTexture(OUTGOINGDETAIL);
	cudaUnbindTexture(STATEENCODE);

	cudaFree(G_AllLTS);
	cudaFree(G_AllStates);
	cudaFree(G_AllTransitions);
	//udaFree(GlobalBuckets);
	//cudaFree(GlobalOpenHash);
	//cudaFree(GlobalVisitedHash);
	free(AllLTS);
	free(AllStates);
	free(AllTransitions);
	free(H_GlobalBuckets);
	free(H_Globalhash);
}

int main()
{
	//read data from file
	int i;
	unsigned int * AllLTS;
	unsigned int * AllStates;
	unsigned char * AllTransitions;
	unsigned char * AllSyncTrans;

	unsigned int InitialV;
	unsigned int LTSNum;
	unsigned int StatesNUM;
	unsigned int AlltransNum;
	unsigned int AllsynctransNum;
	//unsigned int Synindexencodebyte;
	//unsigned int LTSEncodebyte;
	unsigned int EventEncodebyte;

	unsigned int * LTSStateEncodebits;
	unsigned int * LTSStateEncodebytes;

	ifstream file1; //for all LTS
	ifstream file2; //for All states
	ifstream file3; //for all trans
	ifstream file4; //for all sync trans;
	ifstream file5; //for other parameters

	file1.open("./test/encode/alllts.txt");
	file2.open("./test/encode/allstates.txt");
	file3.open("./test/encode/alltrans.txt");
	file4.open("./test/encode/allsynctrans.txt");
	file5.open("./test/encode/parameters.txt");

	//parameters
	file5>>InitialV;
	file5>>LTSNum;
	file5>>StatesNUM;
	file5>>AlltransNum;
	file5>>AllsynctransNum;
	//file5>>Synindexencodebyte;
	//file5>>LTSEncodebyte;
	file5>>EventEncodebyte;

	LTSStateEncodebits = new unsigned int[LTSNum];
	LTSStateEncodebytes = new unsigned int[LTSNum];

	for(i=0; i < LTSNum; i++){
		file5>>LTSStateEncodebits[i];
	}

	for(i=0; i < LTSNum; i++){
		file5>>LTSStateEncodebytes[i];
	}

	AllLTS = new unsigned int[LTSNum];
	AllStates = new unsigned int[StatesNUM];
	AllTransitions = new unsigned char[AlltransNum];
	AllSyncTrans = new unsigned char[AllsynctransNum];

	file5.close();

	for(i=0; i <LTSNum; i++){
		file1>>AllLTS[i];
	}
	file1.close();

	for(i=0; i < StatesNUM; i++){
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

	CallCudaBFS(AllLTS,AllStates,AllTransitions,AllSyncTrans,InitialV,LTSStateEncodebytes,LTSStateEncodebits, LTSNum, StatesNUM,AlltransNum,AllsynctransNum,EventEncodebyte);

}


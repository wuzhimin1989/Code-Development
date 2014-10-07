
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
	unsigned int statevector;

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
	unsigned int beginInt;
	unsigned int endInt;
	unsigned int synbeginInt;
	unsigned int synendInt;

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

		for( i = replaceendbyte - 1; i > replacebeginbyte; i--)
			tostate = tostate | (OutgoingTs[i] << 8 * (replaceendbyte - 1 - i));
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

	if((EEncode+Secode)*(*count + 2) > 32){
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

__device__ unsigned int CudaGenerateKey(unsigned int KV,  int snum)
{
	return KV;

}

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

__device__ bool CudaHashStore(LocalRecord beginHV, unsigned int layer, unsigned int PLTSNum, LocalRecord * T1, LocalRecord * T2, LocalRecord * T3, LocalRecord * RkickoutRecord){
	unsigned int localKey, localhash;
	LocalRecord kickoutRecord;
	char tmp;

	localKey = CudaGenerateKey(beginHV.statevector, PLTSNum);
	localhash = Localhash1(localKey);
	if(atomicCAS(&(T1[localhash].statevector), EMPTYVECT32, beginHV.statevector)!=EMPTYVECT32){
		if(T1[localhash].statevector == beginHV.statevector){
			return false;
		}else{
			kickoutRecord.statevector = atomicExch(&(T1[localhash].statevector), beginHV.statevector);
			kickoutRecord.localmark = T1[localhash].localmark;
			T1[localhash].localmark = beginHV.localmark;

			localKey = CudaGenerateKey(kickoutRecord.statevector, PLTSNum);
			localhash = Localhash2(localKey);
			if(atomicCAS(&(T2[localhash].statevector), EMPTYVECT32, kickoutRecord.statevector)==EMPTYVECT32){
				T2[localhash].localmark = kickoutRecord.localmark;
			}else{
				if(T2[localhash].statevector == kickoutRecord.statevector){
					return false;
				}else{
					kickoutRecord.statevector = atomicExch(&(T2[localhash].statevector), kickoutRecord.statevector);
					tmp = T2[localhash].localmark;
					T2[localhash].localmark = kickoutRecord.localmark;
					kickoutRecord.localmark = tmp;

					localKey = CudaGenerateKey(kickoutRecord.statevector, PLTSNum);
					localhash = Localhash3(localKey);
					if(atomicCAS(&(T3[localhash].statevector), EMPTYVECT32, kickoutRecord.statevector)==EMPTYVECT32){
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

							localKey = CudaGenerateKey(kickoutRecord.statevector, PLTSNum);
							localhash = Localhash1(localKey);
							if(atomicCAS(&(T1[localhash].statevector), EMPTYVECT32, kickoutRecord.statevector)==EMPTYVECT32){
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

	return false;
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
		return false;
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

__device__ bool CudaVisitedGlobalHashstore(LocalRecord * HT, unsigned int hasbucket, unsigned int hashv, LocalRecord insertedrecord, unsigned int ltsnum){
	Bucket buckethash;
	LocalRecord kickoutRecord;

	unsigned int kickoutkey, kickouthash1, kickouthash2, kickouthash3;

	char tmp;

	while(true){
		buckethash = GlobalBuckets[hasbucket];
		if(atomicCAS(&(HT[buckethash.beginindex + hashv].statevector),EMPTYVECT32,insertedrecord.statevector)==EMPTYVECT32){
			HT[buckethash.beginindex + hashv].localmark = insertedrecord.localmark;
			break;
		}else{
			kickoutRecord.statevector = atomicExch(&(HT[buckethash.beginindex + hashv].statevector), kickoutRecord.statevector);
			tmp = HT[buckethash.beginindex + hashv].localmark;
			HT[buckethash.beginindex + hashv].localmark = insertedrecord.localmark;
			kickoutRecord.localmark = tmp;

			kickoutkey = CudaGenerateKey(kickoutRecord.statevector, ltsnum);
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

			if(atomicCAS(&(HT[buckethash.beginindex + hashv].statevector), EMPTYVECT32, kickoutRecord.statevector) == EMPTYVECT32){
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
	LocalRecord kickoutRecord;
	LocalRecord insertRecord;
	LocalRecord visitedRecord;
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
	LocalRecord * RecordTable1 = (LocalRecord *)&(SynStateInteractive[vthreadgroupnuminblock*PLTSNum]);
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
		
		for(i = 0; i < vthreadgroupnuminblock; i++)
			GroupStore[i].statevector = EMPTYVECT32;
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
		GlobalOpenHash[globalbuckethash].statevector = EMPTYVECT32;
	}
	
	do{
		if(GroupStore[vthreadgroupID].statevector != EMPTYVECT32){
			layer = (unsigned int)GroupStore[vthreadgroupID].localmark;
		
			localstate = CudaGetStateinVec(InvthreadgroupID, GroupStore[vthreadgroupID].statevector);
			belonglts = InvthreadgroupID;
			ifanyoutgoing = CudaGetAllsuccessors(belonglts, localstate-1, &SuccessorMark);
			ifglobaldup = false;
			//The successor generation consists of two steps: 1. For trans in alltransitions, process them directly. 2.For trans in allsynctrans, parallel sync is needed.
			if(ifanyoutgoing){
				outgoingcount=0;
				i = SuccessorMark.beginInt;
				//calculate global hash position for visited stateV
				if(InvthreadgroupID == 0){
					globalbuckethash = Buckethash(GroupStore[vthreadgroupID].statevector);
					hkey = CudaGenerateKey(GroupStore[vthreadgroupID].statevector, PLTSNum);
					ifglobaldup = CudaVisitedGlobalHashcal(GlobalVisitedHash, GlobalBuckets[globalbuckethash],hkey, GroupStore[vthreadgroupID], &visitedstore);
				}

				localstateV = GroupStore[vthreadgroupID].statevector;
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
						insertRecord.localmark = (char)(layer+1);
						insertRecord.statevector = localstateV;
					
					//hash store and duplicate elimination module.....
						Ifcollisionhappens = CudaHashStore(insertRecord, layer, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
						outgoingcount++;
					}
					localstateV = GroupStore[vthreadgroupID].statevector;
					if(Ifcollisionhappens){
						break;
					}
			
			      }

				//synchronization part
				j = SuccessorMark.synbeginInt;
			
				if(!Ifcollisionhappens){
					bool  ifmatch;
					int tmpcount=0;
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
							CudaDecodeTransitions(1, j, tmpcount, &SynStateInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], &SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], tex1Dfetch(TRANSEBYTES,belonglts), tex1Dfetch(LTSSTATEBITS, belonglts));
							
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
							tmpcount++;
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
							continue;
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
							insertRecord.localmark = (char)(layer+1);
							insertRecord.statevector = localstateV;

							if(!Ifcollisionhappens)
							{
								Ifcollisionhappens = CudaHashStore(insertRecord, layer,  PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
								outgoingcount++;
							}
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
							if(Ifcollisionhappens){
								for(k = 511; k > 0; k--){
									if(kickoutRecord.statevector != EMPTYVECT32){
										if(atomicCAS(&(RecordTable3[k].statevector), EMPTYVECT32, kickoutRecord.statevector)==EMPTYVECT32){
											RecordTable3[k].localmark = (char)(layer + 1);
										}
										kickoutRecord.statevector = EMPTYVECT32;
									}else{
										if(atomicCAS(&(RecordTable3[k].statevector), EMPTYVECT32, localstateV) == EMPTYVECT32){
											RecordTable3[k].localmark = (char)(layer + 1);
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
						localstateV = GroupStore[vthreadgroupID].statevector;
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
					IFDeadlockDetected = true;
					break;
				}
			}
		}
		CudaInterBlocksSyn(gridDim.x);
		if(GroupStore[vthreadgroupID].statevector != EMPTYVECT32){
			if(!IFDeadlockDetected && InWarptid == 0&&!Ifcollisionhappens&&!ifglobaldup){
				//copy visited state to global memory
				CudaVisitedGlobalHashstore(GlobalVisitedHash, globalbuckethash, visitedstore, GroupStore[vthreadgroupID], PLTSNum);
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
							if(atomicCAS(&(RecordTable1[k].statevector), kickoutRecord.statevector, EMPTYVECT32) == kickoutRecord.statevector){
								kickoutRecord.localmark = RecordTable1[k].localmark;
								kickoutRecord.toevent = RecordTable1[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}

						if(RecordTable2[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
							kickoutRecord.statevector = RecordTable2[k].statevector;
							if(atomicCAS(&(RecordTable2[k].statevector), kickoutRecord.statevector, EMPTYVECT32) == kickoutRecord.statevector){
								kickoutRecord.localmark = RecordTable2[k].localmark;
								kickoutRecord.toevent = RecordTable2[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}

						if(RecordTable3[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
							kickoutRecord.statevector = RecordTable3[k].statevector;
							if(atomicCAS(&(RecordTable3[k].statevector), kickoutRecord.statevector, EMPTYVECT32) == kickoutRecord.statevector){
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
		if(IFDeadlockDetected)
			break;

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

	unsigned int tmpoutT;

	int outgoingcount;

	unsigned int offsetborder; //used to mark the border of successors.
	bool ifanyoutgoing, ifgetnewstatev, ifglobaldup; //ifglobaldup means if this state is duplicated

	int vthreadgroupnuminblock;
	int vthreadgroupnuminwarp;
	unsigned char tmpT[4];

	unsigned int localKey, localhash;
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
	LocalRecord * RecordTable1 = (LocalRecord *)&(SynStateInteractive[i]);
	LocalRecord * RecordTable2 = &RecordTable1[blockDim.x];
	LocalRecord * RecordTable3 = &RecordTable2[blockDim.x];
	LocalRecord * GroupStore = &RecordTable3[blockDim.x];
	Bucket * WarpCBindex = (Bucket *)&GroupStore[vthreadgroupnuminblock];
	
	if(Inblocktid == 0){
		for(i = 0; i < vthreadgroupnuminblock * PLTSNum; i++){
			ifnooutgoing[i] = false; 
			SynEventInteractive[i] = EMPTYVECT32;
		}
		for(i = 0; i < blockDim.x; i++){
			RecordTable1[i].statevector = EMPTYVECT32;
			RecordTable2[i].statevector = EMPTYVECT32;
			RecordTable3[i].statevector = EMPTYVECT32;
		}

		for(i = 0; i < vthreadgroupnuminblock; i++)
			GroupStore[i].statevector = EMPTYVECT32;
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
		GlobalBucketNum = PGBucketNum;
		GlobalOpenHash = new LocalRecord[blockDim.x * 3 * PLTSNum * 4 ];
		GlobalBuckets = new Bucket[GlobalBucketNum];
		GlobalVisitedHash = new LocalRecord[blockDim.x * 3 * PLTSNum * 4]; //bucket/2
		communicationGstore = new Bucket[100];
		
		for(i = 0; i < blockDim.x * 3 * PLTSNum * 4; i++)
			GlobalOpenHash[i].statevector = EMPTYVECT32;

		for(i = 0; i < blockDim.x * 3 * PLTSNum * 4; i++)
			GlobalVisitedHash[i].statevector = EMPTYVECT32;

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
	layer = 0;
	if(InvthreadgroupID == 0 && vthreadgroupID < StartNum){
		getindex = vthreadgroupnuminblock * blockIdx.x + vthreadgroupID;
		GroupStore[vthreadgroupID].statevector = PG_Startlist[getindex];
		GroupStore[vthreadgroupID].localmark = GroupStore[vthreadgroupID].localmark | layer;
		needsyndupdetect[vthreadgroupID] = false;
	}

	CudaInterBlocksSyn(gridDim.x);
	//while(GroupStore[vthreadgroupID].statevector == EMPTYVECT32);

	do{
		if(GroupStore[vthreadgroupID].statevector != EMPTYVECT32){
			localstate = CudaGetStateinVec(InvthreadgroupID, GroupStore[vthreadgroupID].statevector);

			belonglts = InvthreadgroupID;
			ifanyoutgoing = CudaGetAllsuccessors(belonglts, localstate-1, &SuccessorMark);
			ifglobaldup = false;
			//The successor generation consists of two steps: 1. For trans in alltransitions, process them directly. 2.For trans in allsynctrans, parallel sync is needed.
			if(ifanyoutgoing){
				outgoingcount = 0;
				i = SuccessorMark.beginInt;
				//calculate global hash position for visited stateV
				if(InvthreadgroupID == 0){
					globalbuckethash = Buckethash(GroupStore[vthreadgroupID].statevector);
					hkey = CudaGenerateKey(GroupStore[vthreadgroupID].statevector,  PLTSNum);
					ifglobaldup = CudaVisitedGlobalHashcal(GlobalVisitedHash, GlobalBuckets[globalbuckethash],hkey, GroupStore[vthreadgroupID], &visitedstore);
				}

				localstateV = GroupStore[vthreadgroupID].statevector;
				visitedRecord.statevector = localstateV;
				visitedRecord.localmark = (char)layer;

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
						insertRecord.localmark = (char)(layer+1);
						insertRecord.statevector = localstateV;
					
						//hash store and duplicate elimination module.....
						Ifcollisionhappens = CudaHashStore(insertRecord, layer, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
						outgoingcount++;
					
					}
					localstateV = GroupStore[vthreadgroupID].statevector;
					if(Ifcollisionhappens){
						break;
					}
			
				}
				//synchronization part
				j = SuccessorMark.synbeginInt;
			
				if(!Ifcollisionhappens && SuccessorMark.synbeginInt != SuccessorMark.synendInt && !ifglobaldup){
					bool  ifmatch;
					int tmpcount=0;
					int tmpj = 0;
					m = 0;
					x = -1;
					CudaDecodeTransitions(0,SuccessorMark.synendInt-1, (SuccessorMark.synendInt - j)*(4/tex1Dfetch(TRANSEBYTES, belonglts))-1,&localstate2, &maxtransevent, tex1Dfetch(TRANSEBYTES, belonglts), tex1Dfetch(LTSSTATEBITS, belonglts));
					while(j < SuccessorMark.synendInt){
						ifmatch = false;
						if(m == 0 && syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum]){
							if(j == SuccessorMark.synendInt)
								break;
							CudaDecodeTransitions(1, j, tmpcount, &SynStateInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], &SynEventInteractive[InvthreadgroupID + vthreadgroupID * PLTSNum], tex1Dfetch(TRANSEBYTES,belonglts), tex1Dfetch(LTSSTATEBITS, belonglts));
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
							tmpcount++;
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = false;

						}

						for(i=0; i<PLTSNum; i++){
							if(i == InvthreadgroupID)
								continue;

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
							insertRecord.localmark = (char)((int)GroupStore[vthreadgroupID].localmark+1);
							insertRecord.statevector = localstateV;

							if(!Ifcollisionhappens)
							{
								Ifcollisionhappens = CudaHashStore(insertRecord, layer, PLTSNum, RecordTable1, RecordTable2, RecordTable3, &kickoutRecord);
								outgoingcount++;
							}
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
							if(Ifcollisionhappens){
								for(k = 511; k > 0; k--){
									if(kickoutRecord.statevector != EMPTYVECT32){
										if(atomicCAS(&(RecordTable3[k].statevector), EMPTYVECT32, kickoutRecord.statevector) == EMPTYVECT32){
											RecordTable3[k].localmark = (char)(layer + 1);
										}
										kickoutRecord.statevector = EMPTYVECT32;
									}else{
										if(atomicCAS(&(RecordTable3[k].statevector), EMPTYVECT32, localstateV) == EMPTYVECT32){
											RecordTable3[k].localmark = (char)(layer + 1);
										}
									}
								}
							}
						}

						if(!ifmatch && m == 0){
							syncduplicate[InvthreadgroupID + vthreadgroupID * PLTSNum] = true;
						}
						localstateV = GroupStore[vthreadgroupID].statevector;

					}
				}
				if(outgoingcount == 0 && !ifglobaldup)
					ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			
			}else{
				ifnooutgoing[vthreadgroupID*PLTSNum + InvthreadgroupID] = true;
			}

			if(InvthreadgroupID == 0&&!ifglobaldup){
				for(i = 0; i < PLTSNum; i++){
					if(!ifnooutgoing[i + vthreadgroupID * PLTSNum] && !Ifcollisionhappens)
						break;
				}

				if(i == PLTSNum){
					IFDeadlockDetected = true;
					break;
				}
			}
		}
		CudaInterBlocksSyn(gridDim.x);

		if(IFDeadlockDetected)
			break;

		if(GroupStore[vthreadgroupID].statevector != EMPTYVECT32){
			if(!IFDeadlockDetected && InvthreadgroupID == 0&&!Ifcollisionhappens&&!ifglobaldup){
				//copy visited state to gl)obal memory
				CudaVisitedGlobalHashstore(GlobalVisitedHash, globalbuckethash, visitedstore, GroupStore[vthreadgroupID], PLTSNum);
				if(InvthreadgroupID == 0){
					GroupStore[vthreadgroupID].statevector = EMPTYVECT32;
				}
			}else if(Ifcollisionhappens){
				if(haveChild)
					cudaDeviceSynchronize();

				//if(IFDeadlockDetected)
				//	break;
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
							if(atomicCAS(&(RecordTable1[k].statevector), RecordTable1[k].statevector, EMPTYVECT32) == kickoutRecord.statevector){
								kickoutRecord.localmark = RecordTable1[k].localmark;
								kickoutRecord.toevent = RecordTable1[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}

						if(RecordTable2[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
							kickoutRecord.statevector = RecordTable2[k].statevector;
							if(atomicCAS(&(RecordTable2[k].statevector), RecordTable2[k].statevector, EMPTYVECT32) == kickoutRecord.statevector){
								kickoutRecord.localmark = RecordTable2[k].localmark;
								kickoutRecord.toevent = RecordTable2[k].toevent;
								GlobalOpenHash[GlobalBuckets[globalbuckethash].beginindex + storeposition] = kickoutRecord;
								storeposition+=32;
							}
						}

						if(RecordTable3[k].statevector != EMPTYVECT32 && storeposition < WarpCBindex[Warpid].endindex){
							kickoutRecord.statevector = RecordTable3[k].statevector;
							if(atomicCAS(&(RecordTable3[k].statevector), RecordTable3[k].statevector, EMPTYVECT32) == kickoutRecord.statevector){
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
		}
		

		__syncthreads();

		if(InvthreadgroupID == 0 && GroupStore[vthreadgroupID].statevector == EMPTYVECT32){
			//got new stateV
			localstateV = EMPTYVECT32;
			ifgetnewstatev = false;
			while(ifgetnewstatev != true){
				for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
					if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
					{
						GroupStore[vthreadgroupID].localmark = RecordTable1[i].localmark;
						GroupStore[vthreadgroupID].toevent = RecordTable1[i].toevent;
						ifgetnewstatev = true;
						break;
					}

				
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
						{
							GroupStore[vthreadgroupID].localmark = RecordTable2[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable2[i].toevent;
							ifgetnewstatev = true;
							break;
						}
						
					}
				}else{
					break;
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupID * PLTSNum; i < (vthreadgroupID+1) * PLTSNum; i++){
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
						{
							GroupStore[vthreadgroupID].localmark = RecordTable3[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable3[i].toevent;
							ifgetnewstatev = true;
							break;
						}
					}
				}else{
					break;
				}

				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable1[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
						{
							GroupStore[vthreadgroupID].localmark = RecordTable1[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable1[i].toevent;
							ifgetnewstatev = true;
							break;
						}
						
					}
				}else{
					break;
				}
				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable2[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
						{
							GroupStore[vthreadgroupID].localmark = RecordTable2[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable2[i].toevent;
							ifgetnewstatev = true;
							break;
						}
						
					}
				}else{
					break;
				}
				if(ifgetnewstatev == false){
					for(i = vthreadgroupnuminblock * PLTSNum; i<(int)(blockDim.x); i++){						
						if((GroupStore[vthreadgroupID].statevector = atomicExch(&(RecordTable3[i].statevector), EMPTYVECT32)) != EMPTYVECT32)
						{
							GroupStore[vthreadgroupID].localmark = RecordTable3[i].localmark;
							GroupStore[vthreadgroupID].toevent = RecordTable3[i].toevent;
							ifgetnewstatev = true;
							break;
						}
					}
				}else{
					break;
				}

				if(ifgetnewstatev == false){
					break;
				}
			}
		}

		//if(IFDeadlockDetected)
		//	break;

		__syncthreads();
		
		if(Inblocktid == launchtime - 1 && ifgetnewstatev == false){
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
		//if(IFDeadlockDetected)
		//	break;

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

				layer = communicationlayer[blockIdx.x];
				
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

	CudaInterBlocksSyn(gridDim.x);
	if(!IFDeadlockDetected && Ingridtid == 0){
		*G_RESULT = 1;
	}else{
		*G_RESULT = 0;
	}
}


void NewStateV(unsigned int * targetV, int tindex, int * index, int *count,  unsigned char* OutgoingTs, unsigned int * bitwidth, unsigned int OutGTbyte, unsigned int EEncode)
{
	unsigned int tmp = *targetV;
	unsigned int tostate = 0;
	int newsbeginbit = 0, endbit;
	unsigned int Secode = bitwidth[tindex];

	int i,j,replacebeginbyte, replaceendbyte;

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
		newsbeginbit += bitwidth[i];
	}

	endbit = newsbeginbit + bitwidth[tindex];

	if(Secode == 8){
		tostate = (int) OutgoingTs[replaceendbyte - 1];
		tostate = tostate << (31 - endbit);

	}else{
		tostate = 0;

		for( i = replaceendbyte - 1; i > replacebeginbyte; i--)
			tostate = tostate | (OutgoingTs[i] << 8 * (replaceendbyte - 1 - i));

		tostate = tostate << (31-Secode);
		tostate = tostate >> (31-Secode);
		tostate = tostate << (31-endbit);

	}
	
	i = tmp >> (endbit + Secode);
	i = i << (endbit + Secode);
	j = tmp << (newsbeginbit + Secode);
	j = j >> (newsbeginbit + Secode);

	* targetV = (int) (i | j | tostate);

	if((EEncode+Secode)*(*count + 1) > 32){
		* index += 1;
		*count = 0;
	}else
		(*count)++;
}

void DecodeTransitions(unsigned int * outgoingT, int beginindex, int count, unsigned int * Tostate, unsigned int * Tevent, unsigned int OutGTe, unsigned int Statebitwidth)
{
	int i, j;
	unsigned int tmp;
	unsigned int startbyte, endbyte;
	startbyte = (count * OutGTe)%4;
	endbyte = ((count + 1)*OutGTe)%4;

	if(endbyte == 0)
		endbyte = 4;

	tmp = outgoingT[beginindex];

	tmp = tmp << (startbyte - 1);
	tmp = tmp >> (startbyte + 3 - endbyte); 

	*Tostate = (tmp << 31 - Statebitwidth) >> (31- Statebitwidth);
	*Tevent = tmp >> Statebitwidth;
}



bool GetAllsuccessors(unsigned int * AllLTS, unsigned int * Allstates, unsigned int * Alltransitions, unsigned int ltsindex, unsigned int sindex, Nodemark * result)
{
	unsigned int statesbegin, transbegin, transborder, syncbegin;
	statesbegin = AllLTS[ltsindex];
	transbegin = Allstates[statesbegin + sindex];
	transborder = Allstates[statesbegin + sindex + 1];

	if(transbegin == 0 && (ltsindex != 0 || sindex !=0))
		return false;

	result->beginInt = transbegin;
	result->endInt = transborder - 4;

	result->synbeginInt = Alltransitions[transborder - 1] | Alltransitions[transborder - 2] | Alltransitions[transborder - 3] | Alltransitions[transborder - 4];

	transborder = Allstates[statesbegin + sindex + 2];

	syncbegin = Alltransitions[transborder - 1] | Alltransitions[transborder - 2] | Alltransitions[transborder - 3] | Alltransitions[transborder - 4];

	result->synendInt = syncbegin - 1;
	return true;
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

int HostGenerateStateSpace(int LTSNum, unsigned int * H_AllLTS, unsigned int * H_AllStates, unsigned int * H_AllTransitions, unsigned int * H_AllSynctrans, unsigned int ** RecordList, unsigned int RequestNum, unsigned int H_InitialStateV, unsigned int * H_LTSStateEncodeBits, unsigned int * OutgoingTEbytes, unsigned int HEventEncodeBits)
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
				NewStateV(succStateV, i, &m, &y, tmpT, H_LTSStateEncodeBits, OutgoingTEbytes[i], HEventEncodeBits );
				
				if(Taskset.insert(*succStateV).second){
					Taskqueue.push(*succStateV);
					SuccessorCount++;
				}
			}

			k = allsucc.synbeginInt;
			tmpcount = 0;
			x = -1;
			y = 0;
			while(k < allsucc.synendInt){
				succStateV = new unsigned int[1];

				DecodeTransitions(H_AllSynctrans, k, tmpcount, &newState, &transevent, OutgoingTEbytes[belonglts], H_LTSStateEncodeBits[i]);

				if(x != k){
					tmpoutT = H_AllSynctrans[k];
					tmpT[0] = (char)(tmpoutT >> 24);
					tmpT[1] = (char)(tmpoutT >> 16);
					tmpT[2] = (char)(tmpoutT >> 8);
					tmpT[3] = (char)tmpoutT;
					x = k;	
				}			
				NewStateV(succStateV, i, &k, &y, tmpT, H_LTSStateEncodeBits, OutgoingTEbytes[i], HEventEncodeBits);

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
	unsigned int H_startsize;
	unsigned int * LTSStateNum = new unsigned int[LTSNum];

	unsigned int Startblocknum;
	unsigned int Startthreadnum1block;
	unsigned int Startthreadgroupnum;

	unsigned int H_GlobalbucketNum;
	//unsigned int * G_GlobalbucketNum;

	int rv[8];
	srand(time(NULL));
	for(i = 0; i < 8; i++){
		rv[i] = rand();
	}

	cudaSetDevice(0);
	
	Startthreadnum1block = 512;
	Startblocknum = 1;
	//Initialize Startlist
	Startthreadgroupnum = (((Startthreadnum1block/32)/LTSNum)*(Startthreadnum1block/32))*Startblocknum;  //initial value, not the final one?
	
	
	i = HostGenerateStateSpace(LTSNum, AllLTS,AllStates,AllTransitions, AllSyncTrans, &H_Startlist, 1, H_InitialSV,H_LTSStateEncodeBits, OutgoingTEbytes,  EventEncodeBits);
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
	i = 512;
	cudaMemcpyToSymbol(GA1, &rv[0], sizeof(int));
	cudaMemcpyToSymbol(GA2, &rv[1], sizeof(int));
	cudaMemcpyToSymbol(GA3, &rv[2], sizeof(int));
	cudaMemcpyToSymbol(GB1, &rv[3], sizeof(int));
	cudaMemcpyToSymbol(GB2, &rv[4], sizeof(int));
	cudaMemcpyToSymbol(GB3, &rv[5], sizeof(int));
	cudaMemcpyToSymbol(TableSize, &i, sizeof(int));

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
	CUDADeadlockBFSVerify<<<g, b, 5120*sizeof(unsigned int)>>>( G_AllLTS, G_AllStates, G_AllTransitions, G_AllSyncTrans, G_Startlist, G_LTSStateEncodeBits, EventEncodeBits, G_OutgoingTEbytes, LTSNum, G_DetectResult, H_GlobalbucketNum, AllLTSStateNum, H_startsize);
	
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


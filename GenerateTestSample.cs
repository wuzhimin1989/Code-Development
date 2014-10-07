using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;


namespace GenerateTestSample
{
    class PLTS
    {
        public uint beginsynlayer;
        public uint totallayer;
        public uint totalstatenum;
        public uint LTSnum;
        public List<uint> eventlist;

        public PLTS(uint a, uint b, uint c)
        {
            beginsynlayer = a;
            LTSnum = b;
            totallayer = c;
            eventlist = new List<uint>();
        }

        public void GeneratePLTS()
        {
            int i, j, m, k;
            uint eidcount = 10;
            uint sidbegin;
            uint sid;
            uint layernum;
            uint current;
            uint Allsnum = 0;
            HashSet<uint> tmpeset = new HashSet<uint>();

    
            Queue<uint> generatelts = new Queue<uint>();
            Dictionary<uint, uint> tmpdic = new Dictionary<uint,uint>();
            List<KeyValuePair<uint,uint>> tmplist;
            
            Random rd = new Random((int)System.DateTime.Today.Ticks);
            
            generatelts.Enqueue(1);

            string path1 = "./test/Lts1.txt";
            string path2 = "./test/Lts2.txt";
            string path3 = "./test/evt.txt";
            string path4 = "./test/allsnum.txt";

            StreamWriter [] sw = new StreamWriter[2];
            sw[0] = new StreamWriter(path1, true);
            sw[1] = new StreamWriter(path2, true);

            StreamWriter sw2 = new StreamWriter(path3, true);
            StreamWriter sw3 = new StreamWriter(path4, true);

            for (i = 0; i < LTSnum; i++)
            {
                layernum = (uint)(Math.Pow(2, beginsynlayer)*2 - 1);
                for (j = 0; j < layernum; j++)
                {
                    current = generatelts.Dequeue();
                    
                    //layernum = layernum * 2;
                    sidbegin = current * 2;
                    for (m = 0; m < 2; m++)
                    {
                        sid = sidbegin + (uint)m;
                        generatelts.Enqueue(sid);
                        if (i == 0)
                        {
                            eidcount = (uint)rd.Next(11, 31);

                            if(!tmpeset.Contains(eidcount))
                            {
                                  tmpeset.Add(eidcount);
                                  if (!this.eventlist.Contains(eidcount))
                                    this.eventlist.Add(eidcount);
                                  //sw[i].Write(eidcount);
                            }
                            else
                            {
                                while (tmpeset.Contains(eidcount))
                                {
                                    eidcount = (uint)rd.Next(11, 31);
                                }
                                tmpeset.Add(eidcount);
                                if (!this.eventlist.Contains(eidcount))
                                    this.eventlist.Add(eidcount);
                                //sw[i].Write(eidcount);
                            }

                        }
                        else
                        {
                            eidcount = (uint)rd.Next(31, 51);

                            if (!tmpeset.Contains(eidcount))
                            {
                                tmpeset.Add(eidcount);
                                if (!this.eventlist.Contains(eidcount))
                                    this.eventlist.Add(eidcount);
                                //sw[i].Write(eidcount);
                            }
                            else
                            {
                                while (tmpeset.Contains(eidcount))
                                {
                                    eidcount = (uint)rd.Next(11, 31);
                                }
                                tmpeset.Add(eidcount);
                                if (!this.eventlist.Contains(eidcount))
                                    this.eventlist.Add(eidcount);
                                //sw[i].Write(eidcount);
                            }
                        }
                       
                        //sw[i].Write(" ");
                        tmpdic.Add(eidcount,sid);
                        Allsnum++;
                       
                        
                    }
                    tmpdic = tmpdic.OrderBy(x => x.Key).ToDictionary(x => x.Key, x => x.Value);
                    tmplist = tmpdic.ToList();
                    for(k = 0; k < tmplist.Count; k++){
                        sw[i].Write(tmplist[k].Key);
                        sw[i].Write(" ");
                        sw[i].Write(tmplist[k].Value);
                        sw[i].Write(" ");
                    }
                    
                    tmpeset.Clear();
                    tmpdic.Clear();
                    sw[i].Write("\n");
                }

                layernum = (uint)(Math.Pow(2, beginsynlayer + 1));
                layernum = (uint)(layernum * Math.Pow(4, totallayer - beginsynlayer) * 4 - layernum) / 3; 
                for (k = 0; k < layernum; k++ )
                {
                    current = generatelts.Dequeue();
                    sidbegin = current * 4;
                    if (rd.Next(0, 2) == 0)
                    {
                        for (m = 0; m < 4; m++)
                        {
                            sid = sidbegin + (uint)m;
                            generatelts.Enqueue(sid);
                            if (i == 0)
                            {
                                eidcount = (uint)rd.Next(11, 31);

                                if (!tmpeset.Contains(eidcount))
                                {
                                    tmpeset.Add(eidcount);
                                    if(!this.eventlist.Contains(eidcount))
                                        this.eventlist.Add(eidcount);
                                    //sw[i].Write(eidcount);
                                }
                                else
                                {
                                    while (tmpeset.Contains(eidcount))
                                    {
                                        eidcount = (uint)rd.Next(11, 31);
                                    }
                                    tmpeset.Add(eidcount);
                                    if (!this.eventlist.Contains(eidcount))
                                        this.eventlist.Add(eidcount);
                                    //sw[i].Write(eidcount);
                                }

                            }
                            else
                            {
                                eidcount = (uint)rd.Next(31, 51);

                                if (!tmpeset.Contains(eidcount))
                                {
                                    tmpeset.Add(eidcount);
                                    if (!this.eventlist.Contains(eidcount))
                                        this.eventlist.Add(eidcount);
                                   // sw[i].Write(eidcount);
                                }
                                else
                                {
                                    while (tmpeset.Contains(eidcount))
                                    {
                                        eidcount = (uint)rd.Next(31, 51);
                                    }
                                    tmpeset.Add(eidcount);
                                    if (!this.eventlist.Contains(eidcount))
                                        this.eventlist.Add(eidcount);
                                    //sw[i].Write(eidcount);
                                }
                            }
                            
                            tmpdic.Add(eidcount, sid);
                            Allsnum++;
                            //sw[i].Write(" ");
                            //sw[i].Write(sid);
                           
                        }
                        tmpdic = tmpdic.OrderBy(x => x.Key).ToDictionary(x => x.Key, x => x.Value);
                        tmplist = tmpdic.ToList();
                        for(m = 0; m < tmplist.Count; m++){
                            sw[i].Write(tmplist[m].Key);
                            sw[i].Write(" ");
                            sw[i].Write(tmplist[m].Value);
                            sw[i].Write(" ");
                        }
                    
                        tmpeset.Clear();
                        tmpdic.Clear();
                        sw[i].Write("\n");
                    }
                    else
                    {
                        sid = sidbegin;
                        generatelts.Enqueue(sid);
                        if(i == 0)
                            eidcount = (uint)rd.Next(11, 31);
                        else
                            eidcount = (uint)rd.Next(31, 51);
                        sw[i].Write(eidcount);
                        sw[i].Write(" ");
                        sw[i].Write(sid);
                       
                        sw[i].Write(" 0 1 ");

                        for (m = 1; m < 4; m++)
                        {
                            sid = sidbegin + (uint)m;
                            generatelts.Enqueue(sid);

                            eidcount = (uint)rd.Next(0, 11);
                            if (!tmpeset.Contains(eidcount))
                            {
                                //sw[i].Write(eidcount);
                                tmpeset.Add(eidcount);
                                if (!this.eventlist.Contains(eidcount))
                                    this.eventlist.Add(eidcount);
                            }
                            else
                            {
                                while (tmpeset.Contains(eidcount))
                                {
                                    eidcount = (uint)rd.Next(11, 31);
                                }
                                tmpeset.Add(eidcount);
                                if (!this.eventlist.Contains(eidcount))
                                    this.eventlist.Add(eidcount);
                                //sw[i].Write(eidcount);
                            }
                            tmpdic.Add(eidcount, sid);
                            Allsnum++;
                            //sw[i].Write(" ");
                            //sw[i].Write(sid);
                        }
                        tmpdic = tmpdic.OrderBy(x => x.Key).ToDictionary(x => x.Key, x => x.Value);
                        tmplist = tmpdic.ToList();
                        for(m = 0; m < tmplist.Count; m++){
                            sw[i].Write(tmplist[m].Key);
                            sw[i].Write(" ");
                            sw[i].Write(tmplist[m].Value);
                            sw[i].Write(" ");
                        }
                    
                        tmpeset.Clear();
                        tmpdic.Clear();
                        sw[i].Write("\n");
                    }
                }
                sw[i].Flush();
                sw[i].Close();
                generatelts.Clear();
                generatelts.Enqueue(1);
            }
            

            for (k = 0; k < this.eventlist.Count; k++)
            {
                sw2.Write(this.eventlist[k]);
                sw2.Write(" ");
            }
            sw2.Flush();
            sw2.Close();

            sw3.Write(Allsnum);
            sw3.Write("\n");
            sw3.Flush();
            sw3.Close();
        }
    }
    class OutgoingT
    {
        public uint evetID;
        public uint ToSID;

        public OutgoingT()
        {
            evetID = 0;
            ToSID = 0;
        }

        public OutgoingT(uint a, uint b)
        {
            evetID = b;
            ToSID = a;
        }
    }
    class GenerateSample
    {
        public byte[] eventlist;
        public uint[] statenum;
        public List<OutgoingT>[] outgoingtrans;
        public List<OutgoingT>[] syncoutgoingtrans;

        public uint[] AllLts;
        public uint[] AllStates;
        public List<uint> Alltrans;
        public List<uint> Allsynctrans;

        private uint[] StateEncodeBits;
        private uint[] OutgoingTEbytes;
        private uint EventEncodebits;
        private uint LTSEncodebytes;

        private uint LTSUM;
        private uint ALLSNUM;
        private uint AllTransLength; //num of int
        private uint AllsyncTlength;
        private uint InitialV;
        public uint[] InitialS;

        public GenerateSample()
        {
            LTSUM = 2;
            ALLSNUM = 8;
            statenum = new uint[LTSUM];
            AllTransLength = 0;
            AllsyncTlength = 0;

            Alltrans = new List<uint>();
            Allsynctrans = new List<uint>();
        }
        public void GetInput()
        {
            uint i, j = 0, m = 0, k, x;
            uint totalstatesnum = 0;

            bool storemark = false;

            byte[] tmp = new byte[4];
            uint tmpe;
            uint tmps;
            uint tmpT;

            StreamReader sr1 = new StreamReader(new FileStream("./test/Lts1.txt", FileMode.Open, FileAccess.Read));
            StreamReader sr2 = new StreamReader(new FileStream("./test/Lts2.txt", FileMode.Open, FileAccess.Read));
            StreamReader sr3 = new StreamReader(new FileStream("./test/evt.txt", FileMode.Open, FileAccess.Read));
            StreamReader srI1 = new StreamReader(new FileStream("./test/Lts1INI.txt", FileMode.Open, FileAccess.Read));
            StreamReader srI2 = new StreamReader(new FileStream("./test/Lts2INI.txt", FileMode.Open, FileAccess.Read));
            StreamReader srA = new StreamReader(new FileStream("./test/allsnum.txt", FileMode.Open, FileAccess.Read));

            this.AllLts[0] = 0;
            string str1 = sr1.ReadToEnd().Replace("\r", " ");
            string str2 = sr2.ReadToEnd().Replace("\r", " ");
            string str3 = sr3.ReadToEnd().Replace("\r\n", " ");
            string str4 = srA.ReadToEnd().Replace("\r\n", "");

            str1 = str1.Replace("\n", "0 0 ");
            str2 = str2.Replace("\n", "0 0 ");
            sr1.Close();
            sr2.Close();
            sr3.Close();
            srA.Close();

            string delimStr = " ";
            string[] split1 = str1.Split(delimStr.ToCharArray());
            string[] split2 = str2.Split(delimStr.ToCharArray());
            string[] split3 = str3.Split(delimStr.ToCharArray());
            string[] splitA = str4.Split(delimStr.ToCharArray());

            this.ALLSNUM = UInt32.Parse(splitA[0]);
            this.outgoingtrans = new List<OutgoingT>[this.ALLSNUM];
            this.syncoutgoingtrans = new List<OutgoingT>[this.ALLSNUM];
            for (i = 0; i < this.ALLSNUM; i++)
            {
                this.outgoingtrans[i] = new List<OutgoingT>();
                this.syncoutgoingtrans[i] = new List<OutgoingT>();
            }

            this.eventlist = new byte[split3.Length];
            this.EventEncodebits = (uint)(Math.Log(split3.Length) / Math.Log(2));
            if (Math.Pow(2, this.EventEncodebits) < split3.Length)
                this.EventEncodebits++;

            for (i = 0; i < split1.Length - 1; i++)
            {
                j = i + 1;
                if (UInt32.Parse(split1[i]) != 0)
                {
                    if (UInt32.Parse(split1[j]) != 0)
                    {
                        if (!storemark)
                            outgoingtrans[m].Add(new OutgoingT(UInt32.Parse(split1[j]), UInt32.Parse(split1[i])));
                        else
                            syncoutgoingtrans[m].Add(new OutgoingT(UInt32.Parse(split1[j]), UInt32.Parse(split1[i])));
                        i++;
                    }


                }
                else
                {
                    if (UInt32.Parse(split1[j]) == 0)
                    {
                        i++;
                        m++;
                        storemark = false;
                    }
                    else
                    {
                        storemark = true; //begin to store in sync;
                        i++;
                    }

                }

            }

            this.AllLts[1] = m + 1;
            m++;
            this.StateEncodeBits[0] = (uint)(Math.Log(this.AllLts[1]) / Math.Log(2));
            if (Math.Pow(2, this.StateEncodeBits[0]) < (this.AllLts[1] - this.AllLts[0]))
            {
                this.StateEncodeBits[0]++;
            }

            storemark = false;
            for (i = 0; i < split2.Length; i++)
            {
                j = i + 1;
                if (UInt32.Parse(split2[i]) != 0)
                {
                    if (UInt32.Parse(split2[j]) != 0)
                    {
                        if (!storemark)
                            outgoingtrans[m].Add(new OutgoingT(UInt32.Parse(split2[j]), UInt32.Parse(split2[i])));
                        else
                            syncoutgoingtrans[m].Add(new OutgoingT(UInt32.Parse(split2[j]), UInt32.Parse(split2[i])));

                        i++;
                    }


                }
                else
                {
                    if (UInt32.Parse(split2[j]) == 0)
                    {
                        i++;
                        m++;
                        storemark = false;
                    }
                    else
                    {
                        storemark = true; //begin to store in sync;
                        i++;
                    }

                }
            }

            this.AllLts[2] = m + 1;
            this.StateEncodeBits[1] = (uint)(Math.Log(this.AllLts[2] - this.AllLts[1]) / Math.Log(2));
            if (Math.Pow(2, this.StateEncodeBits[1]) < (this.AllLts[2] - this.AllLts[1]))
            {
                this.StateEncodeBits[1]++;
            }
            x = 0;  //x = total trans num
            //got all lts states;
            for (i = 0; i < 2; i++)
            {
                totalstatesnum = 0;
                for (j = this.AllLts[i]; j < this.AllLts[i + 1]; j++)
                {
                    totalstatesnum += (uint)outgoingtrans[j].Count;
                    totalstatesnum += (uint)syncoutgoingtrans[j].Count;
                }

                x += totalstatesnum;
            }

            for (i = 0; i < 2; i++)
            {
                if((this.EventEncodebits + this.StateEncodeBits[i]) <= 8){
                    this.OutgoingTEbytes[i] = 1;
                }
                else
                {
                    if ((this.EventEncodebits + this.StateEncodeBits[i]) % 8 == 0)
                        this.OutgoingTEbytes[i] = (this.EventEncodebits + this.StateEncodeBits[i]) / 8;
                    else
                        this.OutgoingTEbytes[i] = (this.EventEncodebits + this.StateEncodeBits[i]) / 8 + 1;
                }
               
            }
            //begin encode;
            this.AllStates = new uint[this.ALLSNUM+1];
            for (i = 0; i < this.ALLSNUM + 1; i++)
                this.AllStates[i] = 0;

            int tmpcount = 1;
            int tmpmark = 0;
            for (i = 0; i < 2; i++)
            {
                OutgoingT[] tmpOT;
                for (j = this.AllLts[i]; j < this.AllLts[i + 1]; j++)
                {
                    tmpT = 0;
                    tmpcount = 1;
                    this.AllStates[j] = (uint)this.Alltrans.Count;
                    tmpOT = this.syncoutgoingtrans[j].ToArray();
                    tmpmark = (int)(this.syncoutgoingtrans[j].Count - this.OutgoingTEbytes[i]);
                    for (x = 0; x < this.syncoutgoingtrans[j].Count; x++)
                    {
                        tmpe = tmpOT[x].evetID;
                        tmps = tmpOT[x].ToSID;
                        if (tmpcount < 4 / this.OutgoingTEbytes[i])
                        {
                            tmpT = tmpT | tmps | tmpe << (int)this.StateEncodeBits[i];
                            tmpT = tmpT << (int)(this.OutgoingTEbytes[i] * 8);
                            tmpcount++;

                            continue;
                        }
                        else if(tmpcount == 4/this.OutgoingTEbytes[i])
                        {
                            tmpT = tmpT | tmps | tmpe << (int)this.StateEncodeBits[i];
                            if(tmpcount == 4/this.OutgoingTEbytes[i])
                                tmpT = tmpT << (4 - (int)this.OutgoingTEbytes[i] * tmpcount) * 8;
                            tmpcount = 1;
                        }

                        this.Allsynctrans.Add(tmpT);
                        tmpT = 0;
                    }
                    if (this.syncoutgoingtrans[j].Count != (4 / this.OutgoingTEbytes[i]) && tmpcount != 1)
                    {
                        tmpT = tmpT << (4 - tmpcount * (int)this.OutgoingTEbytes[i]) * 8;
                        this.Allsynctrans.Add(tmpT);
                        tmpT = 0;
                    }
                    tmpcount = 1;
                    tmpOT = this.outgoingtrans[j].ToArray();
                    for (m = 0; m < this.outgoingtrans[j].Count; m++)
                    {
                        tmpe = tmpOT[m].evetID;
                        tmps = tmpOT[m].ToSID;

                        if (tmpcount < 4 / this.OutgoingTEbytes[i])
                        {
                            tmpT = tmpT | tmps | tmpe << (int)this.StateEncodeBits[i];
                            tmpT = tmpT << (int)(this.OutgoingTEbytes[i] * 8);
                            tmpcount++;
                            continue;
                        }
                        else if(tmpcount == 4 / this.OutgoingTEbytes[i])
                        {
                            tmpT = tmpT | tmps | tmpe << (int)this.StateEncodeBits[i];
                            if (tmpcount == 4 / this.OutgoingTEbytes[i])
                                tmpT = tmpT << (4 - (int)this.OutgoingTEbytes[i] * tmpcount) * 8;
                            tmpcount = 1;
                        }
                        this.Alltrans.Add(tmpT);
                        tmpT = 0;

                    }
                    if (this.outgoingtrans[j].Count != (4 / this.OutgoingTEbytes[i]) && tmpcount != 1)
                    {
                        tmpT = tmpT << (4 - tmpcount * (int)this.OutgoingTEbytes[i]) * 8;
                        this.Alltrans.Add(tmpT);
                        tmpT = 0;
                    }

                    //this.Alltrans.Add((uint)(this.Allsynctrans.Count - this.syncoutgoingtrans[j].Count));
                    this.Alltrans.Add((uint)(this.Allsynctrans.Count));
                    //this.AllStates[j + 1] = (uint)this.Alltrans.Count;
                }
            }
            this.AllStates[this.ALLSNUM] = (uint)this.Alltrans.Count;
            //this.AllStates[this.ALLSNUM + 1] = (uint)(this.Alltrans.Count + 1);

            this.AllTransLength = (uint)this.Alltrans.Count;
            this.AllsyncTlength = (uint)this.Allsynctrans.Count;

            string strI1 = srI1.ReadToEnd().Replace("\r\n", " ");
            string strI2 = srI2.ReadToEnd().Replace("\r\n", " ");

            string[] split4 = strI1.Split(delimStr.ToCharArray());
            string[] split5 = strI2.Split(delimStr.ToCharArray());

            uint Ini1 = UInt32.Parse(split4[0]);
            uint Ini2 = UInt32.Parse(split5[0]);

            this.InitialV = Ini1;
            this.InitialV = Ini2 | (this.InitialV << (int)this.StateEncodeBits[1]);
            this.InitialV = this.InitialV << (32 - ((int)(this.StateEncodeBits[0] + this.StateEncodeBits[1])));
        }

        public void Output2F()
        {
            string path1 = "./test/encode/parameters.txt";
            string path2 = "./test/encode/alllts.txt";
            string path3 = "./test/encode/allstates.txt";
            string path4 = "./test/encode/alltrans.txt";
            string path5 = "./test/encode/allsynctrans.txt";

            StreamWriter sw1 = new StreamWriter(path1, true);
            StreamWriter sw2 = new StreamWriter(path2, true);
            StreamWriter sw3 = new StreamWriter(path3, true);
            StreamWriter sw4 = new StreamWriter(path4, true);
            StreamWriter sw5 = new StreamWriter(path5, true);

            //write parameters;
            sw1.Write(this.InitialV);
            sw1.Write(" ");
            sw1.Write(this.LTSUM);
            sw1.Write(" ");
            sw1.Write(this.ALLSNUM);
            sw1.Write(" ");
            sw1.Write(this.AllTransLength);
            sw1.Write(" ");
            sw1.Write(this.AllsyncTlength);
            sw1.Write(" ");
            //sw1.Write(1);
            //sw1.Write(" ");
            //sw1.Write(1);
            //sw1.Write(" ");
            sw1.Write(this.EventEncodebits);
            sw1.Write(" ");

            for (int i = 0; i < 2; i++)
            {
                sw1.Write(this.StateEncodeBits[i]);
                sw1.Write(" ");
            }

            for (int i = 0; i < 2; i++)
            {
                sw1.Write(this.OutgoingTEbytes[i]);
                sw1.Write(" ");
            }
            sw1.Flush();
            sw1.Close();

            for (int i = 0; i < this.LTSUM; i++)
            {
                sw2.Write(this.AllLts[i]);
                sw2.Write(" ");
            }
            sw2.Flush();
            sw2.Close();

            for (int i = 0; i < this.ALLSNUM + 1; i++)
            {
                sw3.Write(this.AllStates[i]);
                sw3.Write(" ");
            }
            sw3.Flush();
            sw3.Close();

            for (int i = 0; i < this.AllTransLength; i++)
            {
                sw4.Write(this.Alltrans[i]);
                sw4.Write(" ");
            }
            sw4.Flush();
            sw4.Close();

            for (int i = 0; i < this.AllsyncTlength; i++)
            {
                sw5.Write(this.Allsynctrans[i]);
                sw5.Write(" ");
            }
            sw5.Flush();
            sw5.Close();

        }


        static void Main(string[] args)
        {
            //PLTS LTSSAMPLE = new PLTS(3, 2, 5);
            //LTSSAMPLE.GeneratePLTS();   

            GenerateSample SG = new GenerateSample();
           
            SG.StateEncodeBits = new uint[SG.LTSUM];
            SG.OutgoingTEbytes = new uint[SG.LTSUM];
            SG.InitialS = new uint[SG.LTSUM];
           
            SG.AllLts = new uint[3];
            SG.InitialS = new uint[3];
            SG.GetInput();
            SG.Output2F();
        }
    }
}

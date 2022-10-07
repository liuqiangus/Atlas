/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011-2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Jaume Nin <jaume.nin@cttc.cat>
 *          Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/lte-module.h"
#include "ns3/netanim-module.h"

//#include "ns3/gtk-config-store.h"

#include "ns3/edge-http-helper.h"
#include "ns3/edge-http-client.h"
#include "ns3/edge-http-server.h"
#include "ns3/edge-http-variables.h"

using namespace ns3;

/**
 * Sample simulation script for LTE+EPC. It instantiates several eNodeBs,
 * attaches one UE per eNodeB starts a flow for each UE to and from a remote host.
 * It also starts another flow between each UE pair.
 */

NS_LOG_COMPONENT_DEFINE ("my-lena-simple-epc");

///////////////////////// global variable for data collection ///////////////////////
std::vector<int> Latencies_Overall;
std::vector<int> Latencies_Uplink;
std::vector<int> Latencies_Queuing;
std::vector<int> Latencies_Compute;
std::vector<int> Latencies_Downlink;
std::vector<int> Size_Uplink;
std::vector<int> Size_Downlink;
std::vector<int> Queue_Size;
bool Verbose = false;

void
ServerConnectionEstablished (Ptr<const EdgeHttpServer>, Ptr<Socket>)
{
  if (Verbose) NS_LOG_INFO ("Client has established a connection to the server.");
}

void
MainObjectGenerated (uint32_t size)
{
  if (Verbose) NS_LOG_INFO ("Server generated a DL main object of " << size << " bytes.");
}

void
RequestGenerated (uint32_t size)
{
    Size_Uplink.push_back(size); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.

  if (Verbose) NS_LOG_INFO ("Client generated a UL request of " << size << " bytes.");
}

void
ServerRxRtt (const Time &delay, const Address &address)
{
  Latencies_Uplink.push_back(delay.GetMilliSeconds()); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.
  if (Verbose) NS_LOG_INFO ("Server received a UL request " << delay.GetMilliSeconds() << " (ms) from " << address);

}

void
ServerQueue (const Time &delay, const uint16_t &QueueSize, const Address &address)
{
  Latencies_Queuing.push_back(delay.GetMilliSeconds()); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.
  Queue_Size.push_back(QueueSize);
  if (Verbose) NS_LOG_INFO ("Server queued the UL request " << delay.GetMilliSeconds() << " (ms) from " << address);

}

void
ServerCompute (const Time &delay, const Address &address)
{
  Latencies_Compute.push_back(delay.GetMilliSeconds()); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.
  if (Verbose) NS_LOG_INFO ("Server compute the UL request " << delay.GetMilliSeconds() << " (ms) from " << address);

}

void
ServerTx (Ptr<const Packet> packet)
{
  Size_Downlink.push_back(packet->GetSize ()); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.

  if (Verbose) NS_LOG_INFO ("Server sent a DL packet of " << packet->GetSize () << " bytes.");
}

void
ClientRx (Ptr<const Packet> packet, const Address &address)
{
  if (Verbose) NS_LOG_INFO ("Client received a packet of " << packet->GetSize () << " bytes from " << address);
}

void
ClientRxDelay (const Time &delay, const Address &address)
{
  Latencies_Downlink.push_back(delay.GetMilliSeconds()); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.
  if (Verbose) NS_LOG_INFO ("Client received a packet with DL delay " << delay.GetMilliSeconds() << " (ms) from " << address);
}

void
ClientRxRtt (const Time &delay)
{
  Latencies_Overall.push_back(delay.GetMilliSeconds()); // save the RTT latency to the global vectors, which will be saved to file at the end of the simulation.
  if (Verbose) 
  {
    NS_LOG_INFO ("Client received a packet with RTT " << delay.GetMilliSeconds() << " (ms)");
  }
  else 
  { 
    std::cout<<delay.GetMilliSeconds()<<",";
    std::cout.flush();
  }
}

void
ClientMainObjectReceived (Ptr<const EdgeHttpClient>, Ptr<const Packet> packet)
{
  Ptr<Packet> p = packet->Copy ();
  EdgeHttpHeader header;
  p->RemoveHeader (header);
  if (header.GetContentLength () == p->GetSize ()
      && header.GetContentType () == EdgeHttpHeader::MAIN_OBJECT)
    {
      if (Verbose) NS_LOG_INFO ("Client has successfully received a main object of "
                   << p->GetSize () << " bytes.");
    }
  else
    {
      if (Verbose) NS_LOG_INFO ("Client failed to parse a main object. ");
    }
}

void 
WritetoFile (std::string filename)
{
  std::ofstream newFile;
  newFile.open(filename);
  newFile << "RTT -- Uplink -- Queuing -- Compute -- Downlink -- UL_DataSize -- DL_DataSize -- Queued_Size\n";
  for(int i = 0; i <= (int) Latencies_Overall.size()-1; i++)
      newFile<<Latencies_Overall.at(i)<<" "<<Latencies_Uplink.at(i)<<" "<<Latencies_Queuing.at(i)<<" "<<Latencies_Compute.at(i)<<" "<<Latencies_Downlink.at(i)<<" "<<Size_Uplink.at(i)<<" "<<Size_Downlink.at(i)<<" "<<Queue_Size.at(i)<<"\n";
  newFile.close();
}

int
main (int argc, char *argv[])
{
  int random_seed = 1;
  uint16_t numUEs = 1;
  double simtime = 10;
  uint8_t bandwidth_ul = 50; // number of PRBs, e.g., 25, 50, or 100
  uint8_t bandwidth_dl = 50; // number of PRBs, e.g., 25, 50, or 100
  std::string filename = "stats.txt";

  // action parameters, from Python under slicing decision
  int mcs_offset_ul = 0; // number of PRBs, e.g., 25, 50, or 100
  int mcs_offset_dl = 0; // number of PRBs, e.g., 25, 50, or 100
  double backhaul_bw = 1000000000; // backhual bandwidth, bits/s
  double cpu_ratio = 1.0; // todo how cpu_ratio affects latency in edge computation

  // learnable parameters 
  double baseline_loss = 38.57; // baseline loss, as the distrance is fixed, so log attenuation model "becomes" baseline gain
  double enb_antenna_gain = 5.0; // antenna gain
  double enb_noise_figure = 5.0; // enb tx noise figure (gain loss by hardware) in dB
  double ue_antenna_gain = 5.0; // antenna gain
  double ue_noise_figure = 9.0; // ue tx noise figure (gain loss by hardware) in dB
  double backhaul_offset = 0; // backhual bandwidth, bits/s
  double backhaul_delay = 0; // backhual delay in milliseconds
  double edge_bw = 22300000000; // backhual bandwidth , bits/s
  double edge_delay = 0; // backhual delay in milliseconds
  double compute_time_mean = 81; //144, factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
  double compute_time_std = 35; //55, factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
  double compute_time_mean_offset = 0; 
  double compute_time_std_offset = 0; 
  double loading_time_offset = 0;

  //  fixed parameters, as the setting in real testbed 
  double distance = 1.0; // always one meter as the testbed is
  int mcs_max_ul = 20; // max mcs in uplink //////// XXXXXXXX the real system is max 20 mcs in UL
  int mcs_max_dl = 28; // max mcs in downlink
  double enb_tx_power = 30.0; // enb tx power in dB
  double ue_tx_power = 30.0; // ue tx power in dB
  uint32_t ul_avg_size = 28859; // average request size of uplink
  uint32_t ul_std_size = 9939; // std request size of uplink
  uint32_t dl_avg_size = 300; // minimum, average request size of downlink
  uint32_t dl_std_size = 1; // std request size of downlink

  // Command line arguments
  CommandLine cmd (__FILE__);
  cmd.AddValue ("numUEs", "Number of UE pairs", numUEs);
  cmd.AddValue ("simtime", "Total duration of the simulation", simtime);
  cmd.AddValue ("filename", "Name of saved file", filename);
  cmd.AddValue ("bandwidth_ul", "UL Bandwidth of eNB [number of PRBs, min 15, max 100]", bandwidth_ul);
  cmd.AddValue ("bandwidth_dl", "DL Bandwidth of eNB [number of PRBs, min 15, max 100]", bandwidth_dl);
  cmd.AddValue ("mcs_offset_ul", "Offset of MCS modulation, [min 0, max 20]", mcs_offset_ul);
  cmd.AddValue ("mcs_offset_dl", "Offset of MCS modulation, [min 0, max 20]", mcs_offset_dl);
  cmd.AddValue ("backhaul_bw", "Bandwidth between RAN and CN [bits/s]", backhaul_bw);
  cmd.AddValue ("cpu_ratio", "Allocated CPU ratio [0,1]", cpu_ratio);
  cmd.AddValue ("enb_antenna_gain", "eNB antenna gain [dB]", enb_antenna_gain);
  cmd.AddValue ("ue_antenna_gain", "UE antenna gain [dB]", ue_antenna_gain);
  cmd.AddValue ("baseline_loss", "Baseline loss in log attenuation model [dB]", baseline_loss);
  cmd.AddValue ("enb_tx_power", "eNB Tx power [dBm]", enb_tx_power);
  cmd.AddValue ("enb_noise_figure", "eNB noise figure [dB]", enb_noise_figure);
  cmd.AddValue ("ue_tx_power", "UE Tx power [dBm]", ue_tx_power);
  cmd.AddValue ("ue_noise_figure", "UE noise figure [dB]", ue_noise_figure);
  cmd.AddValue ("backhaul_offset", "Bandwidth between RAN and CN [bits/s]", backhaul_offset);
  cmd.AddValue ("backhaul_delay", "Delay between RAN and CN [ms]", backhaul_delay);
  cmd.AddValue ("edge_bw", "Bandwidth between CN and Edge [bits/s]", edge_bw);
  cmd.AddValue ("edge_delay", "Delay between CN and Edge [ms]", edge_delay);
  cmd.AddValue ("compute_time_mean_offset", "factor compute time in server [ms]", compute_time_mean_offset);
  cmd.AddValue ("compute_time_std_offset", "factor compute time in server [ms]", compute_time_std_offset);
  cmd.AddValue ("loading_time_offset", "exponential loading time offset in client [ms]", loading_time_offset);
  cmd.AddValue ("random_seed", "the seed", random_seed);
  cmd.Parse (argc, argv);

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults ();

  // parse again so you can override default values from the command line
  cmd.Parse(argc, argv);

  LogComponentEnableAll (LOG_PREFIX_TIME);
//   LogComponentEnableAll (LOG_PREFIX_FUNC);
//   LogComponentEnable ("EdgeHttpClient", LOG_INFO);
//   LogComponentEnable ("EdgeHttpServer", LOG_INFO);
  LogComponentEnable ("my-lena-simple-epc", LOG_INFO);

  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (4194304)); // this is important, large enough
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (6291456)); // this is important, large enough
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue ("DlRlcStats"+filename+".txt"));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue ("UlRlcStats"+filename+".txt"));

  RngSeedManager::SetSeed (random_seed);  
  RngSeedManager::SetRun (random_seed);

  MCS_OFFSET_UL = mcs_offset_ul; // assign the value to the global variable
  MCS_OFFSET_DL = mcs_offset_dl; // assign the value to the global variable
  MCS_MAX_UL = mcs_max_ul; // assign the value to the global variable
  MCS_MAX_DL = mcs_max_dl; // assign the value to the global variable
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// LteHelper and EpcHelper and Nodes ///////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  // Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  Ptr<NoBackhaulEpcHelper> epcHelper = CreateObject<NoBackhaulEpcHelper> (); //  customized backhaul links
  Ptr<Node> pgw = epcHelper->GetPgwNode ();
  Ptr<Node> sgw = epcHelper->GetSgwNode ();

  lteHelper->SetEpcHelper (epcHelper);

  // Create eNB and UEs 
  NodeContainer ueNodes;
  NodeContainer enbNodes;
  enbNodes.Create (1);
  ueNodes.Create (numUEs);

  // Create a single RemoteHost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);

  InternetStackHelper internet;
  internet.Install (remoteHostContainer);


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// Install Mobility Model for All ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  for (uint16_t i = 0; i < numUEs; i++)
    {
      positionAlloc->Add (Vector (distance, 0, 0)); // distance, i, 0)
    }
  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator(positionAlloc);
  mobility.Install(ueNodes);

  // Install Mobility Model for eNB
  Ptr<ListPositionAllocator> positionAlloc_eNB = CreateObject<ListPositionAllocator> ();
  positionAlloc_eNB->Add (Vector (0, 0, 0));
  MobilityHelper mobility_eNB;
  mobility_eNB.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility_eNB.SetPositionAllocator(positionAlloc_eNB);
  mobility_eNB.Install(enbNodes);

  // Install Mobility Model for SGW
  Ptr<ListPositionAllocator> positionAlloc2 = CreateObject<ListPositionAllocator> ();
  positionAlloc2->Add (Vector (0,  0, 10));
  MobilityHelper mobility2;
  mobility2.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility2.SetPositionAllocator (positionAlloc2);
  mobility2.Install (pgw);
  mobility2.Install (sgw);

  // Install Mobility Model for Edge server
  Ptr<ListPositionAllocator> positionAlloc3 = CreateObject<ListPositionAllocator> ();
  positionAlloc3->Add (Vector (0,  0, 20));
  MobilityHelper mobility3;
  mobility3.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility3.SetPositionAllocator (positionAlloc3);
  mobility3.Install (remoteHost);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////  Configurations ////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // TODO Assert using this scheduler, as the MCS modification is done in this method !!!!!!!!!!!!
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler"); // TODO Assert 
  // lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));

  lteHelper->SetSchedulerAttribute ("HarqEnabled",  BooleanValue (true));
  lteHelper->SetHandoverAlgorithmType ("ns3::NoOpHandoverAlgorithm"); // disable automatic handover
  lteHelper->SetAttribute ("FadingModel", StringValue ("")); //no fading model, ns3::TraceFadingLossModel, ns3::SpectrumPropagationLossModel

  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (bandwidth_dl));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (bandwidth_ul));

  lteHelper->SetEnbAntennaModelType ("ns3::IsotropicAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("Gain", DoubleValue (enb_antenna_gain));
  lteHelper->SetUeAntennaModelType ("ns3::IsotropicAntennaModel");
  lteHelper->SetUeAntennaModelAttribute ("Gain", DoubleValue (ue_antenna_gain));
  Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  Config::SetDefault ("ns3::LteAmc::Ber", DoubleValue (0.01));
  Config::SetDefault ("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue (320)); // used for large number of users,  Allowed values: 2 5 10 20 40 80 160 320

  lteHelper->SetPathlossModelType (TypeId::LookupByName ("ns3::LogDistancePropagationLossModel")); //  L = L_0 + 10 n log_{10}(\frac{d}{d_0})
  lteHelper->SetPathlossModelAttribute ("Exponent", DoubleValue (3.9)); // n
  lteHelper->SetPathlossModelAttribute ("ReferenceLoss", DoubleValue (baseline_loss)); //ref. loss in dB at 1m for 2.025GHz, L_0
  lteHelper->SetPathlossModelAttribute ("ReferenceDistance", DoubleValue (1)); // d_0

  // // set frequency. This is important because it changes the behavior of the path loss model
  lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (3000)); //  3000, 200
  lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (21000)); // 21000, 18200
  lteHelper->SetUeDeviceAttribute ("DlEarfcn", UintegerValue (3000)); // 3000, 200 to be revised


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////// Install Procotol Stacks ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Install LTE Devices to the nodes
  NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice (ueNodes);

  //////////////////////////////////////  Configurations ////////////////////////////////////////////////////////////
  
  
  Ptr<LteEnbNetDevice> plteEnbDev = enbLteDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbPhy> enbPhy = plteEnbDev->GetPhy ();
  enbPhy->SetAttribute ("TxPower", DoubleValue (enb_tx_power));
  enbPhy->SetAttribute ("NoiseFigure", DoubleValue (enb_noise_figure));
  enbPhy->SetAttribute ("MacToChannelDelay", UintegerValue (2));

  Ptr<LteUeNetDevice> plteUeDev = ueLteDevs.Get (0)->GetObject<LteUeNetDevice> ();
  Ptr<LteUePhy> uePhy = plteUeDev->GetPhy ();
  uePhy->SetAttribute ("TxPower", DoubleValue (ue_tx_power));
  uePhy->SetAttribute ("NoiseFigure", DoubleValue (ue_noise_figure));
  uePhy->SetAttribute ("EnableUplinkPowerControl", BooleanValue (true));

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Install the IP stack on the UEs
  internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueLteDevs));

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// customized backhual between eNB and Core //////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Ipv4AddressHelper s1uIpv4AddressHelper;

  // Create networks of the S1 interfaces
  s1uIpv4AddressHelper.SetBase ("10.0.0.0", "255.255.255.252");

  Ptr<Node> enb = enbNodes.Get (0);
  Ptr<LteEnbNetDevice> enb1LteDev = enbLteDevs.Get(0)->GetObject<LteEnbNetDevice> ();
  std::vector<uint16_t> cellIds = enb1LteDev->GetCellIds ();
  // std::vector<uint16_t> cellIds (enb1CellIds, 0);

  // Create a point to point link between the eNB and the SGW with
  // the corresponding new NetDevices on each side
  PointToPointHelper p2pb;
  p2pb.SetDeviceAttribute ("DataRate", DataRateValue (backhaul_bw + backhaul_offset));
  p2pb.SetChannelAttribute ("Delay", TimeValue (MilliSeconds(backhaul_delay)));
  p2pb.SetDeviceAttribute ("Mtu", UintegerValue (1500));

  NetDeviceContainer sgwEnbDevices = p2pb.Install (sgw, enb);

  Ipv4InterfaceContainer sgwEnbIpIfaces = s1uIpv4AddressHelper.Assign (sgwEnbDevices);
  s1uIpv4AddressHelper.NewNetwork ();

  Ipv4Address sgwS1uAddress = sgwEnbIpIfaces.GetAddress (0);
  Ipv4Address enbS1uAddress = sgwEnbIpIfaces.GetAddress (1);

  // Create S1 interface between the SGW and the eNB
  epcHelper->AddS1Interface (enb, enbS1uAddress, sgwS1uAddress, cellIds);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// customized link between Core and Edge /////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Create the Internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (edge_bw));
  p2ph.SetChannelAttribute ("Delay", TimeValue (MilliSeconds(edge_delay)));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));

  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  // interface 0 is localhost, 1 is the p2p device
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////  UE Attachment /////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Assign IP address to UEs, and install applications
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<Node> ueNode = ueNodes.Get (u);
      // Set the default gateway for the UE
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // Attach one UE per eNodeB
  for (uint16_t i = 0; i < numUEs; i++)
    {
      lteHelper->Attach (ueLteDevs.Get(i), enbLteDevs.Get(0));
      // side effect: the default EPS bearer will be activated
    }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////  Animation ////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // for (uint16_t i = 0; i < numUEs; i++) //  set animation location
  //   {
  //     AnimationInterface::SetConstantPosition (ueNodes.Get (i), distance, i); 
  //   }
  // AnimationInterface::SetConstantPosition (enbNodes.Get (0), 0, 0);
  // AnimationInterface::SetConstantPosition (pgw, 0, 10);
  // AnimationInterface::SetConstantPosition (remoteHostContainer.Get (0), 0, 20);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////// HTTP Application ////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Create HTTP server helper
  EdgeHttpServerHelper serverHelper (remoteHostAddr);

  // Install HTTP server
  ApplicationContainer serverApps = serverHelper.Install (remoteHost);
  Ptr<EdgeHttpServer> httpServer = serverApps.Get (0)->GetObject<EdgeHttpServer> ();

  // Example of connecting to the trace sources
  httpServer->TraceConnectWithoutContext ("ConnectionEstablished",
                                          MakeCallback (&ServerConnectionEstablished));
  httpServer->TraceConnectWithoutContext ("MainObject", MakeCallback (&MainObjectGenerated));
  httpServer->TraceConnectWithoutContext ("Tx", MakeCallback (&ServerTx));
  httpServer->TraceConnectWithoutContext ("RxRtt", MakeCallback (&ServerRxRtt));
  httpServer->TraceConnectWithoutContext ("Queue", MakeCallback (&ServerQueue));
  httpServer->TraceConnectWithoutContext ("Compute", MakeCallback (&ServerCompute));

  // Setup HTTP variables for the server
  PointerValue varPtr_server;
  httpServer->GetAttribute ("Variables", varPtr_server);
  Ptr<EdgeHttpVariables> httpVariables_server = varPtr_server.Get<EdgeHttpVariables> ();
  httpVariables_server->SetMainObjectSizeMean (dl_avg_size); 
  httpVariables_server->SetMainObjectSizeStdDev (dl_std_size);
  // httpVariables_server->SetComputeTimeMean (MilliSeconds((compute_time+compute_time_offset)/cpu_ratio)); // to do, whether if this model appropriate 
  httpVariables_server->SetComputeTimeMean ((compute_time_mean+compute_time_mean_offset)/cpu_ratio); 
  httpVariables_server->SetComputeTimeStdDev ((compute_time_std+compute_time_std_offset)/cpu_ratio);
  
  // Create HTTP client helper
  EdgeHttpClientHelper clientHelper (remoteHostAddr);

  // Install HTTP client
  ApplicationContainer clientApps = clientHelper.Install (ueNodes);

  for (uint16_t i = 0; i < numUEs; i++)
    {
      Ptr<EdgeHttpClient> httpClient = clientApps.Get (i)->GetObject<EdgeHttpClient> ();

        // Setup HTTP variables for the server
      PointerValue varPtr_client;
      httpClient->GetAttribute ("Variables", varPtr_client);
      Ptr<EdgeHttpVariables> httpVariables_client = varPtr_client.Get<EdgeHttpVariables> ();
      httpVariables_client->SetRequestSizeMean (ul_avg_size);
      httpVariables_client->SetRequestSizeStdDev (ul_std_size); 
      httpVariables_client->SetParsingTimeMean (MilliSeconds(loading_time_offset)); 

      // Example of connecting to the trace sources 
      httpClient->TraceConnectWithoutContext ("TxRequestTrace", MakeCallback (&RequestGenerated));
      httpClient->TraceConnectWithoutContext ("RxMainObject", MakeCallback (&ClientMainObjectReceived));
      // httpClient->TraceConnectWithoutContext ("Rx", MakeCallback (&ClientRx));
      httpClient->TraceConnectWithoutContext ("RxDelay", MakeCallback (&ClientRxDelay));
      httpClient->TraceConnectWithoutContext ("RxRtt", MakeCallback (&ClientRxRtt));
      clientApps.Get (i)->SetStartTime (MilliSeconds(i)); // prefer not start all the applications at the same time
    }

  clientApps.Stop (Seconds(simtime));
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // lteHelper->EnableTraces (); // enable this for collecting PER stats
  lteHelper->EnableRlcTraces ();
  // Uncomment to enable PCAP tracing
  // p2ph.EnablePcapAll("my-lena-simple-epc");

  Simulator::Stop (Seconds(simtime));

  // AnimationInterface anim ("my-lena-simple-epc.xml"); // Mandatory // animation

  Simulator::Run ();

  /*GtkConfigStore config;
  config.ConfigureAttributes();*/

  Simulator::Destroy ();

  WritetoFile(filename); // write the stats to files

  return 0;
}

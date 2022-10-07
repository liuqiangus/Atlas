# Atlas
Atlas: Automate Online Service Configuration in Network Slicing



# Network Simulator and Prototype Build

Network Simulator.


Install NS-3 3.36, either with official instruction or provided bash file.

Add additional files (i.e., $edge/$) in NS-3 $contrib/$ folder, and rebuild.

Add additional files (i.e., main.cc) in NS-3 scratch/ folder, and rebuild.

Validate the network simulator can be connected by $simulator.py$ script.


Network Prototype. (The detailed installation are in https://github.com/int-unl/End-to-End-Slicing.git)

Install OpenAirInterface RAN with official instruction.

Install OpenAirInterface CORE with official instruction, where the dockerized network functions are required, e.g., SPGW-U and HSS.

Modify SPGW-C and rebuild to enable core network slicing, validate it redirects specific mobile users to corresponded SPGW-Us.

Connect smartphones with RAN with programmed USIM to match the PLMN and other parameters, validate they can access to the SPGW-U docker.

Install the provided Android application to smartphones (Android 11).

Install the provided edge server application in individual SPGW-U dockers, validate smartphones can connect with their servers with periodically performance updates.

Install FlexRAN controller 2.0 with official instruction, validate its slicing capability when changing PRB allocation on RAN to different mobile users.

Initialize the SDN switch and install OpenDayLight to connect the switch, validate the provided $tn_server.py$ can connect OpenDayLight.

Connect SDN switch between the RAN and CORE desktop, validate it can enforce bandwidth allocation to different SPGW-U dockers.

As everything is well validated individually, close all scripts and program.

The order of running the network prototype is: OpenDayLight -- Transport controller -- FlexRAN controller -- CORE -- edge server applications -- RAN -- Configure slicing parameter -- Smartphone (disable airplane mode and open the app) -- start $main_online.py$.

The demo procedures of bringing up the network prototype is online: 




# Experiment workflow


Download the open-source codes from 
Install package dependencies, e.g., sklearn, scipy, torch.


Stage 1: learning-based simulator.

Run $main\_simulator.py$, where the arguments vary according to different experiments.

Run $plot\_simulator.py$ to reproduce the figures in the paper, after all corresponded experiments are done.

Update the searched optimal simulation parameter to $parameter.py$.


Stage 2: offline training.

Run $main\_offline.py$, where the arguments vary according to different experiments.

Run $plot\_offline.py$ to reproduce the figures in the paper, after all corresponded experiments are done.

Update the searched optimal resource configuration to $parameter.py$.


Stage 3: online learning.

Bring up the network prototype (see below), validate it is live and can be connected by the $system.py$ script.

Run $main\_online.py$, where the arguments vary according to different experiments.

Run $plot\_online.py$ to reproduce the figures in the paper, after all corresponded experiments are done.# 

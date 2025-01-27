### Using the network_manager module for simulating network using TC

#### Server Config
Use the docker-compose for hosting the fileserver and the running the client

OR 

1. Run this command in your server
    ```
    # Listen for network change requests
    nc -kluv 4444 | /bin/sh &
    ```
    This will listen on UDP port 4444 and execute those as shell commands. The player will send tc network change commands on that port

2. Configure the server address. Add this to /etc/hosts (On mac & Linux)
    ```
    <SERVER_IP_ADDRESS>        server
    ```
    <SERVER_IP_ADDRESS> could be localhost if the serve is on the host or it can be an external IP address (if the server is running on another machine) or a docker container IP address (edited)


#### Running the client

1. Add the network_manager module by adding the option --mod_analyzer network_manager (edited) 
2. You can specify the BW profile by adding the option --bw_profile "4000 35 0 1\n250 35 0 4"


#### BW_PROFILE format
`bw_profile` is given using the format : 
```
<BW_IN_Kbps> <LATENCY> <PACKET_DROP_IN_PERCENTAGE> <SUSTAIN_TIME>
```
If you skip the <SUSTAIN_TIME> it will be infinity
Each new network config is separated by a new line.

So "4000 35 0 1\n250 35 0 4" would mean,
1. Sustain 4000 Kbps, 35ms latency and no packet drop for 1s
2. Sustain 250 Kbps, 35ms latency and no packet drop for 4s
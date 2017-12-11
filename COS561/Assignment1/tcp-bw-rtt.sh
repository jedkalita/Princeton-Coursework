#! /bin/sh

#server bw and rtt in kazakhstan
./iperf3 -c iperf.it-north.net -p 5204 -t 30 > kazakh-bw.txt
ping iperf.it-north.net -p 5204 -t 30 > kazakh-rtt.txt

#server bw and rtt in france
./iperf3 -c bouygues.testdebit.info -p 5204 -t 30 > france-bw.txt
ping bouygues.testdebit.info -p 5204 -t 30 > france-rtt.txt

#server bw and rtt in netherlands
./iperf3 -c speedtest.serverius.net -p 5002 -t 30 > ned-bw.txt
ping speedtest.serverius.net -p 5002 -t 30 > ned-ping.txt



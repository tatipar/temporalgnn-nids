The following was extracted from: https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html

Cite: 

```text
@inproceedings{liu2022error,
title={Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
author={Liu, Lisa and Engelen, Gints and Lynar, Timothy and Essam, Daryl and Joosen, Wouter},
booktitle={2022 IEEE Conference on Communications and Network Security (CNS)},
pages={254--262},
year={2022},
organization={IEEE}
}
```

# 11. Infiltration

According to the authors of the The Infiltration attack consists of 3 components:
- Dropbox Download: The authors did not provide additional information about this attack, but we assume that some kind of malicious file was downloaded by the victim through Dropbox.
- NMAP Portscan: The infected victim executes portscans on the inside network.
- Communication victim - attacker: Reports of the portscan are sent to the attacker by the victim.

This attack saw significant label corruption on both days, to the point that it was largely impossible to reverse-engineer the original labelling logic. We based our investigation on the IPs of the victim and attacker in order to devise a new labelling logic. Given the 3 very different components of this attack, we decided to split the label in 3, labelling each flow in accordance with the component of the Infiltration attack that it belongs to. Future researchers making use of this dataset are free to decide whether they want to keep the labelling this way, or merge all 3 Infiltration components under a single label.

## 11.1 Dropbox Download

### 28-02-2018

We found traffic showing the victim host communicating with IP addresses belonging to Dropbox: [162.125.3.1, 162.125.3.5, 162.125.3.6, 162.125.248.1, 162.125.18.133]. Whatever is being downloaded is locked behind TLS, so we cannot verify the downloaded content, but assume this to be a malicious file. We note that in the original dataset, on Wednesday 28-02-2018 all flows that contain one of the 5 Dropbox IPs listed above (either as Src or Dst) are all labelled benign.

If we have to speculate, our best guess is that the victim is downloading the malicious files through a link that has been shared with them on 162.125.3.6 (rather than getting the file through say a local dropbox sync) - the hostname for this ip is dl-web.dropbox.com. This occurs at 14:43 and 17:43.

We found a few more "Dropbox" servers the victim is communicating with, which start with 52.xx.xxx.xxx and 104.xx.xxx.xxx. Communication with these IPs likely contains auxiliary data that gets loaded when opening Dropbox on the browser, and as such we labelled traffic going to these IPs as "Attempted".

Note that we found 2 separate rounds of "Dropbox Download" taking place. This is also reflected in our labelling logic.

Labelling logic:

```text
Labelled as Infiltration - Dropbox Download

Src IP == 172.31.69.24 &&
Dst IP in [162.125.3.1, 162.125.3.5, 162.125.3.6, 162.125.248.1, 162.125.18.133]

Attack time window is either:
Time Start UTC (First Packet): 28-02-2018 14:33:24 (unix: 1519828404) &&
Time End UTC (Last Packet): 28-02-2018 14:46:12 (unix: 1519829172)
Or:
Time Start UTC (First Packet): 2018-02-28 17:42:51 (unix: 1519839771) &&
Time End UTC (Last Packet): 2018-02-28 17:43:44 (unix: 1519839824)
```

```text
Labelled as Infiltration - Dropbox Download - Attempted (Category 4 - Attack Artefact)

Src IP == 172.31.69.24 &&
Dst IP in [104.16.100.29, 104.16.99.29, 52.84.128.3, 52.85.101.236, 52.85.131.81, 52.85.95.206]

Attack time window is either:
Time Start UTC (First Packet): 28-02-2018 14:33:24 (unix: 1519828404) &&
Time End UTC (Last Packet): 28-02-2018 14:46:12 (unix: 1519829172)
Or:
Time Start UTC (First Packet): 2018-02-28 17:42:51 (unix: 1519839771) &&
Time End UTC (Last Packet): 2018-02-28 17:43:44 (unix: 1519839824)
```

```text
Labelled as Infiltration - Dropbox Download - Attempted (Category 0 - No payload sent by attacker)

Src IP == 172.31.69.24 &&
Dst IP in [162.125.3.1, 162.125.3.5, 162.125.3.6, 162.125.248.1, 162.125.18.133] &&
Total Length of Fwd Packets == 0 &&

Attack time window is either:
Time Start UTC (First Packet): 28-02-2018 14:33:24 (unix: 1519828404) &&
Time End UTC (Last Packet): 28-02-2018 14:46:12 (unix: 1519829172)
Or:
Time Start UTC (First Packet): 2018-02-28 17:42:51 (unix: 1519839771) &&
Time End UTC (Last Packet): 2018-02-28 17:43:44 (unix: 1519839824)
```

### 01-03-2018

Concerning the original labelling logic for this day, the only consistent trend we could find is that the Infiltration label was not used outside of the strict time windows listed on the dataset website.

Other than that the traffic seems very similar to that of Wednesday 28-02-2018. For this day, there only seem to be 4 Dropbox IP's that the victim is communicating with: [162.125.3.1, 162.125.3.6, 162.125.248.1, 162.125.18.133]

It looks like content is again downloaded from a dropbox link. dl-web.dropbox.com appears to have been accessed just once on this date.

Just like the previous day, we found a few more dropbox servers where we believe communication with these IPs likely contains auxiliary data that gets loaded when opening Dropbox on the browser. Traffic sent to these IPs is again labelled as "Attempted".

Note that on this day too we found 2 separate rounds of "Dropbox Download" taking place. This is again reflected in our labelling logic.

Labelling logic:

```text
Labelled as Infiltration - Dropbox Download

Src IP == 172.31.69.13 &&
Dst IP in [162.125.3.1, 162.125.3.6, 162.125.248.1, 162.125.18.133]

Attack time window is either:
Time Start UTC (First Packet): 01-03-2018 13:53:10 (unix: 1519912390) &&
Time End UTC (Last Packet): 01-03-2018 13:59:20 (unix: 1519912760)
Or:
Time Start UTC (First Packet): 01-03-2018 14:03:52 (unix: 1519913032) &&
Time End UTC (Last Packet): 01-03-2018 15:34:14 (unix: 1519918454)
```

```text
Labelled as Infiltration - Dropbox Download - Attempted (Category 4 - Attack Artefact)

Src IP == 172.31.69.13 &&
Dst IP in [104.16.100.29, 13.32.168.125, 52.85.112.72]

Attack time window is either:
Time Start UTC (First Packet): 01-03-2018 13:53:10 (unix: 1519912390) &&
Time End UTC (Last Packet): 01-03-2018 13:59:20 (unix: 1519912760)
Or:
Time Start UTC (First Packet): 01-03-2018 14:03:52 (unix: 1519913032) &&
Time End UTC (Last Packet): 01-03-2018 15:34:14 (unix: 1519918454)
```

```text
Labelled as Infiltration - Dropbox Download - Attempted (Category 0 - No payload sent by attacker)

Src IP == 172.31.69.13 &&
Dst IP in [162.125.3.1, 162.125.3.6, 162.125.248.1, 162.125.18.133]
Total Length of Fwd Packets == 0 &&

Attack time window is either:
Time Start UTC (First Packet): 01-03-2018 13:53:10 (unix: 1519912390) &&
Time End UTC (Last Packet): 01-03-2018 13:59:20 (unix: 1519912760)
Or:
Time Start UTC (First Packet): 01-03-2018 14:03:52 (unix: 1519913032) &&
Time End UTC (Last Packet): 01-03-2018 15:34:14 (unix: 1519918454)
```

## 11.2 Communication Victim - Attacker

### 28-02-2018

Note that for Wednesday-28-02-2018, capEC2AMAZ-O4EL3NG-172.31.69.24-part2 is the only PCAP file that contains this type of Infiltration traffic (i.e. Communication Victim Attacker).

We also note that there are quite a lot of flows labelled as "Infiltration" between 8.6.0.1 and 8.6.0.4, which are actually ARP packets that have been erroneously processed by the CICFlowMeter tool (an ARP packet does not contain an IP header). Our labelling logic labels all these flows as Benign.

Upon crosschecking the IP addresses for Infiltration listed on the website, the only flows we found are those with Src IP == 172.31.69.24 and Dst IP == 13.58.225.34. This means that Infiltration flows only go from victim to attacker (this is the same in CICIDS 2017). However, in the originally released dataset, all flows going between the 2 IP's described above are labelled as Benign. In total there are 44 flows, and almost all of them last longer than a minute, which is again similar to Infiltration in CICIDS 2017. After extensively looking through the PCAP, we confirmed that traffic between these two IP's is indeed infiltration traffic, where the infected victim sends an NMAP report to the attacker. Based on this information we also concluded that the infected victim is the one who executes the portscan, and not the attacker.

Here we again have two separate rounds of malicious traffic.

Labelling logic:

```text
Labelled as Infiltration - Communication Victim Attacker

Src IP == 172.31.69.24 &&
Dst IP == 13.58.225.34

Attack time window is either:
Time Start UTC (First Packet): 28-02-2018 14:45:40 (unix: 1519829140) &&
Time End UTC (Last Packet): 28-02-2018 16:08:55 (unix: 1519834135)
Or:
Time Start UTC (First Packet): 28-02-2018 17:43:59 (unix: 1519839839) &&
Time End UTC (Last Packet): 28-02-2018 18:40:00 (unix: 1519843200)
```

```text
Labelled as Infiltration - Communication Victim Attacker - Attempted (Category 0 - No payload sent by attacker)

Src IP == 172.31.69.24 &&
Dst IP == 13.58.225.34
Total Length of Fwd Packets == 0 &&

Attack time window is either:
Time Start UTC (First Packet): 28-01-2018 14:45:40 (unix: 1519829140) &&
Time End UTC (Last Packet): 28-02-2018 16:08:55 (unix: 1519834135)
Or:
Time Start UTC (First Packet): 28-02-2018 17:43:59 (unix: 1519839839) &&
Time End UTC (Last Packet): 28-02-2018 18:39:59 (unix: 1519843200)
```

### 01-03-2018

We again were able to track down flows sent between the victim and attacker IP as indicated on the dataset website. However, the nature of the traffic on this day is different from the previous day. When stitched together, the transferred data was mostly illegible, with speckles of clear text that make it seem like console output is being sent. Despite the flows going from victim to attacker, we noted that all large packets are going in the direction of the victim machine, i.e. 172.31.69.24.

For this day we identified 3 separate rounds of traffic, again reflected in our labelling logic.

Labelling logic:

```text
Labelled as Infiltration - Communication Victim Attacker

Src IP == 172.31.69.13 &&
Dst IP == 13.58.225.34

Attack time window is either:
Time Start UTC (First Packet): 01-03-2018 13:57:54 (unix: 1519912674) &&
Time End UTC (Last Packet): 01-03-2018 13:59:05 (unix: 1519912745)
Or:
Time Start UTC (First Packet): 01-03-2018 14:04:35 (unix: 1519913075) &&
Time End UTC (Last Packet): 01-03-2018 18:17:25 (unix: 1519928245)
Or:
Time Start UTC (First Packet): 01-03-2018 18:18:15 (unix: 1519928295) &&
Time End UTC (Last Packet): 01-03-2018 19:37:21 (unix: 1519933041)
```

```text
Labelled as Infiltration - Communication Victim Attacker - Attempted (Category 0 - No payload sent by attacker)

Src IP == 172.31.69.13 &&
Dst IP == 13.58.225.34
Total Length of Fwd Packets == 0 &&

Attack time window is either:
Time Start UTC (First Packet): 01-03-2018 13:57:54 (unix: 1519912674) &&
Time End UTC (Last Packet): 01-03-2018 13:59:05 (unix: 1519912745)
Or:
Time Start UTC (First Packet): 01-03-2018 14:04:35 (unix: 1519913075) &&
Time End UTC (Last Packet): 01-03-2018 18:17:25 (unix: 1519928245)
Or:
Time Start UTC (First Packet): 01-03-2018 18:18:15 (unix: 1519928295) &&
Time End UTC (Last Packet): 01-03-2018 19:37:21 (unix: 1519933041)
```

## 11.3 NMAP Portscan

### 28-02-2018

Our analysis determined that the infected victim performs portscans against 21 other hosts. An easy way to verify this is by filtering on certain ports we know for sure will only be present in a port scan. For example, when we filtered by port 32776 or some other port that isn't reserved for a common application, and aggregate by DST IPs we got 21 entries.

We also found other traffic - different from portscan traffic - which consistently occurred in flows with all victims. This traffic was basically all NBNS/ICMP/DHCP traffic. 4 UDP messages always follow the ICMP traffic. After manual inspection we concluded that, of the above-mentioned traffic types, only DHCP is background traffic, and so the rest can be considered malicious. We filter out the DHCP traffic by filtering out flows with Src port 68.

Labelling logic:

```text
Labelled as Infiltration - NMAP Portscan

Src IP == 172.31.69.24 &&
Dst IP in [172.31.69.1, 172.31.69.10, 172.31.69.11, 172.31.69.12, 172.31.69.13, 172.31.69.14, 172.31.69.16, 172.31.69.17, 172.31.69.19, 172.31.69.20, 172.31.69.23, 172.31.69.4, 172.31.69.5, 172.31.69.6, 172.31.69.8, 172.31.69.9, 172.31.69.7, 172.31.69.22, 172.31.69.15, 172.31.69.21, 172.31.69.18] &&
Src Port not [68] &&
Time Start UTC (First Packet): 28-02-2018 14:46:22 (unix: 1519829182) &&
Time End UTC (Last Packet): 28-02-2018 18:39:00.746247 (unix: 1519843140.746247)
```

### 01-03-2018

For this day, the procedure to establish the portscan traffic was analogous to the previous day. Here too we filter out DHCP traffic happening on Src Port 68.

Labelling logic:

```text
Labelled as Infiltration - NMAP Portscan

Src IP == 172.31.69.13 &&
Dst IP in [172.31.69.1, 172.31.69.11, 172.31.69.12, 172.31.69.16, 172.31.69.8, 172.31.69.9, 172.31.69.10, 172.31.69.14, 172.31.69.4, 172.31.69.5, 172.31.69.6, 172.31.69.17, 172.31.69.20, 172.31.69.23, 172.31.69.24, 172.31.69.19, 172.31.69.7, 172.31.69.15, 172.31.69.18, 172.31.69.22, 172.31.69.21] &&
Src Port not [68] &&
Time Start UTC (First Packet): 01-03-2018 14:09:48.354333 (unix: 1519913388.354333) &&
Time End UTC (Last Packet): 01-03-2018 19:38:12.182726 (unix: 1519933092.182726)
```


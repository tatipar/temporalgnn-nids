"""List members of a remote CSE-CIC-IDS2018 zip (pcap.zip or logs.zip) without downloading it."""
import sys, urllib.parse, zipfile
from pcap_fetch_lib import RangeFile
day, name = sys.argv[1], sys.argv[2]
url = "https://cse-cic-ids2018.s3.amazonaws.com/" + urllib.parse.quote(f"Original Network Traffic and Log data/{day}/{name}")
for i in zipfile.ZipFile(RangeFile(url)).infolist():
    print(f"{i.file_size}\t{i.filename}")

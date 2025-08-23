#!/bin/bash
#

for bdf in `amd-smi list|grep BDF|awk '{print $2}'`; do
	echo $bdf
	sudo lspci -s 0000:08:00.0 -vvv|egrep "LnkCap:|LnkSta:|MaxPayload|MaxReadReq"
	echo
done

#!/bin/bash
#

# sudo niccli --list devices
list_bcm57508="0 1 2 3 4 5 6 7"

for index in $list_bcm57508; do
	echo Checking device $index
        sudo niccli --dev $index nvm --getoption support_rdma --scope 0
        sudo niccli --dev $index nvm --getoption performance_profile
        sudo niccli --dev $index nvm --getoption pcie_relaxed_ordering
        sudo niccli -dev $index get_qos
        sudo niccli -dev $index dump pri2cos
        sudo niccli -dev $index get_dscp2prio 
        echo
done


# To enable RDMA
# sudo niccli --dev <index|pci b:d:f> nvm --setoption support_rdma --scope <pf number> --value 1
# To enable Performance Profile
# sudo niccli --dev <index|pci b:d:f> nvm --setoption performance_profile --value 1
# To enable PCIe Relaxed Ordering
# sudo niccli --dev <index|pci b:d:f> nvm --setoption pcie_relaxed_ordering --value 1

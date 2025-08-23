# Environment check before any benchmarks

## References
- [Single-node network configuration for AMD Instinct accelerators](https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/develop/how-to/single-node-config.html)
- [Multi-node network configuration for AMD Instinct accelerators](https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/multi-node-config.html)
- [AMD Instinct MI300X workload optimization](https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/workload.html)
- [(GitHub) ROCm/cluster-networking](https://github.com/ROCm/cluster-networking)
- [RoCE cluster network configuration guide for AMD Instinct accelerators](https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/roce-network-config.html)
- [BCM957608 Ethernet Networking Guide for AMD Instinct MI300X GPU Clusters](https://docs.broadcom.com/doc/957608-AN2XX)

## Check
1. NUMA auto-balancing
```/proc/sys/kernel/numa_balancing``` should be 0.
Execute ```numa_balancing.sh```

2. Disable ACS
Execute ```dis_acs.sh```. Confirm by ```sudo lspci -vvv |grep ACSCtl```.

3. Check PCIe link speed and payload size
Execute ```pcie-check_gpu.sh```.

4. [Broadcom NIC] Check RDMA, PCIe and QoS
Execute ```broadcom-check.sh```

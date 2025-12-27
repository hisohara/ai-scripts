#!/usr/bin/bash

set -x

server="gpu-6" # change this
client="gpu-12" # change this

path_to_perftest=<dir to your perftest>
rdmadev=(bnxt_re0 bnxt_re1 bnxt_re2 bnxt_re3 bnxt_re4 bnxt_re5 bnxt_re7 bnxt_re8) # !!! adjust this according to your rdma (openfabrics) NIC device name
rocm_dev=(3 0 2 1 7 4 6 5) # adjust this to reflect what GPU ROCm ID aligns with the rdma (openfabrics) NIC device

num_dev=${#rdmadev[@]}

# bandwidth tests unidirectional
for benchmark in ib_read_bw ib_write_bw ib_send_bw; do
    for i in $(seq 0 $((num_dev - 1))); do
        printf "deviceinfo -- rdmadev: %s,\tlocal_ipaddr: %s,\trocm_dev: %s\n" ${rdmadev[i]} ${server} ${rocm_dev[i]}

        # H2H bandwidth
        killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw
        ${path_to_perftest}/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits -F -a &
        ssh $USER@${client} "killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw"
        ssh $USER@${client} "${path_to_perftest}/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits -F -a ${server}" 2>&1 | tee ${benchmark}_h2h_${rdmadev[i]}_unidi.log

        # D2D bandwidth
        killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw
        ${path_to_perftest}/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits --use_rocm=${rocm_dev[i]} -F -a &
        ssh $USER@${client} "killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw"
        ssh $USER@${client} "${path_to_perftest}/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits --use_rocm=${rocm_dev[i]} -F -a ${server}" 2>&1 | tee ${benchmark}_d2d_${rdmadev[i]}_rocm${rocm_dev[i]}_unidi.log

    done
done

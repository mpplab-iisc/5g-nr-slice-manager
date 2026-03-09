# 5g-nr-slice-manager

## Create sample config files

BW: 20 MHz, Time Horizon: 3 ms, Number of UEs: 3,
slices per UE: 2, slice_SLA (Mbps,ms): (9,10),(0.2,1),
MCS: 16

```
python milp-5g-nr.py --sample-config=configs/cfg-bw-20M-time-ms-3-mcs-16-ue-3-embb-thr-9-embb-lat-10-urllc-thr-0.2-urllc-lat-1.json --BW=20 --time-horizon-ms=3.0 --mcs=16 --ue-count=3 --embb-mbps=9 --embb-latency-ms=10 --urllc-mbps=0.2 --urllc-latency-ms=1
```

## Create MILPs

```
python milp-5g-nr.py --config=configs/cfg-bw-20M-time-ms-3-mcs-16-ue-3-embb-thr-9-embb-lat-10-urllc-thr-0.2-urllc-lat-1.json --output=milp/
```


#!/usr/bin/env bash

declare -A actuators=(
    [11]="Lsp"
    [12]="Lsr"
    [13]="Lsy"
    [14]="Lep"
    [15]="Lwr"
    [21]="Rsp"
    [22]="Rsr"
    [23]="Rsy"
    [24]="Rep"
    [25]="Rwr"
    [31]="Lhp"
    [32]="Lhr"
    [33]="Lhy"
    [34]="Lkp"
    [35]="Lap"
    [41]="Rhp"
    [42]="Rhr"
    [43]="Rhy"
    [44]="Rkp"
    [45]="Rap"
)

echo "Setting up CAN interfaces..."

ifaces="$(ip link show | grep -oP 'can[0-9]+' | sort -u)"

for interface in $ifaces; do
    echo "Bringing up $interface..."
    sudo ip link set $interface down
    sudo ip link set $interface type can bitrate 1000000
    sudo ip link set $interface txqueuelen 1000
    sudo ip link set $interface up
done

# TODO Discover actuators
# read -p "Press enter to discover actuators..."

selected="$(for i in {1..4}; do for j in {1..5}; do echo $i$j ${actuators[$i$j]}; done; done | fzf --reverse --height 20 | awk '{print $1}')"

read -p "Selected actuator: $selected. Press enter to request feedback..."

id=$(printf "%.2X" "$selected")

FOURPI=12.5663706144

for interface in $ifaces; do
    resp="true"
    while [ "$resp" ]; do
        resp="$(candump  -T 20 $interface,0200${id}FD:00FFFFFF & sleep .01;
               cansend $interface 0200FD${id}#00.00.00.00.00.00.00.00)"
        posbytes=$(echo "$resp" | awk '{print $4 $5}')
        posfloat=$(echo "ibase=16; $posbytes/FFFF" | bc -l)
        posrad=$(echo "-$FOURPI + $posfloat*$FOURPI * 2" | bc -l)
        echo $posrad
    done
    echo "Did not recieve response."
done

#!/usr/bin/env bash

selected="$*"

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

if [ -z "$selected" ]; then
    selected="$(for i in {1..4}; do for j in {1..5}; do echo $i$j ${actuators[$i$j]}; done; done | fzf -m --reverse --height 20 | awk '{print $1}')"
fi

echo "Selected actuators: $(echo $selected | tr '\n' ' ')"
read -p "Press enter to zero..."

for id_dec in $selected; do
    id=$(printf "%.2X" "$id_dec")
    resp=""

    for interface in $ifaces; do
        resp="$(candump  -T 200 $interface,0200${id}FD:00FFFFFF & sleep .01;
                cansend $interface 0600FD${id}#01.01.CD.00.00.00.00.00)"
        if [ "$resp" ]; then
            echo "Actuator $id_dec (${actuators[$id_dec]}) set to zero: $resp"
            break
        fi
    done

    if [ -z "$resp" ]; then
        echo "No response from actuator $id_dec (${actuators[$id_dec]})"
    fi
done

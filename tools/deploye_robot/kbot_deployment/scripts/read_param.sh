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

declare -A params=(
    #[0x7005]="run_mode"       # uint8    | 1 | W/R | 0: operation mode, 1: position mode (PP), 2: velocity mode, 3: operation mode current mode, 5: position mode (CSP)
    [0x7006]="iq_ref | float | -90 to 90 A"
    [0x700A]="spd_ref | float | -20 to 20 rad/s"        
    [0x700B]="limit_torque | float | 0 to 120 Nm"   
    [0x7010]="cur_kp | float | Default: 0.17"         
    [0x7011]="cur_ki | float | Default: 0.012"         
    [0x7014]="cur_filt_gain | float | 0 to 1.0, Default: 0.1"  
    [0x7016]="loc_ref | float | Unit: rad"        
    [0x7017]="limit_spd | float | 0 to 20 rad/s"      
    [0x7018]="limit_cur | float | 0 to 90 A"      
    [0x7019]="mechPos | float | Mechanical angle of the loading coil (rad)"        
    [0x701A]="iqf | float | -90 to 90 A"            
    [0x701B]="mechVel | float | -15 to 15 rad/s"        
    [0x701C]="VBUS | float | Unit: V"           
    [0x701E]="loc_kp | float | Default: 60"         
    [0x701F]="spd_kp | float | Default: 6"         
    [0x7020]="spd_ki | float | Default: 0.02"         
    [0x7021]="spd_filt_gain | float | Default: 0.1"  
)


#param="$({ echo "all"; for key in "${!params[@]}"; do value="${params[$key]}"; echo "$value | $key"; done } | fzf --reverse --height 20)"
params="$(for key in "${!params[@]}"; do value="${params[$key]}"; echo "$value | $key"; done  | fzf -m --reverse --height 20)"

if [ -z "$params" ]; then
    echo "No parameter selected. Exiting."
    exit 1
fi


echo "Setting up CAN interfaces..."

ifaces="$(ip link show | grep -oP 'can[0-9]+' | sort -u)"

for interface in $ifaces; do
    echo "Bringing up $interface..."
    sudo ip link set $interface down 2>/dev/null
    sudo ip link set $interface type can bitrate 1000000 2>/dev/null
    sudo ip link set $interface txqueuelen 1000 2>/dev/null
    sudo ip link set $interface up 2>/dev/null
done

# TODO Discover actuators
# read -p "Press enter to discover actuators..."

# selected="$(for i in {1..4}; do for j in {1..5}; do echo $i$j ${actuators[$i$j]}; done; done | fzf -m --reverse --height 20 | awk '{print $1}')"
selected="$(for i in {1..4}; do for j in {1..5}; do echo $i$j; done; done)"


while read -r param; do
    param_name="$(echo $param | awk -F'|' '{print $1 "|" $3}')"
    param_code="$(echo $param | awk -F'x' '{print $NF}')"
    param_littleendian="${param_code:2:5}.${param_code:0:2}"

    echo "$param_name"
    echo "------------------------------"

    for id_dec in $selected; do
        id=$(printf "%.2X" "$id_dec")
        resp=""

        for interface in $ifaces; do
            resp="$(candump  -T 200 $interface,0200${id}FD:00FFFF00 & sleep .01;
                    cansend $interface "1100FD${id}#${param_littleendian}.00.00.00.00.00.00")"

            if [ "$resp" ]; then
                resp_val="$(echo "$resp" | awk '{print $(NF-3)$(NF-2)$(NF-1)$NF}')"
                resp_float="$(python -c "import struct; print(struct.unpack('<f', bytes.fromhex('$resp_val'))[0])")"
                echo "${actuators[$id_dec]} (can id $id_dec): $(printf "%.4f" $resp_float) | $resp"
                break
            fi
        done

        if [ -z "$resp" ]; then
            echo "No response from actuator $id_dec (${actuators[$id_dec]})"
        fi
    done
done <<< "$params"

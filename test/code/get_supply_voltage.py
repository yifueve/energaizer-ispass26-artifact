import os
import argparse
import json

def extract_voltage(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if "Graphics" in line and "mV" in line:
                # Split by ':' and get the second part (the value part)
                value_part = line.split(':')[1].strip()
                # Extract the numerical value
                voltage_str = value_part.split()[0]
                # Convert to float
                return float(voltage_str)
    # Return None if voltage information is not found
    return None

def extract_power(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            return float(line)
    return None

def extract_freq(file_path):
    freqs = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if "graphics" in line:
                continue
            if "MHz" in line:
                line = line.replace("MHz", '')
            freqs.append(int(line))

    freqs.sort()
    return freqs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--save_to_voltage', default='supply_voltage.json', type=str)
    parser.add_argument('--save_to_idle_power', default='idle_power.json', type=str)
    parser.add_argument('--pwd', type=str, required=True)

    args = parser.parse_args()

    # Query frequencies
    os.system('nvidia-smi --query-supported-clocks=gr --format=csv -i {} >> tmp_freq.csv'.format(args.device))
    freq = extract_freq('tmp_freq.csv')
    os.remove('tmp_freq.csv')

    voltages = {}
    powers = {}

    for f in freq:
        os.system('echo {} | sudo -S nvidia-smi -i {} -lgc {},{}'.format(args.pwd, args.device, f, f))
        os.system('nvidia-smi -q -d VOLTAGE -i {} >> tmp.csv'.format(args.device))
        os.system('nvidia-smi --query-gpu=power.draw -i {} --format=csv,noheader,nounits -f tmp2.csv'.format(args.device))

        # parse
        voltage = extract_voltage('tmp.csv')
        voltages[f] = voltage

        power = extract_power('tmp2.csv')
        powers[f] = power

        os.remove('tmp.csv')
        os.remove('tmp2.csv')

    # Save to json
    with open(args.save_to_voltage, 'w') as f:
        json.dump(voltages, f)

    with open(args.save_to_idle_power, 'w') as f:
        json.dump(powers, f)
    
    # Unlock clock freq
    os.system('echo {} | sudo -S nvidia-smi -i {} -rgc'.format(args.pwd, args.device))


if __name__ == '__main__':
    main()

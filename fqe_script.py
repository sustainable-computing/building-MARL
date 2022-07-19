import subprocess

def main():
    test_zones = ['Core_top', 'Core_mid', 'Core_bottom',
                  'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                  'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                  'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
    
    for zone in test_zones[:1]:
        stdout = f"data/fqe_data/{zone}.stdout"
        stderr = f"data/fqe_data/{zone}.stderr"
        proc = subprocess.Popen(["python", "fq_evaluation.py", "--zone", zone],
                                stderr=open(stderr, "w+"),
                                stdout=open(stdout, "w+"))
        print(zone, proc.pid)

if __name__ == "__main__":
    main()
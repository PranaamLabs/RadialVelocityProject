import pandas as pd

def get_lists(filename):
    with open(filename, 'r') as f:
        data = f.read()
    lines = data.splitlines()
    lines.pop(0)
    hjd = []
    radial_vel = []
    radial_vel_uncer = []

    for line in lines:
        if not line.strip(): 
            continue
            
        parts = line.split()
        hjd.append(float(parts[0]))
        radial_vel.append(float(parts[1]))
        radial_vel_uncer.append(float(parts[2]))

    return hjd, radial_vel, radial_vel_uncer


def make_csv(filename):
    file= filename[5:-4] #cut data/ and .txt
    hjd, radial_vel, radial_vel_uncer = get_lists(filename)
    data = {
        "HJD (days)" : hjd,
        "Radial Velocity (m/s)" : radial_vel,
        "Radial Velocity Uncertainty (m/s)" : radial_vel_uncer
    }

    df = pd.DataFrame(data)
    df.to_csv(f'data/csv/{file}.csv', index=False)

if __name__=="__main__":
    #make_csv('data/butler.txt')
    #make_csv('data/howardfulton.txt')
    #make_csv('data/naef_14Her_b.txt')
    make_csv('data/hf_hd3651b.txt')
import numpy as np
import random
import argparse

# --- SantaLucia-inspired dinucleotide weights ---
DINUC_WEIGHT = {
    'AA':1.00,'TT':1.00,'AT':0.88,'TA':0.58,
    'CA':1.45,'TG':1.45,'GT':1.44,'AC':1.44,
    'CT':1.28,'AG':1.28,'GA':1.30,'TC':1.30,
    'CG':2.17,'GC':2.24,'GG':1.84,'CC':1.84
}

ALPHABET = ['A','C','G','T']


# -------------------------------
# 1. Sequence generation
# -------------------------------
def generate_sequence(n, entropy_level=1.0):
    if entropy_level == 1.0:
        probs = [0.25]*4
    else:
        biased = np.array([0.4, 0.1, 0.1, 0.4])
        uniform = np.ones(4)/4
        probs = entropy_level * uniform + (1-entropy_level) * biased
        probs /= probs.sum()
    
    return ''.join(np.random.choice(ALPHABET, size=n, p=probs))


# -------------------------------
# 2. Mutation
# -------------------------------
def mutate(seq, mutation_rate=0.1):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < mutation_rate:
            seq[i] = random.choice(ALPHABET)
    return ''.join(seq)


# -------------------------------
# 3. Causal influence
# -------------------------------
def compute_influence(seq, k=5, lambda_decay=0.5):
    n = len(seq)

    # causal dinucleotide contribution
    f = np.zeros(n)
    for k_pos in range(1, n):
        dinuc = seq[k_pos-1] + seq[k_pos]
        f[k_pos] = DINUC_WEIGHT[dinuc]

    # causal aggregation
    I = np.zeros(n)
    for i in range(n):
        for k_pos in range(max(0, i-k), i+1):
            dist = i - k_pos
            I[i] += np.exp(-lambda_decay * dist) * f[k_pos]

    return I


# -------------------------------
# 4. Influence → gap (DIRECT mapping)
# -------------------------------
def influence_to_gap(influence, w_min=1, w_max=10, noise=0.0):
    inf = np.array(influence)

    inf_norm = (inf - inf.min()) / (inf.max() - inf.min() + 1e-8)

    gaps = w_min + (w_max - w_min) * inf_norm

    if noise > 0:
        gaps += np.random.normal(0, noise, size=len(gaps))

    return np.clip(gaps.astype(int), w_min, w_max)


# -------------------------------
# 5. Generate instance
# -------------------------------
def generate_instance_m(
    m=3,
    n=100,
    entropy=1.0,
    mutation_rate=0.1,
    k=10, #fixed window size
    lambda_decay=0.5,
    gap_min=1,
    gap_max=10,
    gap_noise=0.0
):
    S0 = generate_sequence(n, entropy)

    S_list, I_list, G_list = [], [], []

    for _ in range(m):
        Sj = mutate(S0, mutation_rate)
        Ij = compute_influence(Sj, k, lambda_decay)
        Gj = influence_to_gap(Ij, gap_min, gap_max, gap_noise)

        S_list.append(Sj)
        I_list.append(Ij)
        G_list.append(Gj)

    return {"S": S_list, "G": G_list, "I": I_list, "S0": S0}


# -------------------------------
# 6. Save to file
# -------------------------------
def save_instance(instance, filepath):
    with open(filepath, "w") as f:
        m = len(instance["S"])
        
        # first line: number of sequences
        f.write(str(m) + "\n")
        
        # then sequence + gaps per block
        for seq, gaps in zip(instance["S"], instance["G"]):
            f.write(seq + "\n")
            f.write(" ".join(map(str, gaps)) + "\n")

# -------------------------------
# 7. CLI
# -------------------------------
if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-o", "--output", type=str, help="Output file path")
    #parser.add_argument("-m", type=int, default=3)
    #parser.add_argument("-n", type=int, default=100)
    #parser.add_argument("-r", type=int, default=10) #now many instances in the group (=10 by default)

    #args = parser.parse_args()
    sigma = 4 # always DNA and RNA molecules
    mx = [2, 3, 5, 10]
    nx = [50, 100, 200, 500]
    
    for m in mx:
        for n in nx:
            for i in range(10):
    
                instance = generate_instance_m(m, n)
                save_instance(instance, f"real_{n}_{m}_{i}.out" )  #args.output)
                print(f"Instance saved to real_{n}_{m}_{i}.out")

        

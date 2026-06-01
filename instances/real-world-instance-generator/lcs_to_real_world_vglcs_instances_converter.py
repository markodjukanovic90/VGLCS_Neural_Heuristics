import numpy as np
import random
import argparse
import os

# =========================================================
# PARAMETERS
# =========================================================

DINUC_WEIGHT = {
    'AA':1.00,'TT':1.00,'AT':0.88,'TA':0.58,
    'CA':1.45,'TG':1.45,'GT':1.44,'AC':1.44,
    'CT':1.28,'AG':1.28,'GA':1.30,'TC':1.30,
    'CG':2.17,'GC':2.24,'GG':1.84,'CC':1.84
}

ALPHABET = ['A','C','G','T']


# =========================================================
# 1. Sequence generation
# =========================================================

def generate_sequence(n, entropy_level=1.0):
    if entropy_level == 1.0:
        probs = [0.25]*4
    else:
        biased = np.array([0.4, 0.1, 0.1, 0.4])
        uniform = np.ones(4)/4
        probs = entropy_level * uniform + (1-entropy_level) * biased
        probs /= probs.sum()
    
    return ''.join(np.random.choice(ALPHABET, size=n, p=probs))


def mutate(seq, mutation_rate=0.1):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < mutation_rate:
            seq[i] = random.choice(ALPHABET)
    return ''.join(seq)


# =========================================================
# 2. Causal influence
# =========================================================

def compute_influence(seq, k=10, lambda_decay=0.5):
    n = len(seq)

    # causal dinucleotide contribution
    f = np.zeros(n)
    for i in range(1, n):
        f[i] = DINUC_WEIGHT[seq[i-1] + seq[i]]

    # causal aggregation
    I = np.zeros(n)
    for i in range(n):
        for j in range(max(0, i-k), i+1):
            dist = i - j
            I[i] += np.exp(-lambda_decay * dist) * f[j]

    return I


# =========================================================
# 3. Influence → gap
# =========================================================

def influence_to_gap(influence, w_min=1, w_max=10, noise=0.0):
    inf = np.array(influence)

    inf_norm = (inf - inf.min()) / (inf.max() - inf.min() + 1e-8)

    gaps = w_min + (w_max - w_min) * inf_norm

    if noise > 0:
        gaps += np.random.normal(0, noise, size=len(gaps))

    return np.clip(gaps.astype(int), w_min, w_max)


# =========================================================
# 4. Synthetic instance generation
# =========================================================

def generate_instance_m(
    m=3,
    n=100,
    entropy=1.0,
    mutation_rate=0.1,
    k=10,
    lambda_decay=0.5,
    gap_min=1,
    gap_max=10,
    gap_noise=0.0
):
    S0 = generate_sequence(n, entropy)

    S_list, G_list = [], []

    for _ in range(m):
        Sj = mutate(S0, mutation_rate)
        Ij = compute_influence(Sj, k, lambda_decay)
        Gj = influence_to_gap(Ij, gap_min, gap_max, gap_noise)

        S_list.append(Sj)
        G_list.append(Gj)

    return S_list, G_list


# =========================================================
# 5. Load LCS instance
# =========================================================

def load_lcs_instance(filepath):
    sequences = []

    with open(filepath, "r") as f:
        lines = f.readlines()

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 2:
                sequences.append(parts[1])

    return sequences


# =========================================================
# 6. Generate gaps for given sequences
# =========================================================

def generate_gaps_for_sequences(
    sequences,
    k=10,
    lambda_decay=0.5,
    gap_min=1,
    gap_max=10,
    gap_noise=0.0
):
    G_list = []

    for seq in sequences:
        I = compute_influence(seq, k, lambda_decay)
        G = influence_to_gap(I, gap_min, gap_max, gap_noise)
        G_list.append(G)

    return G_list


# =========================================================
# 7. Save VGLCS format
# =========================================================

def save_vglcs_instance(sequences, gaps, filepath):
    with open(filepath, "w") as f:
        m = len(sequences)
        f.write(str(m) + "\n")

        for seq, g in zip(sequences, gaps):
            f.write(seq + "\n")
            f.write(" ".join(map(str, g)) + "\n")


# =========================================================
# 8. Convert directory (LCS → VGLCS)
# =========================================================

def convert_lcs_dir_to_vglcs(
    input_dir,
    output_dir,
    k=10,
    lambda_decay=0.5,
    gap_min=1,
    gap_max=10,
    gap_noise=0.0
):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if not os.path.isfile(input_path):
            continue
        print(input_path)
        sequences = load_lcs_instance(input_path)

        gaps = generate_gaps_for_sequences(
            sequences,
            k,
            lambda_decay,
            gap_min,
            gap_max,
            gap_noise
        )

        output_path = os.path.join(output_dir, filename)
        save_vglcs_instance(sequences, gaps, output_path)

        print(f"Converted: {filename}")


# =========================================================
# 9. MAIN CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["synthetic", "convert"], required=True)

    # synthetic
    parser.add_argument("-m", type=int, default=3)
    parser.add_argument("-n", type=int, default=100)

    # convert
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    # common
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--lambda_decay", type=float, default=0.5)
    parser.add_argument("--gap_min", type=int, default=1)
    parser.add_argument("--gap_max", type=int, default=10)
    parser.add_argument("--gap_noise", type=float, default=0.0)

    args = parser.parse_args()

    # -------------------------------
    # SYNTHETIC MODE
    # -------------------------------
    if args.mode == "synthetic":
        sequences, gaps = generate_instance_m(
            m=args.m,
            n=args.n,
            k=args.k,
            lambda_decay=args.lambda_decay,
            gap_min=args.gap_min,
            gap_max=args.gap_max,
            gap_noise=args.gap_noise
        )

        save_vglcs_instance(sequences, gaps, "synthetic.out")
        print("Saved synthetic.out")

    # -------------------------------
    # CONVERSION MODE
    # -------------------------------
    elif args.mode == "convert":
        convert_lcs_dir_to_vglcs(
            args.input_dir,
            args.output_dir,
            args.k,
            args.lambda_decay,
            args.gap_min,
            args.gap_max,
            args.gap_noise
        )

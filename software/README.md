# Beam Search with Neural Network Guidance

## Compilation

To compile the program, execute:

```bash
make
```

This will generate the executable `beamsearch`.

---

# Solving an Instance

The following command-line parameters are available when solving an instance:

| Parameter | Description |
|------------|------------|
| `-i` | Path to the input instance. |
| `-o` | Path to the output file where results will be stored. |
| `-b` | Beam width. |
| `-t` | Time limit (in seconds). |
| `-hidden_layers` | Number of hidden layers in the neural network guiding the beam search. |
| `-units` | Number of hidden units in each hidden layer (one value per layer). |
| `-weights` | Indicates that a file named `weights.txt` is available in the execution directory and should be used to initialize the neural network weights. |
| `-activation_function` | Activation function used by the neural network. |
| `-feature_configuration` | Feature set used to represent beam search states. |

## Activation Functions

| Value | Function |
|---------|---------|
| `1` | tanh |
| `2` | ReLU |
| `3` | Sigmoid |

## Feature Configurations

| Value | Description |
|---------|-------------|
| `1` | Uses the maximum, minimum, standard deviation, and average values of p^(L,v) and l^v, together with the length of the partial solution associated with node v (9 features in total). |
| `2` | Configuration 1 + alphabet size. |
| `3` | Configuration 2 + number of input strings and restricted strings. |
| `4` | Configuration 3 + lengths of input and restricted strings (assumes all input strings have equal length and all restricted strings have equal length). |

## Example

```bash
./beamsearch \
    -i sigma-4/mglcs_5_500_4_1.txt \
    -o out-111.txt \
    -hidden_layers 3 \
    -units 5 5 5 \
    -weights \
    -activation_function 1 \
    -feature_configuration 1
```

---

# Training a Neural Network

The following parameters are available when training a neural network:

| Parameter | Description |
|------------|------------|
| `-hidden_layers` | Number of hidden layers. |
| `-units` | Number of hidden units in each hidden layer (one value per layer). |
| `-train` | Path to a training instance. Use one `-train` argument for each training file. |
| `-validation` | Path to a validation instance. Use one `-validation` argument for each validation file. |
| `-weight_limit` | Maximum absolute value allowed for network weights during training. |
| `-training_beam_width` | Beam width used during training. |
| `-training_time_limit` | Training time limit (in seconds). |
| `-activation_function` | Activation function used by the neural network. |
| `-feature_configuration` | Feature configuration used during training (same options as in solving mode). |
| `-ga_configuration` | Genetic algorithm variant used for training. |
| `-rho` | Elite inheritance probability (required only when using BRKGA). |

## Genetic Algorithm Configurations

| Value | Description |
|---------|-------------|
| `1` | Random-Key Genetic Algorithm (RKGA). |
| `2` | Biased Random-Key Genetic Algorithm (BRKGA). Requires parameter `-rho`. |
| `3` | RKGA with lexicase selection for elite population selection. |

## Example

```bash
./beamsearch \
    -train train/mglcs_3_500_2_0.txt \
    -train train/mglcs_3_100_4_0.txt \
    -validation validate/mglcs_10_200_2_1.txt \
    -validation validate/mglcs_5_500_4_1.txt \
    -hidden_layers 3 \
    -units 5 5 5 \
    -weight_limit 1 \
    -training_beam_width 10 \
    -training_time_limit 200 \
    -activation_function 3 \
    -feature_configuration 1 \
    -ga_configuration 1
```

---

# Neural Network Weights

After training, the learned neural network weights are written to:

```text
weights.txt
```

This file can subsequently be used during solving by specifying the `-weights` flag.

---

# Notes

- The number of values provided after `-units` must match the value specified by `-hidden_layers`.
- When using `-weights`, a valid `weights.txt` file must be present in the current working directory.
- Feature configurations used during solving should match those used during training.
- For BRKGA (`-ga_configuration 2`), the elite inheritance probability must be specified via `-rho`.

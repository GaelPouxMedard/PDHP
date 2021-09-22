# PDHP
The script PDHP.py is an implementation of the Powered Dirichlet-Hawkes process prior coupled to a simple Dirichlet-Multinomial language model, that can be ran from console.

## Usage
Run the script PDHP.py using the following syntax: [keyword]=[value][space]
The keywords are:

- data_file (str) (*) => The file that contains events
- kernel_file (str) (**) => The file that contains Gaussian kernel's parameters
- output_folder (str) (***) => Where to save the files (words index and particles every 1000 iterations)
- r (float or comma separated floats) => Exponent used for the Powered Dirichlet-Hawkes prior. If a list is provided, experiments will be ran for each value
- runs (int) => Number of runs on the given dataset
- theta0 (float) => Value of the symmetric Dirichlet-Multinomial prior to model textual content
- alpha0 (float) => Value of the symmetric Dirichlet prior to model temporal kernels weights
- number_samples (int) => Number of samples using in Gibbs sampling inference of alpha
- number_particles (int) => Number of particles used by SMC algorithm; each aprticle keeps track clusters allcoations hypotheses
- print_progress (bool) => Whether to print script's progress every 100 documents


(*) The data file must follow the following structure for each event entry:
```
[timestamp][tabulation][comma-separated words][end line]
```

(**) The kernel file must follow the following structure:
```
[lambda0][end line]
[end line]
[mean_1][end line]
[mean_2][end line]
...
[mean_K][end line]
[end line]
[sigma_1][end line]
[sigma_2][end line]
...
[sigma_K][end line]
```


(***) The particles output file has the following structure:
```
Particle[tabulation]particle index (int)[tabulation]particle weight (float)[tabulation]events clusters (array of int)[end line]
Cluster[tabulation]cluster index (int)[tabulation]alpha0 (float)[tabulation]inferred alpha (array of floats)[tabulation]textual likelihood (float) [tabulation]number of words in the cluster (int)[tabulation]words distribution (array of ints)[end line]
Cluster[tabulation]cluster index (int)[tabulation]alpha0 (float)[tabulation]inferred alpha (array of floats)[tabulation]textual likelihood (float) [tabulation]number of words in the cluster (int)[tabulation]words distribution (array of ints)[end line]
...
Cluster[tabulation]cluster index (int)[tabulation]alpha0 (float)[tabulation]inferred alpha (array of floats)[tabulation]textual likelihood (float) [tabulation]number of words in the cluster (int)[tabulation]words distribution (array of ints)[end line]
Particle[tabulation]particle index (int)[tabulation]particle weight (float)[tabulation]events clusters (array of int)[end line]
Cluster[tabulation]cluster index (int)[tabulation]alpha0 (float)[tabulation]inferred alpha (array of floats)[tabulation]textual likelihood (float) [tabulation]number of words in the cluster (int)[tabulation]words distribution (array of ints)[end line]
Cluster[tabulation]cluster index (int)[tabulation]alpha0 (float)[tabulation]inferred alpha (array of floats)[tabulation]textual likelihood (float) [tabulation]number of words in the cluster (int)[tabulation]words distribution (array of ints)[end line]
```

## Dependencies
Numpy, Scipy, re








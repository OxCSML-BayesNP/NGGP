# The Normal-Generalised Gamma-Pareto process
Code for the paper: "The Normal-Generalised Gamma-Pareto process: A novel pure-jump Levy process with flexible tail and jump-activity properties".

## Requirements
This code buids on the Particles Library (https://github.com/nchopin/particles) (version 0.1).
To install the required pacakges, run
```
pip install -r requirements.txt
```

## Datasets:
All return data can be found in the folder data/, where there are two folders:

  - data_minute_tech: Dataset composed of the time-series of the stock prices of six large technology companies: Apple (AAPL), Amazon (AMZN), Facebook (FB), Google (GOOG), Microsoft (MSFT) and Netflix (NFLX). The data are sampled every minute from the 10th of July 2019 until the 22nd of January 2020, with approximately 50,000 time points. We subsample 1500 observations as training data to estimate the parameters of each model, and use the rest of the observations as test data.

  - oxford: Dataset obtained from the Realized library (https://realized.oxford-man.ox.ac.uk), put into ```data``` folder, and run ```process_oxford.py```). We collected 14 daily stock data from 05-11-2007 to 07-10-2011 (around the time of subprime mortgage crisis) Download the csv file from the link, put into ```data``` folder, and run ```process_oxford.py```

## IID Simple model

The scripts and results are in the folder simple/

Available models:

  - Normal-Generalised Gamma-Pareto Process (gbfry in scripts)
  - Generalized Gamma Process (gamma in scripts)
  - Normal Stable (ns in scripts)
  - Generalized hyperbolic (gh in scripts)
  - Variance Gamma (vgamma4 for the parametrization og the ggp, vgamma3 for the parametrization of the gh)
  - Normal Inverse-Gamma (nig in scripts)
  - Student (student in scripts)

### Run the scripts

1- Learning from train: You have to use the file run_iid.py to run a chain, which can be done from the terminal with the following line

<code>
 python3 run_iid.py --Nx 2500 --niter 5000 --filename ../data/data_minute_tech/FB_min_train.pkl --model gbfry --run_name chain1
 </code>

 Parameters:

     --Nx 2500: number of particles (2500 here)
     --niter 5000: number of mcmc iterations
     --filename: train data
     --model gbfry: Model used (either gbfry or gamma (even though gamma is actually ggp that is run))
     --run_name chain1: Name of the current chain, usually the names are just chain1, chain2 and chain3

2- Get posterior of the parameters: You have to use the file summarize.py

<code>
python3 summarize.py --filename ../data/data_minute_tech/FB_min_train.pkl --model gbfry --no_states
</code>

You can also run the command
<code>
  sh script_summarize.sh
</code>
to rrun the summarize script on all the tech companies data.

3-Get posterior predictive of y and states: You have to use the file assess_ordered.py

<code>
python3 assess_ordered.py --filename ../data/data_minute_tech/FB_min_train.pkl --thin 10 --model gamma
</code>

Parameters:

     --thin 10, thinning, keep only every 10 samples of the parameters

You can also simply run the command
<code>
  sh script_assess.sh FB
</code>
to run the assess script with all models on a given dataset (here FB)

### Plots of the paper:

The results of our simulations for the iid simple model can be found in simple/plots

## Levy-driven stochastic volatility model
The codes and scripts are located in ```complex``` folder.

### Available models
  - Normal-Gamma process (```gamma_driven_ldsv.py```)
  - Normal-Generalised Gamma-Pareto process (```gbfry_driven_ldsv.py```)

### Running the sampler
```
python run.py --filename filename --model (gbfry or gamma) \
              --Nx num_particles --burnin burnin \
              --niter num_iterations \
              --run_name chain1
```
Put ```gbfry``` for NGGP and ```gamma``` for normal-Gamma process.

### Plotting and assessing
To compute the metrics (only for ```oxford``` data having ground-truth states),
```
python assess.py --filename filename --run_names chain1 chain2 chain3
```

To see the parameter estimates,
```
python plot_chains.py --filename filename --model (gbfry or gamma) \
                      --run_names chain1 chain2 chain3
```

To see the recovered states,
```
python plot_states.py --filename filename --run_names chain1 chain2 chain3
```

# common commands to run CmdStan

stan installation directory: `/opt/cmdstan`

stan file like `~/coder/testfile.stan` and data file like `~/coder/testfile.data.json`

if exit init file like `~/coder/testfile.init.json`

```bash
cd /opt/cmdstan && make -j ~/coder/testfile && cd ~/coder

./testfile optimize data file=testfile.data.json output file=testfile_optim.csv

./testfile sample \
	num_chains=4 num_samples=50000 num_warmup=10000 thin=5 \
	init=testfile.init.json \
	data file=testfile.data.json \
	output file=testfile_fit.csv diagnostic_file=testfile_dia.csv \
	refresh=0 num_threads=4

/opt/cmdstan/bin/stansummary testfile_fit_*.csv

./testfile variational data file=testfile.data.json output file=testfile_advi.csv

./testfile diagnose data file=testfile.data.json output file=testfile_diag.csv
```

if need posterior predictive checks:
```bash
for i in {1..4} \
	do \
		./testfile generate_quantities \
			fitted_params=testfile_fit_${i}.csv \
			data file=testfile.data.json \
			output file=testfile_ppc_${i}.csv & \
	done
```

#!/bin/bash

id=$(sbatch run.sh)
watch "cat slurm-${id: -7}.out"
[davisamu@login-5.SAGA ~/home/sent_graph_followup/perin]$ cat run.sh 
#!/bin/bash

#SBATCH --job-name=SENTIMENT_PERIN
#SBATCH --account=nn9851k
#SBATCH --time=00-01:00:00
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-pytorch/1.6.0-gomkl-2019b-cuda-10.1.243-Python-3.7.4
module load nlpl-transformers/4.5.1-gomkl-2019b-Python-3.7.4
module load nlpl-scipy-ecosystem/2021.01-gomkl-2019b-Python-3.7.4
module load sentencepiece/0.1.94-gomkl-2019b-Python-3.7.4

TRANSFORMERS_OFFLINE=1 python3 train.py --config config/base_norec.yaml
#!/bin/bash

# id=$(sbatch run.sh)
# watch "cat slurm-${id: -7}.out | tail -n 37"

# id=$(sbatch run.sh config/node_multibooked_ca.yaml)
# echo "node_multibooked_ca -- ${id}"

# id=$(sbatch run.sh config/node_multibooked_eu.yaml)
# echo "node_multibooked_eu -- ${id}"

# id=$(sbatch run.sh config/node_norec.yaml)
# echo "node_nodec -- ${id}"

# id=$(sbatch run.sh config/node_mpqa.yaml)
# echo "node_mpqa-- ${id}"

# id=$(sbatch run.sh config/node_darmstadt.yaml)
# echo "node_darmstadt -- ${id}"

id=$(sbatch run.sh config/edge_multibooked_ca.yaml)
echo "edge_multibooked_ca -- ${id}"

# id=$(sbatch run.sh config/seq_multibooked_ca.yaml)
# echo "seq_multibooked_ca -- ${id}"

# id=$(sbatch run.sh config/seq_multibooked_eu.yaml)
# echo "seq_multibooked_eu -- ${id}"

# id=$(sbatch run.sh config/seq_norec.yaml)
# echo "seq_norec -- ${id}"

# id=$(sbatch run.sh config/seq_mpqa.yaml)
# echo "seq_mpqa -- ${id}"

# id=$(sbatch run.sh config/seq_darmstadt.yaml)
# echo "seq_darmstadt -- ${id}"

# sent_graph_followup

## current directions and problems

1. Convert sentiment jsons to MRP json files
    - made changes to mtool to parse 'norec'
    - python3 mtool/main.py --read norec --write mrp "$indata" "$outdata"
    - two options:
        a. without --reify flag: polarity is a property of the expression
        b. with --reify flag: we add a top node for a sentence and the polarity is a property of the edge between this node and each sentiment expression.

    - Currently: trying to load the data in either eds/ptg format
        - problems: unlike MRP, we have sentences which have no nodes/edges/labels. This currently gives errors when trying to read in the data.

2. Run PERIN on the NoReC data in MRP format.
    - Does it work?
    - How does it compare to the ACL results?


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

## PERIN Notes

- relevant files to create
    - perin/model/head/norec_head.py
    - perin/data/parser/from_mrp/norec_parser.py
    - perin/data/parser/to_mrp/norec_parser.py
- eds and ptg as base
    - eds is pretty _vanilla_, but has continuous nodes
    - ptg has some more wild extras that can be ignored
    - ptg has special top nodes making prediction redundant -> should be the
        same for norec
- differences to mrp
    - no properties in norec-graphs
    - labels are only absolute

### Code Questions

- how are properties treated in PERIN? 
    - `node["properties"] = {"transformed": int("property" in node)}`
    - `utils.normalize_properties(data)`
    - *David: There are two ways:*
        - *1) Properties are transformed into regular nodes so that they can use the relative encoding (useful e.g. for quantities). Then we also need to predict the flag "transformed" to convert the properties back.*
        - *2) In PTG, the properties are predicted as part of each node -- there is too much of them and all of them (if I remember correctly) use a limited vocabulary.*
        - *I'm not sure which one is more suitable for the extra properties in some datasets but it should be fairly easy to use either of them.*
- what about top nodes? There seems to always be only one in PERIN 
    - `sentence["top"] = sentence["tops"][0]`
    - *David: Yeah, the code assumes there is only a single top node but it's possible to extend it to multiple tops. Is there a dataset where this explicit "top" prediction is needed?*

## Extras

With the script `write_graphs.sh` one can create images of the sentiment graphs.
To do so first run the python script `get_ids.py`, and then `write_graphs.sh`.
This takes however a very long time, and might be less than optimal.
The `--strings` option in the script can be deleted to give spans instead of
tokens.


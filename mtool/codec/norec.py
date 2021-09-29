import json
import operator
import os
import sys

from graph import Graph


def read(fp, text=None, reify=False, strict=False, node_centric=False):
    assert not (reify and node_centric), "You can't reify node-centric representation at the moment."

    def anchor(node):
        anchors = list()
        for string in node[1]:
            string = string.split(":")
            anchors.append({"from": int(string[0]), "to": int(string[1])})
        return anchors

    for native in json.load(fp):
        map = dict()
        try:
            graph = Graph(native["sent_id"], flavor=1, framework="norec")
            graph.add_input(native["text"])
            if reify:
                top = graph.add_node(top=True)
            for opinion in native["opinions"]:
                expression = opinion["Polar_expression"]
                properties, values = list(), list()

                if node_centric:
                    expression = graph.add_node(
                        label=opinion["Polarity"],
                        top=True,
                        properties=properties,
                        values=values,
                        anchors=anchor(expression),
                    )
                else:
                    if not reify:
                        properties = ["polarity"]
                        values = [opinion["Polarity"]]
                    expression = graph.add_node(
                        label="Expression",
                        top=not reify,
                        properties=properties,
                        values=values,
                        anchors=anchor(expression),
                    )

                if reify:
                    graph.add_edge(top.id, expression.id, opinion["Polarity"])

                source = opinion["Source"]
                if len(source[1]):
                    key = tuple(source[1])
                    if strict and key in map:
                        source = map[key]
                    else:
                        source = graph.add_node(
                            label="Source" if (not strict) or node_centric else None,
                            anchors=anchor(source),
                        )
                        map[key] = source
                    graph.add_edge(
                        expression.id, source.id, "Source" if strict and (not node_centric) else None
                    )

                target = opinion["Target"]
                if len(target[1]):
                    key = tuple(target[1])
                    if strict and key in map:
                        target = map[key]
                    else:
                        target = graph.add_node(
                            label="Target" if (not strict) or node_centric else None,
                            anchors=anchor(target),
                        )
                        map[key] = target
                    graph.add_edge(
                        expression.id, target.id, "Target" if strict and (not node_centric) else None
                    )
            yield graph, None

        except Exception as error:
            print(
                "codec.norec.read(): ignoring {}: {}" "".format(native, error),
                file=sys.stderr,
            )

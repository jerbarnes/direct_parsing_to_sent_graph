import os
import json
import torch
import sys

from subprocess import run
from data.batch import Batch

sys.path.append("../evaluation")
from evaluate_single_dataset import evaluate


def predict(model, data, input_paths, raw_input_paths, args, logger, output_directory, gpu, mode="validation", epoch=None):
    model.eval()
    input_files = {(f, l): input_paths[(f, l)] for f, l in args.frameworks}

    sentences = {(f, l): {} for f, l in args.frameworks}
    for framework, language in args.frameworks:
        with open(input_files[(framework, language)], encoding="utf8") as f:
            for line in f.readlines():
                line = json.loads(line)

                line["nodes"] = []
                line["edges"] = []
                line["tops"] = []
                line["framework"] = framework
                line["language"] = language
                sentences[(framework, language)][line["id"]] = line

    for i, batch in enumerate(data):
        with torch.no_grad():
            all_predictions = model(Batch.to(batch, gpu), inference=True)

        for (framework, language), predictions in all_predictions.items():
            for prediction in predictions:
                for key, value in prediction.items():
                    sentences[(framework, language)][prediction["id"]][key] = value

    for framework, language in args.frameworks:
        output_path = f"{output_directory}/prediction_{mode}_{epoch}_{framework}_{language}.json"
        with open(output_path, "w", encoding="utf8") as f:
            for sentence in sentences[(framework, language)].values():
                json.dump(sentence, f, ensure_ascii=False)
                f.write("\n")
                f.flush()

        run(["./convert.sh", output_path, raw_input_paths[(framework, language)]])
        prec, rec, f1 = evaluate(raw_input_paths[(framework, language)], f"{output_path}_converted")

        if logger is not None:
            print(mode, f1, flush=True)
            logger.log_evaluation(prec, rec, f1, framework, language, mode)

    return f1

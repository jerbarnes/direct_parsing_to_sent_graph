import os
import json
import torch
import wandb

from PIL import Image
from subprocess import run

from data.batch import Batch
from utility.evaluate import evaluate
from utility.utils import resize_to_square


def sentence_condition(s, f, l):
    return ("framework" not in s or f == s["framework"]) and ("framework" in s or f in s["targets"])


def predict(model, data, input_paths, raw_input_paths, args, logger, output_directory, gpu, epoch=None):
    model.eval()
    input_files = {(f, l): input_paths[(f, l)] for f, l in args.frameworks}

    sentences = {(f, l): {} for f, l in args.frameworks}
    for framework, language in args.frameworks:
        with open(input_files[(framework, language)], encoding="utf8") as f:
            for line in f.readlines():
                line = json.loads(line)

                if not sentence_condition(line, framework, language):
                    continue

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
        output_path = f"{output_directory}/prediction_{epoch}_{framework}_{language}.json"
        with open(output_path, "w", encoding="utf8") as f:
            for sentence in sentences[(framework, language)].values():
                json.dump(sentence, f, ensure_ascii=False)
                f.write("\n")
                f.flush()

        score = run(["./evaluate.sh", output_path, raw_input_paths[(framework, language)]], capture_output=True, text=True)
        print(score.stdout, flush=True)
        print(score.stderr, flush=True)
        score = float(score.stdout[len("Sentiment Tuple F1: "):-1])
        logger.log_evaluation(score, framework, language)

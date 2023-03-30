// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { localization } from "@responsible-ai/localization";

import {
  ILabeledStatistic,
  TotalCohortSamples
} from "../Interfaces/IStatistic";

import { JointDataset } from "./JointDataset";

export enum QuestionAnsweringMetrics {
  ExactMatchRatio = "exactMatchRatio",
  F1Score = "f1Score"
}

function calculatef1Score(actual: string[], predicted: string[]): number {
  let sum = 0;
  for (const [i] of actual.entries()) {
    const actualTokens = actual[i].split(" ");
    const predTokens = predicted[i].split(" ");
    const truePositives = actualTokens.filter((value) =>
      predTokens.includes(value)
    ).length;
    const falsePositives = predTokens.filter(
      (value) => !actualTokens.includes(value)
    ).length;
    const falseNegatives = actualTokens.filter(
      (value) => !predTokens.includes(value)
    ).length;

    let precision = 0;
    let recall = 0;
    if (truePositives !== 0 || falsePositives !== 0) {
      precision = truePositives / (truePositives + falsePositives);
    }
    if (truePositives !== 0 || falseNegatives !== 0) {
      recall = truePositives / (truePositives + falseNegatives);
    }
    if (precision !== 0 || recall !== 0) {
      sum = sum + 2 * ((precision * recall) / (precision + recall));
    }
  }
  return sum / actual.length;
}

export const generateQuestionAnsweringStats: (
  jointDataset: JointDataset,
  selectionIndexes: number[][]
) => ILabeledStatistic[][] = (
  jointDataset: JointDataset,
  selectionIndexes: number[][]
): ILabeledStatistic[][] => {
  return selectionIndexes.map((selectionArray) => {
    const matchingLabels = [];
    const count = selectionArray.length;
    let trueYs: string[] = [];
    let predYs: string[] = [];
    if (jointDataset.strDataDict) {
      trueYs = jointDataset.strDataDict.map(
        (row) => row[JointDataset.TrueYLabel]
      );
      predYs = jointDataset.strDataDict.map(
        (row) => row[JointDataset.PredictedYLabel]
      );
    }

    const trueYSubset = selectionArray.map((i) => trueYs[i]);
    const predYSubset = selectionArray.map((i) => predYs[i]);
    matchingLabels.push(
      trueYSubset.filter((trueY, index) => trueY === predYSubset[index]).length
    );
    const sum = matchingLabels.reduce((prev, curr) => prev + curr, 0);
    const exactMatchRatio = sum / selectionArray.length;

    const f1Score = calculatef1Score(trueYs, predYs);

    return [
      {
        key: TotalCohortSamples,
        label: localization.Interpret.Statistics.samples,
        stat: count
      },
      {
        key: QuestionAnsweringMetrics.ExactMatchRatio,
        label: localization.Interpret.Statistics.exactMatchRatio,
        stat: exactMatchRatio
      },
      {
        key: QuestionAnsweringMetrics.F1Score,
        label: localization.Interpret.Statistics.f1Score,
        stat: f1Score
      }
    ];
  });
};

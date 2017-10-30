--------------------------------------------------------------------------------
-- Parameters of the preprocessed Cityscapes dataset
-- Written by Camille Couprie, Pauline Luc, Natalia Neverova
--------------------------------------------------------------------------------
-- Copyright 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

classes = {'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
          'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
          'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
          'bicycle', 'unlabeled'}

nclasses = #classes-1

colormap = {[1] = {0.5, 0.25, 0.5},    -- road
            [2] = {0.95, 0.14, 0.91},  -- sidewalk
            [3] = {0.27, 0.27, 0.27}, -- building
            [4] = {0.4, 0.4, 0.61},    -- wall
            [5] = {0.745, 0.6, 0.6},   -- fence
            [6] = {0.6, 0.6, 0.6},     -- pole
            [7] = {0.98, 0.66, 0.11},  -- traffic light
            [8] = {0.86, 0.86,  0},    -- traffic sign
            [9] = {0.41, 0.55, 0.14},  -- vegetation
            [10] = {0.59, 0.98, 0.59},  -- terrain
            [11] = {0.27, 0.51, 0.71},  -- sky
            [12] = {0.86, 0.27, 0.23},  -- person
            [13] = {1, 0, 0},           -- rider
            [14] = {0, 0, 0.55},        -- car
            [15] = {0, 0, 0.27},        -- truck
            [16] = {0, 0.55, 0.39},     -- bus
            [17] = {0, 0.31, 0.39},     -- train
            [18] = {0, 0, 0.9},         -- motorcycle
            [19] = {0.46, 0.04, 0.13},  -- bicycle
            [20] = {0, 0, 0}}           -- unevaluated

movingObjects = {12, 13, 14, 15, 16, 17, 18, 19}

oh, ow, oc = 128, 256, 3

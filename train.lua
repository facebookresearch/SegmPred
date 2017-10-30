--------------------------------------------------------------------------------
-- Training a multiscale convnet to predict next frame from some previous images
-- and semantic segmentations
-- Written by Camille Couprie, Pauline Luc, Natalia Neverova
--------------------------------------------------------------------------------
-- Copyright 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'paths'
local display = require 'display'
local tnt = require 'torchnet'
paths.dofile('utils/metrics.lua')
paths.dofile('utils/dataset.lua')
paths.dofile('utils/utils.lua')

-- setting options -------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:option('--devid', 1, 'GPU id')
cmd:option('--saveDir','saves','Directory to save the data')
cmd:option('--dataDir','Data/', 'dataset path')
cmd:option('--optim', 'sgd', 'Optim scheme')
cmd:option('--nEpoches', 5000, 'Number of epoches')
cmd:option('--nIters', 1000, 'Number of training iterations per epoch')
cmd:option('--nItersTest', 25, 'Number of testing iterations per epoch')
cmd:option('--lr', 0.01, 'Learning rate of the frame generator')
cmd:option('--batchSize', 4, 'Minibatch size')
cmd:option('--nInputFrames', 4, 'Number of input frames (excluding prediction)')
cmd:option('--nTargetFrames', 1, 'Number of predicted frames')
cmd:option('--hInput', 64, 'Frame height')
cmd:option('--wInput', 64, 'Frame width')
cmd:option('--crit', 'gdll1', 'loss : Abs, MSE, GDL, gdll1, SpatialClassNLL')
cmd:option('--saveFreq', 40, 'saving after this number of iterations')
opt = cmd:parse(arg)
print('Running with training options:', opt)

opt.modelConfig = {learningRate = opt.lr}
opt.nscales = 2
opt.segm = 1
torch.setnumthreads(1)
torch.manualSeed(1)
cutorch.setDevice(opt.devid)

if paths.filep(opt.saveDir) or paths.dirp(opt.saveDir) then
    os.execute('rm -r ' .. opt.saveDir .. '.bkp')
    os.execute('mv ' .. opt.saveDir .. ' ' .. opt.saveDir .. '.bkp')
    print('Copied existing '..opt.saveDir..' into '..opt.saveDir..'.bkp')
end
os.execute('mkdir -p ' .. opt.saveDir)

opt.nChannels = nclasses
opt.nclasses = nclasses

local trainBatchList = getNBatches(opt.dataDir,'train')
local valBatchList = getNBatches(opt.dataDir,'val')
if opt.nItersTest>#valBatchList then
  print('Only '..#valBatchList..' test batches available')
end
if opt.nItersTest==0 then opt.nItersTest=#valBatchList end
print('Training on '..#trainBatchList..' batches')
print('Validation on '..#valBatchList..' batches')

-- creating the model ----------------------------------------------------------
paths.dofile("utils/model.lua")
local model = getPyrModel(opt)
local preprocessInput = getPyrPreprocessor(opt)
local preprocessTarget = getPyrPreprocessor(opt)
local modelW, modelDW = model:getParameters()

-- defining the loss -----------------------------------------------------------
local lossPixel = nn.ParallelCriterion()
for i = 1, opt.nscales do
  if not opt.crit=='gdll1' then
    lossPixel:add(nn[opt.crit .. 'Criterion']())
  else
    local crit = nn.MultiCriterion()
    lossPixel:add(crit:add(nn.AbsCriterion(),1):add(nn.GDLCriterion(1)))
  end
end
lossPixel:cuda()

-- shortcuts -------------------------------------------------------------------
local ob = opt.batchSize
local tf = opt.nTargetFrames
local inpf = opt.nInputFrames
local hi, wi = opt.hInput, opt.wInput
local ch = opt.nChannels
local confusion = tnt.SemSegmMeter{classes = classes}

-- basic routines --------------------------------------------------------------
function getBatch(set, iIter)
  if set == 'train' then iIter = math.random(1, #trainBatchList) end
  local sample = torch.load(paths.concat(opt.dataDir, set, 'batch_'..iIter..'.t7'))
  local segmInputE, segmTargetE
  if set == 'train' then
    segmInputE = sample.R8s[{{},{1,inpf}}]:cuda()
    segmTargetE = sample.R8s[{{},{inpf+1,inpf+tf}}]:cuda()
  else
    local RGBs = sample.RGBs
    local h = math.random(1, oh-hi)
    local w = math.random(1, ow-wi)
    segmInputE = sample.R8s[{{},{1,inpf},{},{h, h+hi-1},{w,w+wi-1}}]:cuda()
    segmTargetE = sample.R8s[{{},{inpf+1, inpf+tf},{},{h, h+hi-1},{w,w+wi-1}}]:cuda()
  end
  segmTargetE:resize(ob, tf*ch, wi, hi)
  segmInputE:resize(ob, inpf*ch, wi, hi)
  return preprocessInput:forward(segmInputE), preprocessTarget:forward(segmTargetE)
end

function training(iIter)
    local input, target = getBatch('train', iIter)
    local err = 0
    local feval = function(x)
        assert(x == modelW)
        model:zeroGradParameters()
        local output = model:forward(input)
        local l2err = lossPixel:forward(output, target)
        derr_dpred = lossPixel:backward(output, target)
        model:backward(input,derr_dpred)
        err = l2err
        return l2err, modelDW
    end
    optim.sgd(feval, modelW, opt.modelConfig, modelState)
    return err
end

function testing(iEpoch)
  confusion:reset()
  for j = 1, opt.nItersTest do
    xlua.progress(j, opt.nItersTest)
    local input, target = getBatch('val', j)
    local pred = model:forward(input)

    local spredF = squeeze_segm_map(pred[opt.nscales]:clone(),opt.nclasses,ob,hi,wi)
    spredF = spredF:view(ob, tf, 1, hi, wi)
    local stargetF = squeeze_segm_map(target[opt.nscales]:clone(),opt.nclasses,ob,hi,wi)
    stargetF = stargetF:view(ob, tf, 1, hi, wi)
    for i = 1,ob do
      confusion:add(spredF[i][1][1], stargetF[i][1][1])
    end
  end
end

-- main training loop ----------------------------------------------------------
for iEpoch = 1, opt.nEpoches do
  local sumGenErr = 0
  for iIter = 1, opt.nIters do
    xlua.progress(iIter, opt.nIters)
    sumGenErr = sumGenErr + training(iIter)
  end
  local avgGenErr = sumGenErr / (opt.nIters * opt.batchSize)
  print("Epoch "..iEpoch..'/'..opt.nEpoches.."; Generator error = ".. avgGenErr)
  torch.save(paths.concat(opt.saveDir,'model.t7'),{generator=model, opt=opt})
  if iEpoch % opt.saveFreq == 0 then
    print('Saving the model...')
    model:clearState()
    torch.save(paths.concat(opt.saveDir, 'model_'..iEpoch..'epochs.t7'),
      {generator=model, opt=opt})
    collectgarbage()
  end
  testing(iEpoch)
  print('Validation [IoU] '..confusion:value('map'))
end

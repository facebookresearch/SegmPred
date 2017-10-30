--------------------------------------------------------------------------------
-- Testing a multiscale convnet to predict next frame from some previous images
-- and semantic segmentations
-- Written by Camille Couprie, Pauline Luc, Natalia Neverova
--------------------------------------------------------------------------------
-- Copyright 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

require 'torch'
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

-- set options -----------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:option('--modelID', 'AR_dil_ft', 'AR_dil_ft or AR')
cmd:option('--nRecFrames', 0, 'N of recurrent frames (0 short term, 2 midterm)')
cmd:option('--save', false, 'saving generated predictions')
cmd:option('--saveDir', 'results', 'directory for exporting predictions')
cmd:option('--dataDir', 'Data/', 'directory with the dataset')
cmd:option('--nseq', 0, 'amount of test sequences (0 - all in Data/val/)')
cmd:option('--delaygif', 50, 'speed of the generated animation')
local opttest = cmd:parse(arg)
print('Running with test options:', opttest)

-- load and set model parameters -----------------------------------------------
modelPaths = {}
modelPaths['AR_dil_ft'] = 'trained_models/S2S_dil_AR_ft_cpu.t7'
modelPaths['AR'] = 'trained_models/S2S_AR_cpu.t7'
if modelPaths[opttest.modelID]==nil then
  modelPaths[opttest.modelID] = opttest.modelID
end
print('Loading a pretrained model from ' .. modelPaths[opttest.modelID])
assert(paths.filep(modelPaths[opttest.modelID]), "Pretrained model not found")
local loaded = torch.load(modelPaths[opttest.modelID])
opt = loaded.opt
for k,v in pairs(opttest) do opt[k] = opttest[k] end
opt.nTargetFrames = 1
opt.datasetFrameRate = 3
print('Input frames: ' .. opt.nInputFrames)
print('Target frames: ' .. opt.nTargetFrames)
print('Recurrent steps ' .. opt.nRecFrames)

-- load and check the data -----------------------------------------------------
local batchList = getNBatches(opt.dataDir,'val')
if opt.nseq == 0 then opt.nseq = #batchList end
assert(opt.nseq<=#batchList and opt.nseq>0,
  "Found "..#batchList.." batches out of "..opt.nseq)
print('Number of test sequences: ' .. opt.nseq)

-- create directories ----------------------------------------------------------
if opt.save and paths.filep(opt.saveDir) or paths.dirp(opt.saveDir) then
  if paths.dirp(opt.saveDir .. '.bkp') then
    os.execute('sudo rm -R ' .. opt.saveDir .. '.bkp')
  end
  os.execute('sudo mv ' .. opt.saveDir .. ' ' .. opt.saveDir .. '.bkp')
  print('Copied existing '..opt.saveDir..' into '..opt.saveDir..'.bkp')
end

-- load the model --------------------------------------------------------------
paths.dofile('utils/model.lua')
local model = loaded.generator:cuda()
local preprocessInput = getPyrPreprocessor(opt)

-- allocate variables ----------------------------------------------------------
local ob = opt.batchSize
local tf = opt.nTargetFrames
local inpf = opt.nInputFrames
local rf = opt.nRecFrames

local confusion = tnt.SemSegmMeter{classes = classes, skipClass = 20}
local segmInputE = torch.CudaTensor(ob, inpf, nclasses, oh, ow)
local predS = torch.CudaTensor(ob, rf + 1, nclasses * tf, oh, ow):fill(0)
local sinputF, spredF, inputF

for jt = 1,#batchList do -- iterating over batches
  xlua.progress(jt, opt.nseq)
  local frames, segmE = getBatch(batchList[jt]) -- loading new batch
  local inputF = frames[{{},{1,inpf}}]:clone():view(ob, inpf, oc, oh, ow)
  local framesTarget = frames[{{},{inpf + 1, inpf + tf + rf}}]:clone()

  local sinputF = squeeze_segm_map(segmE[{{},{1,inpf}}]:clone(),nclasses,ob,oh,ow)
  local segmTargetE = segmE[{{},{inpf + 1, inpf + tf + rf}}]:clone()
  segmTargetE:resize(ob, (tf + rf) * nclasses, oh, ow)
  local segmTarget = squeeze_segm_map(segmTargetE, nclasses, ob, oh, ow):cuda()

  for k = 1, rf+1 do -- autoregressive inference
    if k<inpf then segmInputE[{{}, {1,inpf-k+1}}] = segmE[{{},{k,inpf}}] end
    if k>1 then
      segmInputE[{{},{math.max(inpf-k+2,1),inpf}}] = predS[{{},{math.max(k-inpf-1,1),k-1}}]
    end
    local input = preprocessInput:forward(resize_batch(segmInputE):cuda())
    if #input<2 then input[2] = input[1] end
    local pred = model:forward(input)
    if type(pred)=='table' then pred = pred[opt.nscales] end
    predS[{{},k}] = pred[{{},{1,nclasses}}]:clone()
  end

  -- assess quality  -----------------------------------------------------------
  local targetF = framesTarget:view(ob, tf+rf, oc, oh, ow)
  local stargetF = segmTarget:view(ob, tf+rf, 1, oh, ow)
  local spredF = squeeze_segm_map(predS:double(),nclasses,ob,oh,ow)
  spredF = spredF:view(ob, tf+rf, 1, oh, ow)
  for i = 1,ob do confusion:add(spredF[i][tf+rf][1],stargetF[i][tf+rf][1]) end

  -- dump predictions ----------------------------------------------------------
  if opt.save then
    local filename_out = opt.saveDir .. '/' .. jt
    os.execute('sudo mkdir -p ' .. filename_out)
    os.execute('sudo chmod 777 ' .. filename_out)
    display_segm(sinputF, 0, (filename_out..'/spred'), inputF, colormap)
    display_segm(spredF, inpf, (filename_out..'/spred'), targetF, colormap, true)
    os.execute('sudo convert $(for ((a=1; a<='..(inpf + tf + rf)..
        '; a++)); do printf -- "-delay '..opt.delaygif..' '..filename_out..
        '/spred_%s.png " $a; done;) '..filename_out..'/results.gif')
  end
end

print('========== PERFORMANCE: ALL CLASSES ==========')
print('IoU SEG ' ..' = ' ..confusion:value('map')..' '
  ..'; per class acc. SEG = '..confusion:value('pc')
  ..'; per pixel acc. SEG = '..confusion:value('pp'))
print('========== PERFORMANCE: MOVING OBJECTS ==========')
print('IoU SEG ' ..' = ' ..confusion:valueOver('iou', movingObjects)..' '
  ..'; per class acc. SEG = '..confusion:valueOver('pc', movingObjects)
  ..'; per pixel acc. SEG = '..confusion:valueOver('pp', movingObjects))

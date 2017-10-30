--------------------------------------------------------------------------------
-- Model definition
-- Written by Camille Couprie, Michael Mathieu, Pauline Luc, Natalia Neverova
--------------------------------------------------------------------------------
-- Copyright 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

require('nngraph')
require('cunn')
require('cudnn')
require('nnx')

local mod_size=4

local modelStruct
if opt.nscales==2 then
  modelStruct = {
       [2] = {
            {'convp', 3, 32*mod_size},
            {'convp', 3, 64*mod_size},
            {'feature'},
            {'convp', 3, 32*mod_size},
            {'convp', 3, nil}},
        [1] = {
            {'convp', 5, 32*mod_size},
            {'convp', 3, 64*mod_size},
            {'feature'},
            {'convp', 3, 32*mod_size},
            {'convp', 5, nil}}}
elseif opt.nscales==1 then
  modelStruct = {
       [1] = {
            {'convp', 3, 32*mod_size},
            {'convp', 3, 64*mod_size},
            {'feature'},
            {'convp', 3, 32*mod_size},
            {'convp', 3, nil}}}
end

scaleList = {}
for k, v in pairs(modelStruct) do
    scaleList[1+#scaleList] = k
end

table.sort(scaleList, function(a,b) return b<a end)

local function getConvNet(struct,nChannels,h,w,nOutputChannels,nOutputElements)
    local isInFCMode, nElements = false, nil
    local input = nn.Identity()()
    local x = nn.Identity()(input)
    local feature = nil

    for i = 1, #struct do
        if struct[i][1] == 'conv' then
            local nOutputs = struct[i][3] or nOutputChannels
            assert(not isInFCMode) -- no convolutions after FC
            assert(nOutputs ~= nil) -- no nil if nOutputChannels is nil
            assert((struct[i][3] ~= nil) or (i == #struct))
            -- no nil except in last layer
            x = cudnn.SpatialConvolution(nChannels, nOutputs,
                                         struct[i][2], struct[i][2],
                                         struct[i][4], struct[i][4]):cuda()(x)
            if struct[i][4] ~= nil then
                nChannels, h, w = nOutputs,
                 math.floor((h - struct[i][2])/struct[i][4]) + 1,
                 math.floor((w - struct[i][2])/struct[i][4]) + 1
            else
                nChannels, h, w = nOutputs, h-struct[i][2]+1, w-struct[i][2]+1
            end
        elseif struct[i][1] == 'convp' then
            local nOutputs = struct[i][3] or nOutputChannels
            assert(struct[i][2] % 2 == 1) -- no even kernel sizes when padding!
            assert(not isInFCMode) -- no convolutions after FC
            assert(nOutputs ~= nil) -- no nil if nOutputChannels is nil
            assert((struct[i][3] ~= nil) or (i == #struct))
            -- no nil except in last layer
            x = cudnn.SpatialConvolution(nChannels, nOutputs,
                                         struct[i][2], struct[i][2],
                                         1, 1, (struct[i][2]-1)/2,
                                         (struct[i][2]-1)/2):cuda()(x)
            nChannels = nOutputs
        elseif struct[i][1] == 'convd' then
            local nOutputs = struct[i][3] or nOutputChannels
            assert(struct[i][2] % 2 == 1) -- no even kernel sizes when padding!
            assert(not isInFCMode) -- no convolutions after FC
            assert(nOutputs ~= nil) -- no nil if nOutputChannels is nil
            assert((struct[i][3] ~= nil) or (i == #struct))
            -- no nil except in last layer
            x = nn.SpatialDilatedConvolution(nChannels, nOutputs,
                                         struct[i][2], struct[i][2],
                                         1, 1, (struct[i][2]-1),
                                         (struct[i][2]-1), dilationW,
                                         dilationH):cuda()(x)
            nChannels = nOutputs
        elseif struct[i][1] == 'maxpool' then
            assert(not isInFCMode) -- no pooling after FC
            x = cudnn.SpatialMaxPooling(struct[i][2], struct[i][2],
                                        struct[i][3], struct[i][3])(x)
            h = math.floor((h - struct[i][2])/struct[i][3] + 1)
            w = math.floor((w - struct[i][2])/struct[i][3] + 1)
        elseif struct[i][1] == 'fc' then
            local nOutputs = struct[i][2] or nOutputElements
            assert(nOutputs ~= nil) -- no nil if nOutputElements is nil
            assert((struct[i][2] ~= nil) or (i == #struct))
            -- no nil except in last layer
            if not isInFCMode then
                nElements = h*w*nChannels
                x = nn.View(nElements):setNumInputDims(3)(x)
                isInFCMode = true
            end
            x = nn.Linear(nElements, nOutputs):cuda()(x)
            nElements = nOutputs
        elseif struct[i][1] == 'feature' then
            assert(feature == nil) -- only one feature layer
            feature = x
        elseif struct[i][1] == 'spatialbatchnorm' then
            x = nn.SpatialBatchNormalization(nChannels)(x)
        else
            error('Unknown network element ' .. struct[i][1])
        end
        if i ~= #struct then
            x = nn.ReLU()(x)
        end
    end

    local net = nn.gModule({input}, {x, feature})

    if isInFCMode then
        return net, nElements
    else
        return net, nChannels, h, w
    end
end


function getPyrModel(opt, in_modules)
    -- assume input/target is between -1 and 1
    local out_modules = {}
    local function getPred(imagesScaled,inputGuess,scale,scaleRatio,in_module,seg)
        local ws, hs = opt.wInput / scale, opt.hInput / scale
        local guessScaled, x = nil, nil
        local nChannels = opt.nChannels
        if seg == 2 then --segmentations
          nChannels = opt.nclasses
        end
        local nChannelsT = 3
        local nInputChannels = opt.nInputFrames*nChannels
        local nOutputChannels = opt.nTargetFrames*nChannels
        if opt.segm == 2 then
            nInputChannels = opt.nInputFrames*(nChannels + opt.nclasses)
            nOutputChannels = opt.nTargetFrames*nChannels
        elseif opt.segm == 3 then
            nChannels = 3
            nInputChannels = opt.nInputFrames*(nChannels + opt.nclasses)
            nOutputChannels = opt.nTargetFrames* opt.nclasses
            nChannelsT =  opt.nclasses
        elseif opt.segm == 4 then
            nInputChannels = opt.nInputFrames*nChannels + opt.nInputFrames
                *opt.nclasses
            nOutputChannels = opt.nTargetFrames* (opt.nclasses+3)
            nChannelsT =  nChannels+opt.nclasses
        elseif opt.segm == 1 then
            nChannelsT =  nChannels
        end
        if inputGuess ~= nil then
            guessScaled = nn.SpatialUpSamplingNearest(scaleRatio)(inputGuess)
            nInputChannels = nInputChannels +opt.nTargetFrames*nChannelsT
	        x = nn.JoinTable(2){imagesScaled, guessScaled}
	    else
	        x = imagesScaled
        end
      	local mod = in_module
      	if not mod then
      	    mod = getConvNet(modelStruct[scale], nInputChannels, hs, ws,
                nOutputChannels)
      	end
      	mod = mod:cuda()
        x = mod({x})
      	out_modules[scale] = mod
        local x, features = x:split(2)
        if inputGuess ~= nil then
            x = nn.CAddTable(){x, guessScaled}
        end
        if opt.segm == 0 or opt.segm ==2 then
            x = nn.Tanh()(x)
        end
        return x, features
    end

    local inputImages = nn.Identity()()
    local pred, features = {}, {}
    for i = 1, #scaleList do
      local scale = scaleList[i]
      local mod = nil
      if in_modules then
    	     mod = in_modules[scale]
	  end
      pred[i], features[i] = getPred(nn.SelectTable(i)(inputImages), pred[i-1],
             scale, (i == 1) or (scaleList[i-1] / scale), mod, 1)

    end
    pred = nn.Identity()(pred)
    features = nn.Identity()(features)
    pred = nn.SelectTable(1){pred, features}
    local model = nn.gModule({inputImages}, {pred})
    model = model:cuda()
    return model, out_modules
end


function getPyrPreprocessor(opt, dataset)
    local net = nn.ConcatTable()
    for i = 1, #scaleList do
        local net2 = nn.Sequential()
        net:add(net2)

        net2:add(nn.FunctionWrapper(
                     function(self) end,
                     function(self, input)
                         return input:view(input:size(1),
                                            -1, input:size(input:dim()-1),
                                           input:size(input:dim()))
                     end,
                     function(self, input, gradOutput)
                         return gradOutput:viewAs(input)
                     end))

        local scale = scaleList[i]
        net2:add(nn.SpatialAveragePooling(scale, scale, scale, scale))
    end
    net:cuda()
    return net
end


--------------------------------------------------------------------------------
GDL, gdlparent = torch.class('nn.GDLCriterion', 'nn.Criterion')

function GDL:__init(alpha)
    gdlparent:__init(self)
    self.alpha = alpha or 1
    assert(alpha == 1) --for now
    local Y = nn.Identity()()
    local Yhat = nn.Identity()()
    local Yi1 = nn.SpatialZeroPadding(0,0,0,-1)(Y)
    local Yi2 = nn.SpatialZeroPadding(0,0,-1,0)(Y)
    local Yj1 = nn.SpatialZeroPadding(0,-1,0,0)(Y)
    local Yj2 = nn.SpatialZeroPadding(-1,0,0,0)(Y)
    local Yhati1 = nn.SpatialZeroPadding(0,0,0,-1)(Yhat)
    local Yhati2 = nn.SpatialZeroPadding(0,0,-1,0)(Yhat)
    local Yhatj1 = nn.SpatialZeroPadding(0,-1,0,0)(Yhat)
    local Yhatj2 = nn.SpatialZeroPadding(-1,0,0,0)(Yhat)
    local term1 = nn.Abs()(nn.CSubTable(){Yi2, Yi1})
    local term2 = nn.Abs()(nn.CSubTable(){Yhati2,  Yhati1})
    local term3 = nn.Abs()(nn.CSubTable(){Yj2, Yj1})
    local term4 = nn.Abs()(nn.CSubTable(){Yhatj2, Yhatj1})
    local term12 = nn.CSubTable(){term1, term2}
    local term34 = nn.CSubTable(){term3, term4}
    self.net = nn.gModule({Yhat, Y}, {term12, term34})
    self.net:cuda()
    self.crit = nn.ParallelCriterion()
    self.crit:add(nn.AbsCriterion())
    self.crit:add(nn.AbsCriterion())
    self.crit:cuda()
    self.target1 = torch.CudaTensor()
    self.target2 = torch.CudaTensor()
end


function GDL:updateOutput(input, target)
    self.netoutput = self.net:updateOutput{input, target}
    self.target1:resizeAs(self.netoutput[1]):zero()
    self.target2:resizeAs(self.netoutput[2]):zero()
    self.target = {self.target1, self.target2}
    self.loss = self.crit:updateOutput(self.netoutput, self.target)
    return self.loss
end


function GDL:updateGradInput(input, target)
    local gradInput =
        self.crit:updateGradInput(self.netoutput, self.target)
    self.gradInput =
        self.net:updateGradInput({input, target}, gradInput)[1]
    return self.gradInput
end

--------------------------------------------------------------------------------
-- A number of useful pre/post-processing and visualizalion functions
-- Written by Camille Couprie, Pauline Luc, Natalia Neverova
--------------------------------------------------------------------------------
-- Copyright 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

function squeeze_segm_map(segm_map,nclasses,batchSize,oh,ow)
  local segm_map_
  if segm_map:dim() == 4 then
    segm_map = segm_map:view(segm_map:size(1), segm_map:size(2)/nclasses,
      nclasses, segm_map:size(3), segm_map:size(4)):double()
    segm_map_ = torch.Tensor(batchSize, segm_map:size(2), oh, ow)
    for l=1,segm_map:size(1) do
      for kk=1,segm_map:size(2) do
          _,segm_map_[{l,kk}] = torch.max(segm_map[{l,kk}],1)
      end
    end
   segm_map_ = segm_map_:view(batchSize,segm_map:size(2), oh, ow)
  elseif segm_map:dim() == 5 then
    segm_map = segm_map:view(segm_map:size(1), segm_map:size(2),
    segm_map:size(3)/nclasses,nclasses, segm_map:size(4),
    segm_map:size(5)):double()
    segm_map_ = torch.Tensor(batchSize, segm_map:size(2), segm_map:size(3), oh, ow)
    for l=1,segm_map:size(1) do
      for m=1,segm_map:size(2) do
        for kk=1,segm_map:size(3) do
            _,segm_map_[{l,m, kk}] = torch.max(segm_map[{l,m, kk}],1)
        end
      end
    end
    segm_map_=segm_map_:view(batchSize,segm_map:size(2),
      segm_map:size(3), oh, ow)
  end
  return segm_map_
end


function resize_batch(sInp)
  local iszs = sInp:size()
  return torch.reshape(sInp, iszs[1], iszs[2]*iszs[3], iszs[4], iszs[5])
end


function getNBatches(sourceDir, set)
  local batchList = {}
  local filedir = paths.concat(sourceDir, set)
  for file in paths.files(filedir) do
    if file:find('batch_') then
      table.insert(batchList, file)
    end
  end
  return batchList
end


function getBatch(batch)
  local sample = torch.load(paths.concat('Data', 'val', batch))
  local RGBs = sample.RGBs
  local frames = RGBs[{{},{}}]:cuda():mul(2/255):add(-1)
  local segmE = sample.R8s[{{},{}}]:cuda()
  return frames, segmE
end


function colorize(inp, colormap)
  local colorized = torch.zeros(3,inp:size(2),inp:size(3))
  for ii = 1,inp:size(2) do
    for jj = 1,inp:size(3) do
      colorized[{{},ii,jj}] = torch.Tensor(colormap[inp[1][ii][jj]])
    end
  end
  return colorized
end


function display_segm(segm,  nb, filename, img, colormap, framed)
  local dm = #segm:size()
  local ob,of,oh,ow = segm:size(1),segm:size(2),segm:size(dm-1),segm:size(dm)
  for n=1,of do
    for b = 1,ob do
      local colored = colorize(segm[b][n]:double(), colormap)
      local new_colored = torch.Tensor(3, oh, ow):fill(0)
      if framed then
        new_colored[1]:fill(1)
        new_colored[{{},{3,oh-2},{3,ow-2}}]=colored[{{},{3,oh-2},{3,ow-2}}]
      else
        new_colored = colored:clone()
      end
      if img ~= nil then
        local imgcopy = img:clone()
        local saved = new_colored:add(imgcopy[b][n]:add(1):div(2):double())
        image.save(filename..'_'..(n+nb)..'.png', saved )
      else
        image.save(filename..'_'..(n+nb)..'.png', new_colored)
      end
    end
  end
end


function display_imgs(todisp, img, nb, filename)
  for k = 1, img:size(2) do
    for b = 1, img:size(1) do
       todisp[1+#todisp] = img[b][k][{{1,3}}]
       if save then
         image.save(filename..'_'..(k+nb)..'.png',img[b][k][{{1,3}}])
       end
    end
  end
end

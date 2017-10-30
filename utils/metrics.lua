--------------------------------------------------------------------------------
-- Evaluation metrics
-- Written by Camille Couprie, Pauline Luc, Natalia Neverova
--------------------------------------------------------------------------------
-- Copyright 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

require 'optim'
local tnt = require 'torchnet'
local argcheck = require 'argcheck'
require 'sys'
local SemSegmMeter = torch.class('tnt.SemSegmMeter','tnt.Meter',tnt)


SemSegmMeter.__init = argcheck{
  noordered = true,
  {name='self', type='tnt.SemSegmMeter'},
  {name='classes', type='table'},
  {name='skipClass', type='number', opt=true},
  {name='movingobject', type='number', opt=true},
  call =
    function(self, classes, skipClass, movingobject)
      self.classes = classes
      self.conf = optim.ConfusionMatrix(#self.classes)
      if movingobject then
        self.movingobject = true
      end
      if skipClass then
        self.sc = skipClass
      end
      self:reset()
    end
}


SemSegmMeter.reset = argcheck{
    {name="self", type="tnt.SemSegmMeter"},
    call = function(self)
        self.conf:zero()
    end
}


SemSegmMeter.add = argcheck{
  {name="self", type="tnt.SemSegmMeter"},
  {name="output", type="torch.*Tensor"},
  {name="target", type="torch.*Tensor"},
  call =
    function(self, output, target)
      if output:dim()==2 and target:dim()==2 then
        output = output:reshape(1, output:size(1), output:size(2)):long()
        target = target:reshape(1, target:size(1), target:size(2))
      end

      target = target:squeeze()
      output = output:squeeze()

      if type(output) == 'number' then
        print(output)
        print(target)
        self.conf:add(output, target)
      else
        assert(output:nElement() == target:nElement(),
          'target and output do not match')

        local N = output:nElement()
        output, target = output:view(N):view(-1,1), target:view(N):view(-1,1)

        local C = #self.classes
        local Mout, Mtar
        if torch.type(output)== 'torch.CudaLongTensor' then
          Mout = torch.CudaTensor(N, C):zero()
          Mtar = torch.CudaTensor(N, C):zero()
        else
          Mout = torch.Tensor(N, C):type(output:type()):zero()
          Mtar = torch.Tensor(N, C):long():zero()
          target = target:long()
        end
        Mout:scatter(2, output, 1)
        Mtar:scatter(2, target, 1)
        Mtar = Mtar:transpose(1,2)
        -- multiply and add to mc
        local tmp
        -- Avoid transferring to GPU
        if torch.type(output)== 'torch.CudaLongTensor' then
          tmp = torch.CudaTensor(C,C):zero()
        else
          tmp = torch.Tensor(C, C):type(Mout:type()):zero()
        end
        tmp:mm(Mtar, Mout)
        if self.movingobject then
          for i=1, 11 do
            tmp[i]:fill(0)
          end
        end
        if self.sc then tmp[self.sc]:fill(0) end
        tmp = tmp:type(self.conf.mat:type())

        self.conf.mat:add(tmp:contiguous())
        local saveMC = optim.ConfusionMatrix(#self.classes)
        saveMC.mat = tmp:long()
        return saveMC
      end
    end
}


SemSegmMeter.value = argcheck{
   {name="self", type="tnt.SemSegmMeter"},
   {name="s", type="string"},
   call =
    function(self, s)
        if s == 'map' then
            self.conf:updateValids()
            return self.conf.averageUnionValid*100
        elseif s == 'pp' then
            self.conf:updateValids()
            return self.conf.totalValid*100
        elseif s == 'pc' then
            self.conf:updateValids()
            return self.conf.averageValid*100
        else
            error('Only map, pp and pc available.')
        end
    end
}


SemSegmMeter.valueOver = argcheck{
  {name="self", type = "tnt.SemSegmMeter"},
  {name="s", type = "string"},
  {name="classesToAverageOver", type = "table"},
  call =
  function(self, s, classesToAverageOver)
    local ctao = classesToAverageOver
    -- Make corresponding mask
    local mctao = torch.ByteTensor(#self.classes):zero()
    for _, c in ipairs(classesToAverageOver) do mctao[c] = 1 end
    local clval, D = self:values(s)
      local mn, ct = 0, 0
      for i = 1, clval:nElement() do
        if clval[i] == clval[i] then
          mn = mn + clval[i]
          ct = ct +1
        end
      end
      local notnan_mask = clval:eq(clval)
      local clval2 = clval[notnan_mask]
      local rem = clval[mctao]
      local notnan_mask = rem:eq(rem)
      rem = rem[notnan_mask]

      if s == 'pp' then
        remD = D[mctao]:sum()
        return 100*remD/rem:sum()
      end

      if rem:nElement() == 0 then return 0/0, 0
      else
        return rem:mean()*100, rem:nElement()
      end
  end
}


SemSegmMeter.values = argcheck{
   {name="self", type="tnt.SemSegmMeter"},
   {name="s", type="string"},
   call =
    function(self, s)
        if s == 'iou' then
            self.conf:updateValids()
            local unionvalids = self.conf.unionvalids:clone()
            -- Do not count IOU for absent classes, should this happen
            local nanval_pc = self.conf.valids:ne(self.conf.valids)
            unionvalids[nanval_pc] = unionvalids[nanval_pc]:fill(0/0)
            return unionvalids
        elseif s == 'pc' then
            self.conf:updateValids()
            return self.conf.valids:clone()
        elseif s == 'pp' then
          self.conf:updateValids()
          R = torch.sum(self.conf.mat,2):float():squeeze()
          D = torch.diag(self.conf.mat)
            return R,D
        else
            error('Only iou and pc available.')
        end
    end
}


SemSegmMeter.print = argcheck{
   {name="self", type="tnt.SemSegmMeter"},
   call =
        function(self)
          self.conf:updateValids()
          print(self.conf)
        end
}

return SemSegmMeter






















--------------------------------------------------------------------------------
-- Not needed anymore

-- If several untagged classes needed
-- SemSegmMeter.severalUntaggedAdd = argcheck{
--   {name="self", type="tnt.SemSegmMeter"},
--   {name="output", type="torch.*Tensor"},
--   {name="target", type="torch.*Tensor"},
--   call =
--     function(self, output, target)
--       assert(output:dim()==4)
--       assert(target:dim()==3)

--       -- Hard predictions
--       local _,hp = torch.max(output, 2)
--       hp = hp:squeeze(2)
--       print(hp:size())

--       -- Unfold predictions in contiguous memory
--       local szp, szt = hp:size(), target:size()
--       local uout, utar = hp:view(szp[1]*szp[2]*szp[3]):contiguous():float(),
--                           target:view(szt[1]*szt[2]*szt[3]):contiguous():float()
--       assert(uout:nElement()==utar:nElement())
--       -- Update
--       local N = uout:nElement()
--       local uoutPtr, utarPtr = uout:data(), utar:data()

--       for n=0,N-1 do
--         if not torch.any(torch.eq(self.sc, utarPtr[n])) then
--           self.conf:add(uoutPtr[n], utarPtr[n])
--         end
--       end
--     end
-- }

-- pcall(loadstring("sct = torch.FloatTensor({" .. skipClasses .."})"))
-- self.sc = sct:clone()
-- sct=nil
-- if self.sc:size(1)==1 then self.sc = self.sc[1] end -- to optimize

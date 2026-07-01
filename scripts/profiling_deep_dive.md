# MATLAB Profiling Deep Dive (New Single-Timepoint Run)

**Overall Pipeline Run Time:** 5.021 s

## `decon_lucy_function`
- **Total Time:** 2.157 s
- **Self Time:** 1.646 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 70 | `gpu_PSF = gpuArray(PSF);` | 1 | 1.071 | 49.7% |
| 91 | `H = decon_psf2otf(gpu_PSF ./ sum(gpu_PSF...` | 1 | 0.460 | 21.3% |
| 163 | `J_2 = max(Y.*real(ifftn(conj(H).*fftn(Re...` | 10 | 0.201 | 9.3% |
| 157 | `ReBlurred = max(real(ifftn(H.*fftn(Y))),...` | 10 | 0.199 | 9.2% |
| 153 | `Y = max(J_2 + lambda*(J_2 - J_3), 0);% p...` | 10 | 0.073 | 3.4% |

---

## `run_gpu_pipeline`
- **Total Time:** 5.02 s
- **Self Time:** 0.671 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 81 | `[deconvolved, err_mat, iter_run] = decon...` | 1 | 2.172 | 43.3% |
| 50 | `rawdata = readzarr(shm_path);` | 1 | 0.721 | 14.4% |
| 157 | `writezarr(block, shmZarrFn, 'bbox', [1, ...` | 8 | 0.343 | 6.8% |
| 164 | `system(cmd);` | 1 | 0.237 | 4.7% |

---

## `parallelReadZarr (MEX-file)`
- **Total Time:** 0.542 s
- **Self Time:** 0.542 s
- **Calls:** 1

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `...rray.imwarp>remapAndResampleInvertible3d`
- **Total Time:** 0.659 s
- **Self Time:** 0.52 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 502 | `tComp = Rout.TransformIntrinsicToWorld /...` | 1 | 0.431 | 65.4% |
| 517 | `B = arrayfun(@warpElemLinear, x, y, z);` | 1 | 0.214 | 32.5% |
| 507 | `fillValue = single(fillValues);` | 1 | 0.007 | 1.1% |
| 468 | `x = gpuArray.colon(1,1,Rout.ImageSize(2)...` | 1 | 0.002 | 0.3% |
| 524 | `B = cast(B,origClass);` | 1 | 0.002 | 0.3% |

---

## `deskewRotateFrame3D`
- **Total Time:** 1.098 s
- **Self Time:** 0.336 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 210 | `[volout] = imwarp(vol_1, affine3d(ds_S*(...` | 1 | 0.915 | 83.3% |
| 255 | `volout = gather(volout);` | 1 | 0.125 | 11.4% |
| 201 | `RA = imref3d(outSize, 1, 1, 1);` | 1 | 0.035 | 3.2% |
| 206 | `vol_1 = gpuArray(vol_1);` | 1 | 0.019 | 1.7% |
| 157 | `center = ([ny nxDs nz]+1)/2;` | 1 | 0.001 | 0.1% |

---

## `parallelWriteZarr (MEX-file)`
- **Total Time:** 0.288 s
- **Self Time:** 0.288 s
- **Calls:** 8

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `decon_psf2otf`
- **Total Time:** 0.323 s
- **Self Time:** 0.264 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 27 | `otf = fftn(psf);` | 1 | 0.196 | 60.7% |
| 20 | `psf     = padarray(psf, padSize, 'post')...` | 1 | 0.052 | 16.1% |
| 24 | `psf    = circshift(psf,-floor(psfSize/2)...` | 1 | 0.042 | 13.0% |
| 16 | `if ~all(psf(:)==0)` | 1 | 0.030 | 9.3% |
| 19 | `padSize = double(outSize - psfSize);` | 1 | 0.001 | 0.3% |

---

## `readzarr`
- **Total Time:** 0.679 s
- **Self Time:** 0.137 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 21 | `data = parallelReadZarr(filepath);` | 1 | 0.673 | 99.1% |
| 12 | `options.inputBbox (1, :) {mustBeNumeric}...` | 1 | 0.004 | 0.6% |
| 13 | `options.sparseData (1, :) {mustBeNumeric...` | 1 | 0.001 | 0.1% |

---

## `buildCUDADevice`
- **Total Time:** 0.089 s
- **Self Time:** 0.089 s
- **Calls:** 2

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `randGetSetDefaultStream`
- **Total Time:** 0.139 s
- **Self Time:** 0.061 s
- **Calls:** 1

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `readtiff`
- **Total Time:** 0.082 s
- **Self Time:** 0.059 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 18 | `data = parallelReadTiff(filepath);` | 1 | 0.081 | 98.8% |
| 10 | `filepath char` | 1 | 0.001 | 1.2% |

---

## `defaultGPUIndex`
- **Total Time:** 0.182 s
- **Self Time:** 0.041 s
- **Calls:** 1

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `writezarr`
- **Total Time:** 0.328 s
- **Self Time:** 0.039 s
- **Calls:** 8

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 77 | `parallelWriteZarr(filepath, data, 'bbox'...` | 8 | 0.312 | 95.1% |
| 15 | `options.shardSize (1, :) {mustBeNumeric}...` | 8 | 0.002 | 0.6% |
| 38 | `init_val = zeros(1, dtype);` | 8 | 0.001 | 0.3% |
| 47 | `blockSize = min(sz, blockSize);` | 8 | 0.001 | 0.3% |
| 113 | `if isunix && groupWrite && newFile` | 8 | 0.001 | 0.3% |

---

## `circshift`
- **Total Time:** 0.029 s
- **Self Time:** 0.029 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 81 | `b = a(idx{:});` | 1 | 0.027 | 93.1% |
| 76 | `idx{k} = mod((0:m-1)-double(rem(p(k),m))...` | 3 | 0.002 | 6.9% |

---

## `imwarpParseInputs`
- **Total Time:** 0.032 s
- **Self Time:** 0.024 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 7 | `parser.addRequired('InputImage',@validat...` | 1 | 0.016 | 50.0% |
| 16 | `parser.parse(varargin{:});` | 1 | 0.005 | 15.6% |
| 14 | `varargin = remapPartialParamNamesImwarp(...` | 1 | 0.004 | 12.5% |
| 8 | `parser.addRequired('GeometricTransform',...` | 1 | 0.002 | 6.2% |
| 11 | `parser.addParameter('SmoothEdges',false,...` | 1 | 0.002 | 6.2% |

---

## `NodeInfo>NodeInfo.NodeInfo`
- **Total Time:** 0.029 s
- **Self Time:** 0.024 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 75 | `machineToWorkerMapping = sortrows(machin...` | 1 | 0.021 | 72.4% |
| 82 | `myRange = cell2mat(machineToWorkerMappin...` | 1 | 0.005 | 17.2% |
| 65 | `machineToWorkerMapping = mediator.Machin...` | 1 | 0.001 | 3.4% |
| 83 | `obj.NumNodeWorkers = numel(myRange);` | 1 | 0.001 | 3.4% |

---

## `parallelReadTiff (MEX-file)`
- **Total Time:** 0.023 s
- **Self Time:** 0.023 s
- **Calls:** 1

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `gpuArray.imwarp`
- **Total Time:** 0.745 s
- **Self Time:** 0.021 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 219 | `outputImage = remapPointsAndResample(par...` | 1 | 0.660 | 88.6% |
| 175 | `parsedInputs = images.geotrans.internal....` | 1 | 0.038 | 5.1% |
| 173 | `[R_A, varargin] = images.geotrans.intern...` | 1 | 0.036 | 4.8% |
| 192 | `images.geotrans.internal.checkImageAgree...` | 1 | 0.003 | 0.4% |
| 207 | `images.internal.checkFillValues(fillValu...` | 1 | 0.003 | 0.4% |

---

## `...eSpatialReferencingObjects>validateTform`
- **Total Time:** 0.028 s
- **Self Time:** 0.02 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 30 | `validateattributes(t,{'images.geotrans.i...` | 1 | 0.027 | 96.4% |

---

## `sortDevicesByComputeMode`
- **Total Time:** 0.108 s
- **Self Time:** 0.013 s
- **Calls:** 1

*No specific line-level hotspots reported or time spent entirely in built-ins.*

---

## `unique>uniqueR2012a`
- **Total Time:** 0.013 s
- **Self Time:** 0.013 s
- **Calls:** 8

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 242 | `c = sortA(groupsSortA);         % Create...` | 8 | 0.003 | 23.1% |
| 183 | `rowvec = isrow(a);` | 8 | 0.002 | 15.4% |
| 222 | `groupsSortA = sortA(1:numelA-1) ~= sortA...` | 8 | 0.002 | 15.4% |
| 265 | `indC(indSortA) = indC;                  ...` | 4 | 0.001 | 7.7% |
| 285 | `end` | 4 | 0.001 | 7.7% |

---

## `unique`
- **Total Time:** 0.025 s
- **Self Time:** 0.012 s
- **Calls:** 8

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 162 | `[varargout{1:nlhs}] = uniqueR2012a(varar...` | 8 | 0.015 | 60.0% |
| 134 | `if flaginds(4) && flaginds(5)` | 8 | 0.002 | 8.0% |
| 119 | `foundflag = startsWith(flagvals,flag,'Ig...` | 8 | 0.001 | 4.0% |
| 150 | `firstOccurrence = ( useR2012a && ~logica...` | 8 | 0.001 | 4.0% |
| 153 | `if flaginds(4) || flaginds(5) % 'stable'...` | 8 | 0.001 | 4.0% |

---

## `gpuArray.padarray`
- **Total Time:** 0.023 s
- **Self Time:** 0.012 s
- **Calls:** 1

### Most Expensive Lines:
| Line | Code | Calls | Time (s) | % Time |
|---|---|---|---|---|
| 68 | `b = padarray_algo(a, padSize, method, pa...` | 1 | 0.009 | 39.1% |
| 65 | `args = matlab.images.internal.stringToCh...` | 1 | 0.006 | 26.1% |
| 66 | `[a, method, padSize, padVal, direction] ...` | 1 | 0.006 | 26.1% |

---

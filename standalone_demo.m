clear, clc;

fprintf('Standalone Microscopy pipeline based on PetaKit5D Demo...\n\n');

% Add the PetaKit5D software to the path
addpath(genpath('/cm/shared/apps_local/petakit5d/'));

% Data path: Pointing to the CROP directory since the raw data is dual-view
dataPath = '/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0/CROP/';

% general parameters
xyPixelSize = 0.136;
dz = 0.3;
Reverse = true; % Demo uses true

% Channels in the CROP directory
ChannelPatterns = {'C00_T000', 'C01_T000', 'C02_T000', 'C03_T000'};

Save16bit = true;
Overwrite = true;
Streaming = false;
parseCluster = false;
cpusPerTask = 24;
configFile = '';
mccMode = false;

Deskew = true;
Rotate = true;
Stitch = false;

DSRCombined = true; % Demo uses true

FFCorrection = false;
LowerLimit = 0.4;
constOffset = 1;
FFImagePaths = {'', '', '', ''};
BackgroundPaths = {'', '', '', ''};

ImageListFullpaths = '';
axisOrder = '-x,y,z';

minModifyTime = 1;
maxModifyTime = 10;
maxWaitLoopNum = 10;

% IMPORTANT: If your CROP images are vertically oriented (because they were rotated 90 deg during cropping), 
% then the scan axis is NO LONGER on the X-axis (the PetaKit default). It is on the Y-axis. 
% To make PetaKit shear the Y-axis, you MUST set inputAxisOrder and outputAxisOrder to 'xyz'.
% If your CROP images are horizontally oriented (NOT rotated during cropping), leave these as 'yxz'.
inputAxisOrder = 'yxz'; 
outputAxisOrder = 'yxz';

XR_microscopeAutomaticProcessing(dataPath, 'xyPixelSize', xyPixelSize, 'dz', dz,  ...
    'Reverse', Reverse, 'ChannelPatterns', ChannelPatterns, 'Save16bit', Save16bit, ...
    'Overwrite', Overwrite, 'Streaming', Streaming, 'Deskew', Deskew, 'Rotate', Rotate, ...
    'Stitch', Stitch, 'ImageListFullpaths', ImageListFullpaths, 'axisOrder', axisOrder, ...
    'FFCorrection', FFCorrection, 'LowerLimit', LowerLimit, 'constOffset', constOffset, ...
    'FFImagePaths', FFImagePaths, 'BackgroundPaths', BackgroundPaths,'minModifyTime', minModifyTime, ...
    'maxModifyTime', maxModifyTime, 'maxWaitLoopNum', maxWaitLoopNum, 'parseCluster', parseCluster, ...
    'cpusPerTask', cpusPerTask, 'configFile', configFile, 'mccMode', mccMode, ...
    'inputAxisOrder', inputAxisOrder, 'outputAxisOrder', outputAxisOrder);

% Set the path to your master file
fileName = '/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260304_YGbeads_30PDMS_AH/hDF_4/hDF_4_MMStack_Pos0.ome.tif';

% 1. Open the TIFF file and read the ImageDescription tag
t = Tiff(fileName, 'r');
omeXML = t.getTag('ImageDescription');
t.close();

% 2. Save the raw XML to a local file so you can view it easily
outputXML = 'extracted_metadata.xml';
fid = fopen(outputXML, 'w');
fprintf(fid, '%s', omeXML);
fclose(fid);
fprintf('Full OME-XML saved to: %s\n\n', outputXML);

% 3. Quick parse to show you the critical ordering info in the console
dimOrder = regexp(omeXML, 'DimensionOrder="([^"]+)"', 'tokens', 'once');
if ~isempty(dimOrder)
    fprintf('Dimension Order: %s\n', dimOrder{1});
end

fprintf('\nChannels in exact order embedded in the TIFF:\n');
channels = regexp(omeXML, '<Channel[^>]*Name="([^"]+)"', 'tokens');
for i = 1:length(channels)
    fprintf('  Index %d: %s\n', i-1, channels{i}{1});
end
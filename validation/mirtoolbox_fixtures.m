function mirtoolbox_fixtures(audio_dir, out_dir, cases_files, files)
%MIRTOOLBOX_FIXTURES Save MIRtoolbox 1.8.2 reference outputs for the py-mir-toolbox parity tests.
%   MIRTOOLBOX_FIXTURES(AUDIO_DIR, OUT_DIR) runs every call listed in the
%   *_cases.tsv files next to this file on four demo files from
%   AUDIO_DIR (the MIRToolboxDemos folder) and writes one
%   <stem>__<feature>.mat file per call and file into OUT_DIR.
%
%   MIRTOOLBOX_FIXTURES(AUDIO_DIR, OUT_DIR, CASES_FILES, FILES) uses other
%   lists of calls (a file name or a cell array of them) and/or a cell
%   array of other audio file names.
%
%   Each .mat file holds the raw data (element x frame x channel), the
%   element positions, frame positions, sampling rate, the peaks when
%   there are any (1-based, one row per peak), the call, and the MATLAB
%   and MIRtoolbox versions. For calls that return a structure (mirstat),
%   each numeric field F is saved as stat_F instead. Convert them for the
%   Python tests with validation/convert_matlab_fixtures.py.
%
%   Requires MIRtoolbox 1.8.2 on the MATLAB path. A call that fails is
%   written to OUT_DIR/failures.txt and skipped.
%
%   Example:
%       mirtoolbox_fixtures('MIRtoolbox1.8.2/MIRToolboxDemos', 'mirtoolbox_mat')

if nargin < 3 || isempty(cases_files)
    here = fileparts(mfilename('fullpath'));
    listing = dir(fullfile(here, '*_cases.tsv'));
    cases_files = fullfile(here, {listing.name});
end
if ischar(cases_files)
    cases_files = {cases_files};
end
if nargin < 4
    files = {'ragtime.wav', 'Cmaj.wav', 'pianoA4.wav', 'trumpet.wav'};
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
mirverbose(0);
mirwaitbar(0);

cases = cell(0, 2);
for k = 1:numel(cases_files)
    cases = [cases; read_cases(cases_files{k})]; %#ok<AGROW>
end
versions = toolbox_versions();
log_id = fopen(fullfile(out_dir, 'failures.txt'), 'w');
closer = onCleanup(@() fclose(log_id));

for i = 1:numel(files)
    f = fullfile(audio_dir, files{i});  % used by the eval'd calls
    [~, stem] = fileparts(files{i});
    for j = 1:size(cases, 1)
        feature = cases{j, 1};
        call = cases{j, 2};
        try
            result = eval(call);
            s = describe(result);
            s.feature = feature;
            s.call = call;
            s.audio = files{i};
            s.matlab_version = versions.matlab;
            s.mirtoolbox_version = versions.mirtoolbox;
            s.mirtoolbox_path = versions.path;
            s.generated = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
            save(fullfile(out_dir, [stem '__' feature '.mat']), '-struct', 's', '-v7');
            fprintf('%s: %s\n', files{i}, feature);
        catch err
            fprintf(log_id, '%s\t%s\t%s\n', files{i}, feature, err.message);
            fprintf(2, '%s: %s FAILED (%s)\n', files{i}, feature, err.message);
        end
    end
end
end


function cases = read_cases(path)
% Feature name and MIRtoolbox call on each non-comment line, tab-separated.
lines = strsplit(fileread(path), {'\r\n', '\n'});
cases = cell(0, 2);
for k = 1:numel(lines)
    line = strtrim(lines{k});
    if isempty(line) || line(1) == '#'
        continue
    end
    parts = strsplit(line, '\t');
    cases(end + 1, :) = {strtrim(parts{1}), strtrim(parts{2})}; %#ok<AGROW>
end
end


function v = toolbox_versions()
% MIRtoolbox 1.8.2's Contents.m still says 1.8.1, so the path is saved too.
v.matlab = version;
v.mirtoolbox = 'unknown';
v.path = fileparts(which('miraudio'));
info = ver;
match = find(~cellfun(@isempty, strfind(lower({info.Name}), 'mirtoolbox')), 1);
if ~isempty(match)
    v.mirtoolbox = info(match).Version;
end
end


function s = describe(x)
% Numeric content of a MIRtoolbox object, for the first file and segment.
if isstruct(x)
    s = describe_stat(x);
    return
end
s.data = unwrap(get(x, 'Data'));
s.pos = unwrap(get(x, 'Pos'));
s.framepos = unwrap(get(x, 'FramePos'));
s.sampling = unwrap(get(x, 'Sampling'));
s = add_peaks(s, x);
end


function s = describe_stat(x)
% mirstat output: one stat_<Field> array per numeric field.
s = struct();
names = fieldnames(x);
for k = 1:numel(names)
    value = unwrap(x.(names{k}));
    if isnumeric(value)
        s.(['stat_' names{k}]) = value;
    end
end
end


function v = unwrap(v)
% Descend {file}{segment}... cells down to the first numeric array.
while iscell(v) && ~isempty(v)
    v = v{1};
end
if iscell(v)
    v = [];
end
end


function s = add_peaks(s, x)
% One row per peak: position of its curve (row, frame, channel) and values.
index = peak_cells(x, 'PeakPos');
if isempty(index)
    return
end
fields = {'peak_pos', 'PeakPosUnit'; 'peak_precise_pos', 'PeakPrecisePos'; ...
          'peak_val', 'PeakVal'; 'peak_precise_val', 'PeakPreciseVal'};
values = cell(size(fields, 1), 1);
for m = 1:size(fields, 1)
    values{m} = peak_cells(x, fields{m, 2});
end
shape = [size(index), 1, 1];
shape = shape(1:3);
rows = [];
for k = 1:numel(index)
    [r, fr, ch] = ind2sub(shape, k);
    idx = index{k}(:);
    n = numel(idx);
    block = [repmat([r, fr, ch], n, 1), idx];
    for m = 1:numel(values)
        block = [block, curve_values(values{m}, k, n)]; %#ok<AGROW>
    end
    rows = [rows; block]; %#ok<AGROW>
end
names = [{'peak_row', 'peak_frame', 'peak_channel', 'peak_index'}, fields(:, 1)'];
for m = 1:numel(names)
    if isempty(rows)
        s.(names{m}) = zeros(0, 1);
    else
        s.(names{m}) = rows(:, m);
    end
end
end


function c = peak_cells(x, field)
% Per-curve peak cells of the first file and segment ({} when absent).
c = get(x, field);
for level = 1:2  % {file}{segment}
    if ~iscell(c) || isempty(c)
        c = {};
        return
    end
    c = c{1};
end
if ~iscell(c)
    c = {c};
end
end


function v = curve_values(cells, k, n)
% Values of curve k as a column of length n (NaN when MIRtoolbox has none).
v = nan(n, 1);
if numel(cells) >= k && numel(cells{k}) == n
    v = cells{k}(:);
end
end

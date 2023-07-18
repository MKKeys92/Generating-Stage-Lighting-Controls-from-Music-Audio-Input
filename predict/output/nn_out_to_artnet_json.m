function [] = nn_out_to_artnet_json(nn_json)

%nn_out_to_artnet_json Summary of this function goes here
%   Detailed explanation goes here

% open nn JSON
fid = fopen(nn_json);
raw = fread(fid,inf);
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);
lighting = struct2cell(val);
lightingstruct = lighting(2,1);
lightingstructinner = cell2mat(lightingstruct);

% use 1800 values
lightingstructinner = lightingstructinner(1:1800,1:75);

% bound it
lightingstructinner = bound(lightingstructinner,0,1);


% open empty artnet JSON
fid = fopen('empty_artnet.json');
raw = fread(fid,inf);
str = char(raw'); 
fclose(fid); 
val_empty_artnet = jsondecode(str);
empty_artnet = struct2cell(val_empty_artnet);
empty_artnet_struct = empty_artnet(2,1);
empty_artnet_structinner = cell2mat(empty_artnet_struct);





% extract features and put them back into 1024 parameter space
% artnet from NN features
artnet_nn_features = artnet_from_NN_features_v1(empty_artnet_structinner, lightingstructinner);

artnet01_nn_features = artnet_nn_features(:,1:512);
artnet02_nn_features = artnet_nn_features(:,513:1024);

% make struct vor JSON encoding
val_nn_artnet = struct('id','nn_artnet','lighting_array01',artnet01_nn_features,'lighting_array02',artnet02_nn_features);
encodedJSON = jsonencode(val_nn_artnet);


% write artnet JSON

% use if you want to write the JSON somewhere else
currentFolder = pwd;

% change for dynamic / static filename

filename_temp = [char('artnet_1024_'),nn_json];
% filename_temp = (filename_output_json);
    
% cd(path_to_write);
fid = fopen(filename_temp,'w');
fprintf(fid, encodedJSON);
fclose('all'); 

% use if you want to write the JSON somewhere else
cd(currentFolder);


end




function y = bound(x,bl,bu)
  % return bounded value clipped between bl and bu
  y=min(max(x,bl),bu);
end







function [artnet_nn_features] = artnet_from_NN_features_v1(artnet_rec,nn_features)

%artnet_from_NN_features Summary of this function goes here
%   Detailed explanation goes here

len = size(artnet_rec);
len = len(1,1);

artnet_rec = artnet_rec / 256;

% comment / uncomment +++++++++

% repair the missing 3rd positions, because of 2D instead 3D output of the
% neural net
nn_features_temp = nn_features;
% Color_Sat LED Beam
nn_features_temp(:,3) = nn_features(:,8);
% DIM LED Beam
nn_features_temp(:,6) = nn_features(:,1);
% PAN LED Beam
nn_features_temp(:,9) = nn_features(:,4);
% ColorHUE LED Beam
nn_features_temp(:,12) = nn_features(:,2);
% TILT LED Beam
nn_features_temp(:,15) = nn_features(:,5);
% Color SAT LED Beam
nn_features_temp(:,18) = nn_features(:,3);
% DIM LED Beam
nn_features_temp(:,21) = nn_features(:,26);
% PAN LED Beam
nn_features_temp(:,24) = nn_features(:,29);
% ColorHUE LED Beam
nn_features_temp(:,27) = nn_features(:,22);
% TILT LED Beam
nn_features_temp(:,30) = nn_features(:,25);
% Color SAT LED Beam
nn_features_temp(:,33) = nn_features(:,28);
% DIM LED Beam
nn_features_temp(:,36) = nn_features(:,31);
% PAN LED Beam
nn_features_temp(:,39) = nn_features(:,34);
% One Patt Color SAT to 1
nn_features_temp(:,42) = 1;
% ParFect Color SAT to 1
nn_features_temp(:,45) = 1;
% ColorHue of Peak Tetra 01 = Tetra 02
nn_features_temp(:,48) = nn_features(:,55);
% Tetra 01 current unused
nn_features_temp(:,51) = 0;
% Position of Peak Tetra 02 = Tetra 01
nn_features_temp(:,54) = nn_features(:,47);
% Color SAT of Peak Tetra 02 = Tetra 01
nn_features_temp(:,57) = nn_features(:,50);
% Tetra 02 current unused
nn_features_temp(:,60) = 0;
% Color SAT of Peak Tetra 03 = Tetra 04
nn_features_temp(:,63) = nn_features(:,70);
% Tetra 03 current unused
nn_features_temp(:,66) = 0;
% Color HUE of Peak Tetra 04 = Tetra 03
nn_features_temp(:,69) = nn_features(:,62);
% Tetra 04 current unused
nn_features_temp(:,72) = 0;
% Tetra 04 current unused
nn_features_temp(:,75) = 0;

% comment / uncomment +++++++++



nn_features = nn_features_temp;




% Feature Extractor LED Beam 150 #01 was
% features_v01(:,1 to 40) <= LED Beam 150
% |: 
% 1 = Dim, 2 = ColorHue, 3 = ColorSat, 4 = Pan, 5 = Tilt
% :|
% so the way back needs to be...

for i = 1 : 8
    % counter DIM source
    j = i * 5;
    j = j - 4;
    % counter DIM target
    k = i;
    k = k * 22 - 1;
    % set to artnet array DIM
    artnet_rec(:,k) = nn_features(:,j);
    % ignore LSB
    artnet_rec(:,(k+1)) = 0;
    
    % counter PAN source
    m = i * 5;
    m = m - 1;
    % counter PAN target
    l = i;
    l = l * 22 - 21;
    % set to artnet array PAN
    artnet_rec(:,l) = nn_features(:,m);
    % ignore LSB
    artnet_rec(:,(l+1)) = 0;
    
    % counter TILT source
    o = i * 5;
    o = o - 0;
    % counter TILT target
    n = i;
    n = n * 22 - 19;
    % set to artnet array PAN
    artnet_rec(:,n) = nn_features(:,o);
    % ignore LSB
    artnet_rec(:,(n+1)) = 0;
    
    % color back mapping
    % counter ColorHue / ColorSat source
    q = i * 5;
    q = q - 3;
    r = q + 1;
    % color temp store
    color_temp_hsv = zeros(len,3);
    color_temp_hsv(:,3) = 1;
    color_temp_rgb = zeros(len,3);
    % extract hue & sat
    color_temp_hsv(:,1) = nn_features(:,q);
    color_temp_hsv(:,2) = nn_features(:,r);
    color_temp_hsv(:,3) = 1;
    color_temp_rgb = hsv2rgb(color_temp_hsv);
    % counter color RGB target
    p = i;
    p = p * 22 - 14;
    % set to artnet array RGB & ignore LSB
    artnet_rec(:,p) = color_temp_rgb(:,1);
    artnet_rec(:,p+1) = 0;
    artnet_rec(:,p+2) = color_temp_rgb(:,2);
    artnet_rec(:,p+3) = 0;
    artnet_rec(:,p+4) = color_temp_rgb(:,3);
    artnet_rec(:,p+5) = 0;
    
end    

% dirty debug LED BEAM #04
artnet_rec(:,74) = artnet_rec(:,52);
artnet_rec(:,76) = artnet_rec(:,54);
artnet_rec(:,78) = artnet_rec(:,56);


% Feature Extractor Mean Value One Patts was
% features_v01(:,41 to 43) <= One Patts
% |: 
% 1 = ColorHue, 2 = ColorSat, 3 = ColorVel
% :|
% so the way back needs to be...

color_temp_hsv_onepatt_mean = zeros(len,3);
color_temp_hsv_onepatt_mean(:,1) = nn_features(:,41);
color_temp_hsv_onepatt_mean(:,2) = nn_features(:,42);
color_temp_hsv_onepatt_mean(:,3) = nn_features(:,43);
color_temp_rgb_onepatt_mean = zeros(len,3);
color_temp_rgb_onepatt_mean = hsv2rgb(color_temp_hsv_onepatt_mean);

for i = 1 : 7

    % color back mapping
    % counter OnePatt color target
    p = i;
    p = p * 4 + 173;
    % set to artnet array RGB
    artnet_rec(:,p) = color_temp_rgb_onepatt_mean(:,1);
    artnet_rec(:,p+1) = color_temp_rgb_onepatt_mean(:,2);
    artnet_rec(:,p+2) = color_temp_rgb_onepatt_mean(:,3);

end




% Feature Extractor Mean Value ParFect was
% features_v01(:,44 to 46) <= ParFect
% |: 
% 1 = ColorHue, 2 = ColorSat, 3 = ColorVel
% :|
% 1st address PARfect 205
% 17chan / ParFect
% so the way back needs to be...

color_temp_hsv_parfect_mean = zeros(len,3);
color_temp_hsv_parfect_mean(:,1) = nn_features(:,44);
color_temp_hsv_parfect_mean(:,2) = nn_features(:,45);
color_temp_hsv_parfect_mean(:,3) = nn_features(:,46);
color_temp_rgb_parfect_mean = zeros(len,3);
color_temp_rgb_parfect_mean = hsv2rgb(color_temp_hsv_parfect_mean);

for i = 1 : 8
    
    % counter rgb target
    p = i;
    p = p * 17 + 190;
    % counter dim target
    r = i;
    r = r * 17 + 203;
    % set to artnet array RGB & ignore LSB
    artnet_rec(:,p) = color_temp_rgb_parfect_mean(:,1);
    artnet_rec(:,p+1) = 0;
    artnet_rec(:,p+2) = color_temp_rgb_parfect_mean(:,2);
    artnet_rec(:,p+3) = 0;
    artnet_rec(:,p+4) = color_temp_rgb_parfect_mean(:,3);
    artnet_rec(:,p+5) = 0;
    % set to artnet DIM & ignore LSB
    artnet_rec(:,r) = color_temp_hsv_parfect_mean(:,3);
    artnet_rec(:,r+1) = 0;
   
end


% dirty debug ParFect #06
artnet_rec(:,292) = artnet_rec(:,275);
artnet_rec(:,294) = artnet_rec(:,277);
artnet_rec(:,296) = artnet_rec(:,279);
artnet_rec(:,305) = artnet_rec(:,288);




% Feature Extractor TETRA BARS was
% features_v01(:,47 to 74) <= Tetra Bars 7 chan each
% Tetra01 start 47
% Tetra02 start 54
% Tetra03 start 61
% Tetra04 start 68
% |: 
% 1 = Position 1 of 18 of MaxVel, 2 = ColorHUE of Peak, 3 = ColorSAT of Peak
% 4 = ColorVel of Peak
% ToDO
% 5 = Envelope of Peak, ...
% :|
% so the way back needs to be...

color_temp_rgb_tetra01 = zeros(len,54);
color_temp_rgb_tetra02 = zeros(len,54);
color_temp_rgb_tetra03 = zeros(len,54);
color_temp_rgb_tetra04 = zeros(len,54);

color_temp_rgb_tetra_short_01 = zeros(len,3);
color_temp_rgb_tetra_short_02 = zeros(len,3);
color_temp_rgb_tetra_short_03 = zeros(len,3);
color_temp_rgb_tetra_short_04 = zeros(len,3);

color_temp_hsv_tetra01 = zeros(len,54);
color_temp_hsv_tetra02 = zeros(len,54);
color_temp_hsv_tetra03 = zeros(len,54);
color_temp_hsv_tetra04 = zeros(len,54);

color_temp_hsv_tetra_short_01 = zeros(len,3);
pos_temp_hsv_tetra_short_01 = zeros(len,1);
color_temp_hsv_tetra_short_02 = zeros(len,3);
pos_temp_hsv_tetra_short_02 = zeros(len,1);
color_temp_hsv_tetra_short_03 = zeros(len,3);
pos_temp_hsv_tetra_short_03 = zeros(len,1);
color_temp_hsv_tetra_short_04 = zeros(len,3);
pos_temp_hsv_tetra_short_04 = zeros(len,1);
% velvel = [3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54];


% TETRA01
color_temp_hsv_tetra_short_01(:,1) = nn_features(:,48);
color_temp_hsv_tetra_short_01(:,2) = nn_features(:,49);
color_temp_hsv_tetra_short_01(:,3) = nn_features(:,50);
color_temp_rgb_tetra_short_01 = hsv2rgb(color_temp_hsv_tetra_short_01);
pos_temp_hsv_tetra_short_01 = round(nn_features(:,47)*18); 

for i = i : len
    pos = pos_temp_hsv_tetra_short_01(i,1);
    pos_red_middle = bound((pos * 3 - 2),1,52);
    % make just a little envelope over the peak. 50% to the left & right
    % to be overhauled
    pos_red_minor = bound((pos_red_middle - 3),1,52);
    pos_red_major = bound((pos_red_middle + 3),1,52);
    hsv_minor_major = zeros(1,3);
    rgb_minor_major = zeros(1,3);
    hsv_minor_major(1,1) = color_temp_hsv_tetra_short_01(i,1);
    hsv_minor_major(1,2) = color_temp_hsv_tetra_short_01(i,2);
    hsv_minor_major(1,3) = (color_temp_hsv_tetra_short_01(i,3))*0.5;
    rgb_minor_major = hsv2rgb(hsv_minor_major);
    % minor
    color_temp_rgb_tetra01(i,pos_red_minor) = rgb_minor_major(1,1);
    color_temp_rgb_tetra01(i,pos_red_minor+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra01(i,pos_red_minor+2) = rgb_minor_major(1,3);
    % major
    color_temp_rgb_tetra01(i,pos_red_major) = rgb_minor_major(1,1);
    color_temp_rgb_tetra01(i,pos_red_major+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra01(i,pos_red_major+2) = rgb_minor_major(1,3);
    % overwrite minor / major rgb values, when they were at the minima or
    % maxima
    color_temp_rgb_tetra01(i,pos_red_middle) = color_temp_rgb_tetra_short_01(i,1);
    color_temp_rgb_tetra01(i,pos_red_middle+1) = color_temp_rgb_tetra_short_01(i,2);
    color_temp_rgb_tetra01(i,pos_red_middle+2) = color_temp_rgb_tetra_short_01(i,3);
end   

for i = 1 : 18
    % counter Red target TETRA01
    p = i;
    p = p * 4 + 565;
    % counter Red source TETRA01
    q = i * 3;
    q = q - 2;
    % set to artnet array RGB
    artnet_rec(:,p) = color_temp_rgb_tetra01(:,q);
    artnet_rec(:,p+1) = color_temp_rgb_tetra01(:,q+1);
    artnet_rec(:,p+2) = color_temp_rgb_tetra01(:,q+2);
end


% TETRA02
color_temp_hsv_tetra_short_02(:,1) = nn_features(:,55);
color_temp_hsv_tetra_short_02(:,2) = nn_features(:,56);
color_temp_hsv_tetra_short_02(:,3) = nn_features(:,57);
color_temp_rgb_tetra_short_02 = hsv2rgb(color_temp_hsv_tetra_short_02);
pos_temp_hsv_tetra_short_02 = round(nn_features(:,54)*18); 

for i = i : len
    pos = pos_temp_hsv_tetra_short_02(i,1);
    pos_red_middle = bound((pos * 3 - 2),1,52);
    % make just a little envelope over the peak. 50%DIM to the left & right
    % to be overhauled
    pos_red_minor = bound((pos_red_middle - 3),1,52);
    pos_red_major = bound((pos_red_middle + 3),1,52);
    
    hsv_minor_major = zeros(1,3);
    rgb_minor_major = zeros(1,3);
    hsv_minor_major(1,1) = color_temp_hsv_tetra_short_02(i,1);
    hsv_minor_major(1,2) = color_temp_hsv_tetra_short_02(i,2);
    hsv_minor_major(1,3) = (color_temp_hsv_tetra_short_02(i,3))*0.5;
    rgb_minor_major = hsv2rgb(hsv_minor_major);
    % minor
    color_temp_rgb_tetra02(i,pos_red_minor) = rgb_minor_major(1,1);
    color_temp_rgb_tetra02(i,pos_red_minor+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra02(i,pos_red_minor+2) = rgb_minor_major(1,3);
    % major
    color_temp_rgb_tetra02(i,pos_red_major) = rgb_minor_major(1,1);
    color_temp_rgb_tetra02(i,pos_red_major+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra02(i,pos_red_major+2) = rgb_minor_major(1,3);
    % overwrite minor / major rgb values, when they were at the minima or
    % maxima
    color_temp_rgb_tetra02(i,pos_red_middle) = color_temp_rgb_tetra_short_02(i,1);
    color_temp_rgb_tetra02(i,pos_red_middle+1) = color_temp_rgb_tetra_short_02(i,2);
    color_temp_rgb_tetra02(i,pos_red_middle+2) = color_temp_rgb_tetra_short_02(i,3);
end   

for i = 1 : 18
    % counter Red target TETRA02
    p = i;
    p = p * 4 + 693;
    % counter Red source TETRA02
    q = i * 3;
    q = q - 2;
    % set to artnet array RGB
    artnet_rec(:,p) = color_temp_rgb_tetra02(:,q);
    artnet_rec(:,p+1) = color_temp_rgb_tetra02(:,q+1);
    artnet_rec(:,p+2) = color_temp_rgb_tetra02(:,q+2);
end


% TETRA03
color_temp_hsv_tetra_short_03(:,1) = nn_features(:,62);
color_temp_hsv_tetra_short_03(:,2) = nn_features(:,63);
color_temp_hsv_tetra_short_03(:,3) = nn_features(:,64);
color_temp_rgb_tetra_short_03 = hsv2rgb(color_temp_hsv_tetra_short_03);
pos_temp_hsv_tetra_short_03 = round(nn_features(:,61)*18); 

for i = i : len
    pos = pos_temp_hsv_tetra_short_03(i,1);
    pos_red_middle = bound((pos * 3 - 2),1,52);
    % make just a little envelope over the peak. 50%DIM to the left & right
    % to be overhauled
    pos_red_minor = bound((pos_red_middle - 3),1,52);
    pos_red_major = bound((pos_red_middle + 3),1,52);
    
    hsv_minor_major = zeros(1,3);
    rgb_minor_major = zeros(1,3);
    hsv_minor_major(1,1) = color_temp_hsv_tetra_short_03(i,1);
    hsv_minor_major(1,2) = color_temp_hsv_tetra_short_03(i,2);
    hsv_minor_major(1,3) = (color_temp_hsv_tetra_short_03(i,3))*0.5;
    rgb_minor_major = hsv2rgb(hsv_minor_major);
    % minor
    color_temp_rgb_tetra03(i,pos_red_minor) = rgb_minor_major(1,1);
    color_temp_rgb_tetra03(i,pos_red_minor+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra03(i,pos_red_minor+2) = rgb_minor_major(1,3);
    % major
    color_temp_rgb_tetra03(i,pos_red_major) = rgb_minor_major(1,1);
    color_temp_rgb_tetra03(i,pos_red_major+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra03(i,pos_red_major+2) = rgb_minor_major(1,3);
    % overwrite minor / major rgb values, when they were at the minima or
    % maxima
    color_temp_rgb_tetra03(i,pos_red_middle) = color_temp_rgb_tetra_short_03(i,1);
    color_temp_rgb_tetra03(i,pos_red_middle+1) = color_temp_rgb_tetra_short_03(i,2);
    color_temp_rgb_tetra03(i,pos_red_middle+2) = color_temp_rgb_tetra_short_03(i,3);
end   

for i = 1 : 18
    % counter Red target TETRA03
    p = i;
    p = p * 4 + 821;
    % counter Red source TETRA03
    q = i * 3;
    q = q - 2;
    % set to artnet array RGB
    artnet_rec(:,p) = color_temp_rgb_tetra03(:,q);
    artnet_rec(:,p+1) = color_temp_rgb_tetra03(:,q+1);
    artnet_rec(:,p+2) = color_temp_rgb_tetra03(:,q+2);
end


% TETRA04
color_temp_hsv_tetra_short_04(:,1) = nn_features(:,69);
color_temp_hsv_tetra_short_04(:,2) = nn_features(:,70);
color_temp_hsv_tetra_short_04(:,3) = nn_features(:,71);
color_temp_rgb_tetra_short_04 = hsv2rgb(color_temp_hsv_tetra_short_04);
pos_temp_hsv_tetra_short_04 = round(nn_features(:,68)*18); 

for i = i : len
    pos = pos_temp_hsv_tetra_short_04(i,1);
    pos_red_middle = bound((pos * 3 - 2),1,52);
    % make just a little envelope over the peak. 50%DIM to the left & right
    % to be overhauled
    pos_red_minor = bound((pos_red_middle - 3),1,52);
    pos_red_major = bound((pos_red_middle + 3),1,52);
    
    hsv_minor_major = zeros(1,3);
    rgb_minor_major = zeros(1,3);
    hsv_minor_major(1,1) = color_temp_hsv_tetra_short_04(i,1);
    hsv_minor_major(1,2) = color_temp_hsv_tetra_short_04(i,2);
    hsv_minor_major(1,3) = (color_temp_hsv_tetra_short_04(i,3))*0.5;
    rgb_minor_major = hsv2rgb(hsv_minor_major);
    % minor
    color_temp_rgb_tetra04(i,pos_red_minor) = rgb_minor_major(1,1);
    color_temp_rgb_tetra04(i,pos_red_minor+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra04(i,pos_red_minor+2) = rgb_minor_major(1,3);
    % major
    color_temp_rgb_tetra04(i,pos_red_major) = rgb_minor_major(1,1);
    color_temp_rgb_tetra04(i,pos_red_major+1) = rgb_minor_major(1,2);
    color_temp_rgb_tetra04(i,pos_red_major+2) = rgb_minor_major(1,3);
    % overwrite minor / major rgb values, 
    % when they were at the minima or maxima
    color_temp_rgb_tetra04(i,pos_red_middle) = color_temp_rgb_tetra_short_04(i,1);
    color_temp_rgb_tetra04(i,pos_red_middle+1) = color_temp_rgb_tetra_short_04(i,2);
    color_temp_rgb_tetra04(i,pos_red_middle+2) = color_temp_rgb_tetra_short_04(i,3);
end   

for i = 1 : 18
    % counter Red target TETRA04
    p = i;
    p = p * 4 + 949;
    % counter Red source TETRA04
    q = i * 3;
    q = q - 2;
    % set to artnet array RGB
    artnet_rec(:,p) = color_temp_rgb_tetra04(:,q);
    artnet_rec(:,p+1) = color_temp_rgb_tetra04(:,q+1);
    artnet_rec(:,p+2) = color_temp_rgb_tetra04(:,q+2);
end


artnet_nn_features = artnet_rec;

end





function [X_BG,X_FG,mask,Features64] = Data()

load('TrainingSamplesDCT_8_new.mat')
mask = im2double(imread('cheetah_mask.bmp'));
img = im2double(imread('cheetah.bmp'));
ZZag = importdata('Zig-Zag Pattern.txt');
X_BG = TrainsampleDCT_BG;
X_FG = TrainsampleDCT_FG;

%Produce Feature Vectors to Test
imgc = [img(:,1:268) img(:,268) img(:,268)];
imgp = padarray(imgc,[7,7],'replicate','post');

DV = zeros(1,64);
Features64 = cell(size(img));
for i = 1:size(imgp,1)-7
   for j = 1:size(imgp,2)-7
      F = imgp(i:i+7,j:j+7);
      D = dct2(F);
      DV(ZZag + 1) = D;
      
      Features64{i,j} = DV';
   end
end  

disp('Data Loaded')

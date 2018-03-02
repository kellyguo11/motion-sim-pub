files = dir('C:\Users\kelly\Desktop\Data\*_skel_k2.mat');

for i = 1:size(files)
    file = files(i);
    path = strcat(file.folder, '\', file.name);
    matFile = matfile(path);
    
    content = matFile.S_K2;
    posList = content.world;
    
    outputFile = fopen(strcat(path, '.txt'), 'w');
    dim = size(posList);
    for joints = 1:dim(1)
       for frames = 1:dim(3)
           for coords = 1:dim(2)
                fprintf(outputFile, '%d', posList(joints, coords, frames));
                if coords ~= dim(2)
                    fprintf(outputFile, ' ');
                end
           end
           if frames ~= dim(3)
               fprintf(outputFile, ',');
           end
       end
       fprintf(outputFile, '\n');
    end
end